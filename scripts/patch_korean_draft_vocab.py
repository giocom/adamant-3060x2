#!/usr/bin/env python3
"""Standalone in-place Korean Draft Vocab patcher for .basalt v2 model files.

- Zero external dependencies (Python 3 standard library only).
- Promotes all 6,807 Hangul tokens into Tier 1 (base, 2-bit fast proxy) and
  removes Tier 2 (55,301 Chinese/Hanja + 3,503 Kana) so Korean drafting does not
  waste time scanning ~59K non-Korean Q8_0 rows on every draft step.
- Exact speculative decoding verify still covers the full 248,320-token head,
  so output quality and valid Hanja/Japanese output are never broken.
- Automatically backs up the original 64 KiB header and mtp/draft_vocab section
  (~490 KiB total) so `--restore` reverts the file byte-for-byte in <0.1 s.

Usage:
  # Inspect current draft_vocab coverage (read-only)
  python3 scripts/patch_korean_draft_vocab.py --inspect

  # Apply Korean Draft Vocab patch (stop container first: docker compose down)
  python3 scripts/patch_korean_draft_vocab.py

  # Restore original draft_vocab byte-for-byte
  python3 scripts/patch_korean_draft_vocab.py --restore
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys

MAGIC = b"BASALTMF"
VERSION = 2
SECTION_ALIGN = 2 << 20
HEADER_BYTES = 64 << 10
HEADER = struct.Struct("<8sIIQ40s")      # 64 bytes
ENTRY = struct.Struct("<64sQQ32s16s")    # 128 bytes


def parse_header(head: bytes) -> dict[str, tuple[int, int, int, bytes]]:
    """Return {section_name: (entry_pos, offset, size, sha256_bytes)}."""
    magic, version, n, _file_bytes, _ = HEADER.unpack_from(head, 0)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"Unsupported .basalt header: magic={magic!r}, version={version}")
    sections: dict[str, tuple[int, int, int, bytes]] = {}
    for i in range(n):
        pos = HEADER.size + i * ENTRY.size
        raw_name, off, size, digest, _ = ENTRY.unpack_from(head, pos)
        name = raw_name.split(b"\0", 1)[0].decode("utf-8")
        sections[name] = (pos, off, size, digest)
    return sections


def read_section(f, sections, name: str) -> bytes:
    _, off, size, _ = sections[name]
    f.seek(off)
    data = f.read(size)
    if len(data) != size:
        raise IOError(f"Short read on section {name!r}: got {len(data)}, expected {size}")
    return data


def verify_section(f, sections, name: str) -> bool:
    _, _, _, expected = sections[name]
    actual = hashlib.sha256(read_section(f, sections, name)).digest()
    return actual == expected


def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(x) for x in cs]))


def decode_bpe_token(tok: str, u2b: dict[str, int]) -> str:
    try:
        return bytes(u2b[c] for c in tok).decode("utf-8", errors="replace")
    except Exception:
        return tok


def is_hangul(s: str) -> bool:
    return any("\uac00" <= c <= "\ud7a3" or "\u3131" <= c <= "\u318e" for c in s)


def split_tiers(ids: tuple[int, ...]) -> tuple[int, set[int], set[int]]:
    n_base = len(ids)
    for i in range(1, len(ids)):
        if ids[i] < ids[i - 1]:
            n_base = i
            break
    return n_base, set(ids[:n_base]), set(ids[n_base:])


def inspect_model(model_path: str) -> None:
    with open(model_path, "rb") as f:
        sections = parse_header(f.read(HEADER_BYTES))
        _, dv_off, dv_size, dv_sha = sections["mtp/draft_vocab"]
        next_offs = sorted(v[1] for v in sections.values() if v[1] > dv_off)
        gap = (next_offs[0] - dv_off) if next_offs else SECTION_ALIGN
        dv = read_section(f, sections, "mtp/draft_vocab")
        vocab = json.loads(read_section(f, sections, "tokenizer/vocab.json").decode("utf-8"))
        ok = hashlib.sha256(dv).digest() == dv_sha

    ids = struct.unpack(f"<{len(dv) // 4}i", dv)
    n_base, base_set, added_set = split_tiers(ids)
    u2b = {v: k for k, v in bytes_to_unicode().items()}

    cats = {k: [0, 0, 0, 0] for k in ("hangul", "cjk", "kana", "ascii", "other")}
    for tok, tid in vocab.items():
        t = decode_bpe_token(tok, u2b)
        if is_hangul(t):
            k = "hangul"
        elif any("\u4e00" <= c <= "\u9fff" for c in t):
            k = "cjk"
        elif any("\u3040" <= c <= "\u30ff" for c in t):
            k = "kana"
        elif all(ord(c) < 128 for c in t):
            k = "ascii"
        else:
            k = "other"
        cats[k][0] += 1
        if tid in base_set:
            cats[k][1] += 1
        elif tid in added_set:
            cats[k][2] += 1
        else:
            cats[k][3] += 1

    print(f"Model:       {model_path}")
    print(f"Section:     mtp/draft_vocab off={dv_off} size={dv_size} B (gap={gap} B, sha256_ok={ok})")
    print(f"Draft vocab: {len(ids)} tokens (base={n_base}, added={len(ids) - n_base})\n")
    print(f"{'Category':10s} {'Total':>8s} {'Base':>8s} {'Added':>8s} {'Missing':>8s}")
    for k, (tot, b, a, ms) in cats.items():
        print(f"{k:10s} {tot:8d} {b:8d} {a:8d} {ms:8d}")


def apply_patch(model_path: str, backup_dir: str) -> None:
    os.makedirs(backup_dir, exist_ok=True)
    hdr_bak = os.path.join(backup_dir, "header.bin")
    dv_bak = os.path.join(backup_dir, "orig_draft_vocab.bin")

    with open(model_path, "rb") as f:
        cur_head = f.read(HEADER_BYTES)
        sections = parse_header(cur_head)
        entry_pos, dv_off, old_size, _ = sections["mtp/draft_vocab"]
        next_offs = sorted(v[1] for v in sections.values() if v[1] > dv_off)
        avail_gap = (next_offs[0] - dv_off) if next_offs else SECTION_ALIGN
        vocab = json.loads(read_section(f, sections, "tokenizer/vocab.json").decode("utf-8"))

        if not os.path.exists(hdr_bak):
            orig_dv = read_section(f, sections, "mtp/draft_vocab")
            with open(hdr_bak, "wb") as bf:
                bf.write(cur_head)
            with open(dv_bak, "wb") as bf:
                bf.write(orig_dv)
            print(f"Backed up original header ({len(cur_head)} B) and draft_vocab ({len(orig_dv)} B) -> {backup_dir}")
        else:
            print(f"Using existing backup in {backup_dir}")
            with open(dv_bak, "rb") as bf:
                orig_dv = bf.read()

    orig_ids = struct.unpack(f"<{len(orig_dv) // 4}i", orig_dv)
    n_base, base_ids, _ = split_tiers(orig_ids)

    u2b = {v: k for k, v in bytes_to_unicode().items()}
    hangul_ids = {tid for tok, tid in vocab.items() if is_hangul(decode_bpe_token(tok, u2b))}

    new_ids = sorted(base_ids | hangul_ids)
    new_dv = struct.pack(f"<{len(new_ids)}i", *new_ids)
    if len(new_dv) > avail_gap:
        raise ValueError(f"New draft_vocab ({len(new_dv)} B) exceeds available gap ({avail_gap} B)")
    new_sha = hashlib.sha256(new_dv).digest()

    with open(model_path, "r+b") as f:
        head = bytearray(f.read(HEADER_BYTES))
        f.seek(dv_off)
        f.write(new_dv)
        if old_size > len(new_dv):
            f.write(b"\0" * (old_size - len(new_dv)))
        f.flush()
        os.fsync(f.fileno())

        ENTRY.pack_into(head, entry_pos, b"mtp/draft_vocab", dv_off, len(new_dv), new_sha, b"\0" * 16)
        f.seek(0)
        f.write(head)
        f.flush()
        os.fsync(f.fileno())

    with open(model_path, "rb") as f:
        sec2 = parse_header(f.read(HEADER_BYTES))
        if not verify_section(f, sec2, "mtp/draft_vocab"):
            raise RuntimeError("SHA-256 verification failed after patching!")

    print("Patched mtp/draft_vocab successfully!")
    print(f"  Original: {len(orig_ids)} tokens (base={n_base}, added={len(orig_ids) - n_base})")
    print(f"  New:      {len(new_ids)} tokens (base={len(new_ids)}, added=0; includes all {len(hangul_ids)} Hangul tokens)")
    print(f"  SHA-256:  {new_sha.hex()[:16]}... (verified OK)")


def restore_patch(model_path: str, backup_dir: str) -> None:
    hdr_bak = os.path.join(backup_dir, "header.bin")
    dv_bak = os.path.join(backup_dir, "orig_draft_vocab.bin")
    if not (os.path.exists(hdr_bak) and os.path.exists(dv_bak)):
        raise FileNotFoundError(f"Backup files not found in {backup_dir}")

    with open(hdr_bak, "rb") as f:
        orig_head = f.read()
    with open(dv_bak, "rb") as f:
        orig_dv = f.read()

    sections = parse_header(orig_head)
    _, off, size, _ = sections["mtp/draft_vocab"]
    if len(orig_dv) != size:
        raise ValueError(f"Backup size mismatch: {len(orig_dv)} vs {size}")

    with open(model_path, "r+b") as f:
        f.seek(off)
        f.write(orig_dv)
        f.flush()
        os.fsync(f.fileno())
        f.seek(0)
        f.write(orig_head)
        f.flush()
        os.fsync(f.fileno())

    with open(model_path, "rb") as f:
        sec2 = parse_header(f.read(HEADER_BYTES))
        if not verify_section(f, sec2, "mtp/draft_vocab"):
            raise RuntimeError("SHA-256 verification failed after restore!")

    print(f"Restored original mtp/draft_vocab ({len(orig_dv) // 4} tokens, verified OK)")


def main() -> int:
    ap = argparse.ArgumentParser(description="In-place Korean Draft Vocab patcher for .basalt v2 files")
    ap.add_argument("--model", default="./models/Qwen3.8-FN-NVFP4-DQ6_K.basalt", help="Path to .basalt model file")
    ap.add_argument("--backup-dir", default="./models/draft_vocab_backup", help="Backup directory for header & original draft_vocab")
    ap.add_argument("--inspect", action="store_true", help="Inspect current draft_vocab token breakdown (read-only)")
    ap.add_argument("--restore", action="store_true", help="Restore original draft_vocab from backup")
    args = ap.parse_args()

    if args.inspect:
        inspect_model(args.model)
    elif args.restore:
        restore_patch(args.model, args.backup_dir)
    else:
        apply_patch(args.model, args.backup_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
