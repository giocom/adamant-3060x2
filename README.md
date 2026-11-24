# adamant-3060x2

`nixcat-dev/adamant`를 **NVIDIA RTX 3060 12GB × 2 (`sm_86`, Ampere)** 환경에서 **Docker Compose**로 빌드·벤치마크·서빙하기 위한 배포 구성 및 상세 설정 가이드입니다.

- **타깃 모델**: `Qwen3.8-Flash-Next` (`Qwen3.8-FN-NVFP4-DQ6_K.basalt`, 122.6 GiB)
- **기본 설정**: GPU 2장(`0,1`), `int8` KV, 128k 컨텍스트(`131,072`), 단일 요청(`parallel: 1`), `vision: "dynamic"` (SSD 스필 0%)
- **핵심 패치 포함**: `Dockerfile` 빌드 시 RTX 3060(28 SM) 데드락을 해결하는 [PR #30](https://github.com/nixcat-dev/adamant/pull/30)([Issue #14](https://github.com/nixcat-dev/adamant/issues/14))을 자동 적용

---

## 1. Quick Start (처음부터 끝까지)

```bash
# 1) 이 저장소 클론
git clone git@github.com:giocom/adamant-3060x2.git
cd adamant-3060x2

# 2) 디렉토리 생성 및 API 키 설정
mkdir -p models cache
cp .env.example .env
nano .env              # ADAMANT_API_KEY 를 원하는 값으로 수정
chmod 600 .env

# 3) 모델 다운로드 (122.6 GiB, 중단 시 같은 명령으로 이어받기 가능)
aria2c -x 16 -s 16 -k 8M \
  --file-allocation=none \
  --continue=true \
  --summary-interval=30 \
  -d ./models \
  -o Qwen3.8-FN-NVFP4-DQ6_K.basalt \
  'https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt/resolve/main/Qwen3.8-FN-NVFP4-DQ6_K.basalt'

# 4) Docker 이미지 빌드 (CUDA 13.0.3 + sm_86 + PR #30 자동 적용, 약 5~10분)
docker compose build

# 5) 서버 기동
docker compose up -d

# 6) 로딩 로그 확인 (약 45~50초 소요)
docker compose logs -f
```

로딩이 끝나면:
- **API 엔드포인트 (OpenAI / Anthropic 호환)**: `http://<서버IP>:8080/v1`
- **실시간 모니터링 대시보드**: `http://<서버IP>:8080/`

---

## 2. 사전 요구사항 점검

설치 전 서버에서 아래 항목을 먼저 확인하세요.

```bash
# 1) GPU 및 NVIDIA 드라이버 (R580 이상 / CUDA 13.0+ 필수)
nvidia-smi

# 2) 시스템 RAM (NVFP4 기준 가용 RAM 70 GiB 이상 권장, 128 GB 시 256k/동시요청 2 여유)
free -h

# 3) 디스크 여유 공간 (모델 122.6 GiB + 빌드 여유 = 최소 140 GiB 이상)
df -h

# 4) Docker Compose 및 NVIDIA Container Runtime 확인
docker compose version
docker info --format '{{.Runtimes}}' | grep nvidia
```

> [!NOTE]
> **Windows 대신 Linux 권장**: Windows(WDDM)는 페이지 고정(pinned) 메모리가 `RAM/2 - 8 GiB`로 제한될 수 있어 64~78 GB RAM 환경에서는 SSD 스필이 크게 발생합니다.

---

## 3. 모델 팩 선택 가이드 (파일 크기 vs 실제 메모리 점유)

`.basalt` 파일은 **① VRAM+RAM에 올라가는 전문가(`Experts`)**와 **② 디스크(SSD)에서 필요 행만 읽는 Engram/PLE 테이블**로 나뉩니다. 따라서 **디스크 용량과 실제 메모리(RAM) 점유량은 다릅니다.**

| 모델 팩 | 디스크 크기 | **전문가(VRAM+RAM)** | KLD (낮을수록 고품질) | Top-1 일치율 | MTP Drafter | 특징 |
|---|---:|---:|---:|---:|---|---|
| `UD-IQ4_XS` | 90.9 GiB | **56.2 GiB** | 0.0818 | 90.52% | Self-distilled | RAM 여유 가장 넉넉, 속도 가장 빠름 |
| **`NVFP4-DQ6_K` (기본 추천)** | **122.0 GiB** | **63.3 GiB** | **0.0624** | **91.65%** | **Self-distilled** | **품질·속도·RAM 최고 밸런스** |
| `UD-Q4_K_XL` | 106.7 GiB | **71.7 GiB** | 0.0496 | 92.63% | 원본 MTP | 품질 최상이나 RAM 더 먹고 속도 느림 |
| `UD-Q5_K_XL` | 150.5 GiB | **91.6 GiB** | 0.0255 | 94.82% | 원본 MTP | **128 GB RAM 증설 시** 0 SSD spill 가능 |

- `NVFP4-DQ6_K`는 디스크 크기가 `UD-Q4_K_XL`보다 15 GiB 크지만, 그 차이는 디스크에 머무는 고정밀 Engram 테이블 때문이며 **실제 RAM 점유는 `NVFP4-DQ6_K`가 8.4 GiB 덜 먹습니다.**

---

## 4. `basalt_config.json` 상세 옵션 해설 (직접 튜닝 가이드)

설정 변경 후에는 `docker compose restart`만 실행하면 바로 반영됩니다(이미지 재빌드 불필요).

### 4-1. 동시요청(`parallel`)과 컨텍스트(`max-context`)
- **`max-context`는 `parallel` 슬롯 수로 균등 분할**됩니다.
  - `"parallel": 1, "max-context": 131072` → 1슬롯 × **128k**
  - `"parallel": 2, "max-context": 262144` → 2슬롯 × **128k** (슬롯당 128k)
  - `"parallel": 1, "max-context": 262144` → 1슬롯 × **256k** (모델 네이티브 최대 길이)
- Qwen3.8-Flash-Next는 48레이어 중 36개가 선형 어텐션(GDN)이고 12개만 일반 어텐션(QSA)이라 KV 캐시가 작습니다 (`int8` 기준 128k 최대 ~2.1 GiB, 256k 최대 ~4.1 GiB).

### 4-2. Vision 설정 (`vision`)

| 값 | 동작 방식 | 대기 VRAM | Pinned RAM | 인코딩 속도 | 추천 상황 |
|---|---|---:|---:|---|---|
| **`"dynamic"` (기본)** | 요청 시 GPU 0 캐시를 잠시(~70 ms) 빌려 인코딩 후 복원 | **0 GiB** | **0.85 GiB** | 매우 빠름 | **가장 추천** |
| `"cpu"` | CPU 사이드카(`basalt-vision`)에서 인코딩 | **0 GiB** | **0 GiB** | 느림 (수 초) | RAM 여유가 극도로 부족할 때 |
| `1` (GPU 번호) | GPU 1에 사이드카 상주 | **~2.2 GiB** | 0 GiB | 매우 빠름 | 이미지를 매우 자주 보낼 때 |
| `"off"` | 비활성화 | 0 GiB | 0 GiB | - | 텍스트 전용 |

### 4-3. `peer-exclusive` (RAM 절약 옵션)
- **기본값 (`false`)**: GPU 1에 올라간 전문가(~10.5 GiB)를 시스템 RAM에도 중복 보관합니다. 덕분에 멀티 GPU 초장문 프리필 파이프라인(`prompt split`)과 빠른 전문가 스왑이 가능합니다.
- **`"peer-exclusive": true`** (`engine` 블록): GPU 1에 올라간 전문가를 **시스템 RAM에서 제외**해 **RAM을 ~10.5 GiB 절약**합니다.
  - 서버 로그에 `experts on SSD` 경고가 뜰 때 켜면 **SSD 스필을 0%**로 만들 수 있습니다.
  - 트레이드오프: 초장문 멀티 GPU `prompt split`이 꺼집니다(일반 프리필/디코드 2장 분담은 유지).

### 4-4. `kv` 및 `kv-resident` (장문 256k / 다중 슬롯 최적화)
- `kv`: `"int8"` 추천 (`fp16` 대비 VRAM 절반, BF16 대비 품질 손실 사실상 0). VRAM이 더 급할 때는 `"k8v4"` 또는 `"q4_0"`.
- **`"kv-resident": 32768`** (`engine` 블록, 최소값 `20480`, `kv: "int8"` 또는 `"fp16"`과 사용):
  - **32k 이하 대화**: 전부 VRAM에서 처리 (속도 손실 0).
  - **32k 초과 대화**: GPU 0 VRAM에는 32k만 유지하고, 전체 KV 캐시(예: 256k ≈ 4 GiB)는 시스템 RAM에 두고 PCIe로 스트리밍합니다.
  - **언제 켜는가?** 12 GB GPU 0에서 256k 기본 Elastic KV를 쓰면 GPU 0 Expert 캐시(~4.8 GiB) 중 **~3.4 GiB를 KV에 뺏겨** 48레이어 전체 미스가 늘어나므로, **128 GB RAM 환경에서 150k~256k를 자주 쓰거나 `parallel: 2 × 256k`를 쓸 때** 켜는 것이 유리합니다.

### 4-5. RAM 용량별 추천 프리셋 요약

| 서버 RAM | 추천 조합 | `basalt_config.json` 핵심 값 |
|---|---|---|
| **78 GB (현재)** | 단일 128k + 빠른 Vision + SSD 스필 0% | `parallel: 1`, `max-context: 131072`, `vision: "dynamic"` *(현재 기본값)* |
| **78 GB** | 동시요청 2 × 128k + SSD 스필 0% | `parallel: 2`, `max-context: 262144`, `vision: "dynamic"`, `engine.peer-exclusive: true` |
| **128 GB** | 단일 256k (네이티브 최대) + 빠른 Vision | `parallel: 1`, `max-context: 262144`, `vision: "dynamic"`, `engine.kv-resident: 32768` |
| **128 GB** | 동시요청 2 × 128k + `prompt split` 유지 | `parallel: 2`, `max-context: 262144`, `vision: "dynamic"` (`peer-exclusive` 불필요) |
| **128 GB** | 동시요청 2 × 256k (합계 512k) | `parallel: 2`, `max-context: 524288`, `vision: "dynamic"`, `engine.kv-resident: 32768` |

---

## 5. `Dockerfile` 및 `PR #30` 빌드 참고

- RTX 3060(Ampere, `sm_86`, 28 SM)에서 `adamant` `main` 브랜치는 `gdn_commit_lazy` 커널의 SM 점유로 인해 추론 시 Layer 0~2 타임아웃이 발생합니다([Issue #14](https://github.com/nixcat-dev/adamant/issues/14)).
- 현재 `Dockerfile`과 `docker-compose.yml`은 `ADAMANT_PR=30`을 기본으로 빌드해 이 문제를 해결합니다.
- **추후 PR #30이 `main`에 머지되거나 최신 버전으로 업데이트할 때**:
  ```bash
  # docker-compose.yml에서 ADAMANT_PR: "" 로 변경 후 재빌드
  docker compose build --no-cache
  docker compose up -d
  ```

---

## 6. 성능 벤치마크 테스트 (`basalt-bench`)

서버 컨테이너를 잠깐 내린 뒤(`docker compose down`) GPU가 비어 있는 상태에서 실행합니다.

```bash
# GPU 인식 및 CPU/RAM 대역폭 확인
docker compose run --rm --entrypoint nvidia-smi adamant -L
docker compose run --rm --entrypoint /app/bin/basalt-bench adamant --hardware

# 프리필(16k, 64k) + 디코드(conv, struct, prose) 측정
docker compose run --rm --entrypoint /app/bin/basalt-bench adamant \
  --engine /app/bin/basalt \
  --fixtures /app/data/bench \
  --model /app/models/Qwen3.8-FN-NVFP4-DQ6_K.basalt \
  --gpu 0,1 --kv int8 \
  -p 16k,64k -t conv,struct,prose
```

### 실측 결과 (RTX 3060 12GB × 2 @ 170W, i9-10920X 10코어, 78 GiB RAM, PCIe 3.0 ×16)
- 로딩: **48초**, **SSD 스필 0%** (`cuda0 9.8 GiB`, `RAM 58.0 GiB`, `cuda1 12.0 GiB`)

| 테스트 항목 | 실측 속도 (`t/s`) | Draft 수용률 | 평균 Draft 길이 |
|---|---:|---:|---:|
| **Prefill 16k (`pp16384`)** | **1,503 tok/s** | - | - |
| **Prefill 64k (`pp65536`)** | **2,099 tok/s** | - | - |
| **Decode 일반 대화 (`tg conv`)** | **53.4 tok/s** | 72.5% | 2.36 |
| **Decode 산문 (`tg prose`)** | **58.0 tok/s** | 69.3% | 1.78 |
| **Decode 코드/구조화 (`tg struct`)** | **104.0 tok/s** | 97.9% | 6.91 |

---

## 7. API 동작 검증 및 LiteLLM 연동

### 헬스체크 & 모델 목록
```bash
curl -s http://127.0.0.1:8080/health

source .env
curl -s -H "Authorization: Bearer ${ADAMANT_API_KEY}" \
  http://127.0.0.1:8080/v1/models
```

### 텍스트 추론 테스트
```bash
source .env
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer ${ADAMANT_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "안녕하세요! 짧게 한 문장으로 인사해 주세요."}],
    "max_tokens": 64
  }'
```

### Vision 이미지 추론 테스트
```bash
source .env
python3 - <<'PY'
import base64, json, os, struct, urllib.request, zlib

def make_png(w, h, rgb):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")

b64 = base64.b64encode(make_png(64, 64, (255, 0, 0))).decode()
req = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps({
        "model": "qwen3.8-flash-next",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image? Answer in one word."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }],
        "max_tokens": 128
    }).encode(),
    headers={
        "Authorization": f"Bearer {os.environ['ADAMANT_API_KEY']}",
        "Content-Type": "application/json"
    }
)
with urllib.request.urlopen(req, timeout=60) as r:
    print(r.read().decode())
PY
```

### LiteLLM 설정 예시
```yaml
model_list:
  - model_name: qwen3.8-flash-next
    litellm_params:
      model: openai/qwen3.8-flash-next
      api_base: http://<서버IP>:8080/v1
      api_key: <ADAMANT_API_KEY>
```

---

## 8. 운영 명령어 & 트러블슈팅

```bash
docker compose ps          # 컨테이너 상태
docker compose logs -f     # 실시간 추론/속도 로그
docker compose restart     # basalt_config.json 변경 후 재적용
docker compose down        # 중지 및 제거
```

- **초장문(128k~256k) 프리필 시 VRAM OOM 발생**: `basalt_config.json`의 `engine.prefill`을 `"auto"`에서 `"8192"` 또는 `"4096"`으로 낮춰 스크래치 버퍼를 줄이세요.
- **로그에 `experts on SSD` 경고가 뜰 때**: `engine`에 `"peer-exclusive": true`를 추가하거나 `parallel`을 `1`로 낮추세요.
- **컨테이너 내부 pinned RAM 할당 실패**: `docker-compose.yml`에 `ipc: host`와 `ulimits.memlock: -1`이 유지되어 있는지 확인하세요.
