# Stage 1: Clone & build nixcat-dev/adamant for NVIDIA Ampere (sm_86, RTX 3060)
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04 AS builder

ARG CUDA_ARCH=86
ARG ADAMANT_REPO=https://github.com/nixcat-dev/adamant.git
ARG ADAMANT_REF=main
# PR #30 fixes the RTX 3060 (28 SM) gdn_commit_lazy verify deadlock (Issue #14).
# Set ADAMANT_PR="" once PR #30 is merged into main.
ARG ADAMANT_PR=30

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build g++ git ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone "${ADAMANT_REPO}" . \
    && git checkout "${ADAMANT_REF}" \
    && if [ -n "${ADAMANT_PR}" ]; then \
         git fetch origin "pull/${ADAMANT_PR}/head:pr-${ADAMANT_PR}" \
         && git checkout "pr-${ADAMANT_PR}"; \
       fi

RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
    && cmake --build build -j"$(nproc)"

# Stage 2: Runtime image
FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /src/bin /app/bin
COPY --from=builder /src/serve /app/serve
COPY --from=builder /src/tools /app/tools
COPY --from=builder /src/data /app/data
COPY --from=builder /src/docs /app/docs
COPY --from=builder /src/requirements.txt /app/requirements.txt
COPY --from=builder /src/LICENSE /src/THIRD_PARTY_LICENSES /app/

RUN python3 -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8080

ENTRYPOINT ["/app/bin/basalt"]
CMD ["serve", "--config", "/app/basalt_config.json"]
