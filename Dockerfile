# ─── Build stage ─────────────────────────────────────────────────────────────
# CUDA 12.6 devel image — compatible with any host driver ≥ 525 (RTX 4070 Ti uses 595.x).
# To upgrade: replace 12.6.0 with a newer tag from hub.docker.com/r/nvidia/cuda.
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# SM 89 = Ada Lovelace (RTX 4070 Ti). Targeting one arch keeps the binary lean
# and avoids the multi-arch PTX JIT overhead at pod startup.
#
# The devel image has libcuda.so only as a stub — libcuda.so.1 doesn't exist.
# Registering it via ldconfig makes it visible to the linker for all targets,
# including transitive deps like libggml-cuda.so. The stub is build-only;
# the real libcuda.so.1 is injected at runtime by the NVIDIA container runtime.
RUN cp /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/lib/libcuda.so.1 \
    && ldconfig \
    && cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DLLAMA_BUILD_WEBUI=OFF \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
    && cmake --build build --target llama-server -j"$(nproc)"

# ─── Runtime stage ───────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# libgomp1: OpenMP threading used by GGML CPU fallback paths.
# libcublas is already present (held) in the nvidia runtime image — no need to reinstall.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /src/build/bin/*.so*       /usr/local/lib/
RUN ldconfig

# Models are mounted from a PVC at /models/.
# Expected layout:
#   /models/small/<filename>.gguf   (set via SOPA_SMALL_MODEL env var)
#   /models/large/<filename>.gguf   (set via SOPA_LARGE_MODEL env var)
RUN mkdir -p /models/small /models/large

EXPOSE 8791

ENTRYPOINT ["/usr/local/bin/llama-server"]

# Start without a pre-loaded model — sopa-core loads small/large on demand via
# POST /sopa/load. Pass --model at runtime (k8s args) when a startup model is needed.
CMD ["--host", "0.0.0.0", "--port", "8791", "--no-mmap"]
