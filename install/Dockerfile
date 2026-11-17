# SPDX-License-Identifier: AGPL-3.0-only
# Context: build.sh's pinned Enntity/atlas engine checkout and this install directory.
# Adapted from SparkGLM's install/Dockerfile (AGPL-3.0-only): the same base
# images, toolchain, NCCL and CUTLASS pins, with DeepSeek-V4.1-Flash's kernel
# target in place of GLM's and without GLM's converter, native libraries and tests.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS builder
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    git ca-certificates build-essential python3 && rm -rf /var/lib/apt/lists/*
RUN apt-get update -qq && apt-get install -y -qq --allow-change-held-packages \
    curl pkg-config cmake libclang-dev libibverbs-dev librdmacm-dev \
    libnccl2=2.31.2-1+cuda13.3 libnccl-dev=2.31.2-1+cuda13.3 && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile minimal --default-toolchain 1.93.1
ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUSTUP_TOOLCHAIN=1.93.1 CUDA_HOME=/usr/local/cuda CUTLASS_HOME=/opt/cutlass
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV NVCC_THREADS=2
RUN git clone https://github.com/NVIDIA/cutlass.git /opt/cutlass && \
    git -C /opt/cutlass checkout cf064d2e6bad2886238ac565b3b49007764f4939
WORKDIR /build
COPY engine/Cargo.toml engine/Cargo.lock engine/rust-toolchain.toml engine/README.md ./
COPY engine/crates/ crates/
COPY engine/kernels/ kernels/
COPY engine/vendor/ vendor/
COPY engine/jinja-templates/ jinja-templates/
# atlas-plugin's quick benchmark embeds its prompts from tests/fixtures.
COPY engine/tests/fixtures/ tests/fixtures/
# One kernel target: deepseek-v4.1-flash, which compiles deepseek-v4-flash's
# kernels and its own overlay (its MODEL.toml's kernel_source).
ENV ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=deepseek-v4.1-flash ATLAS_TARGET_QUANT=nvfp4
ARG BUILD_JOBS=4
# A BuildKit cache mount holds the cargo target directory, so a rebuild after
# an engine change does not start cold. Every cargo step below mounts the same
# cache, and the binary is copied out of the mount because mounted paths never
# reach a layer. The legacy builder ignores the mount; BuildKit is required.
RUN --mount=type=cache,target=/build/target,id=sparkseek-cargo-target,sharing=shared \
    cargo build --locked --release -j "$BUILD_JOBS" -p spark-server && \
    install -D target/release/spark /build/dist/spark
# License notices of every third-party crate compiled into the binary.
COPY install/rust-notices.py /product/install/rust-notices.py
RUN python3 /product/install/rust-notices.py /build /build/dist/notices-rust

# The engine's unit tests for this model, on the same Linux/CUDA build as the
# shipped binary. A stage of their own: the runtime image below copies only
# from `builder`, so BuildKit runs these only when build.sh asks for this
# target (ATLAS_IMAGE_CHECKS=1, the default). Each filter must run tests.
FROM builder AS checks
# The tests read fixtures beside their sources (test_data/).
COPY engine/ /build/
# CUDA stubs permit these CPU-only tests without a GPU.
RUN mkdir -p /opt/cuda-stubs && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /opt/cuda-stubs/libcuda.so.1
# The published 1017 config and what it shares between layers; Engram's hash ids.
RUN --mount=type=cache,target=/build/target,id=sparkseek-cargo-target,sharing=shared \
    set -eu; for filter in deepseek_v41 engram_dsv41; do \
      LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
      cargo test --locked --release -j "$BUILD_JOBS" -p atlas-core --lib "$filter" \
        > /tmp/atlas-config-check.log 2>&1 || { cat /tmp/atlas-config-check.log; exit 1; }; \
      cat /tmp/atlas-config-check.log; \
      grep -E 'test result: ok. [1-9][0-9]* passed;' /tmp/atlas-config-check.log; \
    done
# The V4.1 prompt encoding and tool-call parser, and Engram's token map.
RUN --mount=type=cache,target=/build/target,id=sparkseek-cargo-target,sharing=shared \
    set -eu; for filter in deepseek_v41 engram_dsv41; do \
      LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
      cargo test --locked --release -j "$BUILD_JOBS" -p spark-server --bin spark "$filter" \
        > /tmp/atlas-prompt-check.log 2>&1 || { cat /tmp/atlas-prompt-check.log; exit 1; }; \
      cat /tmp/atlas-prompt-check.log; \
      grep -E 'test result: ok. [1-9][0-9]* passed;' /tmp/atlas-prompt-check.log; \
    done

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3 AS runtime
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
    python3 ca-certificates libnccl2=2.31.2-1+cuda13.3 libibverbs1 librdmacm1 \
    ibverbs-providers libnl-3-200 libnl-route-3-200 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/dist/spark /usr/local/bin/spark
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY --from=builder /opt/cutlass/LICENSE.txt /opt/atlas/notices/CUTLASS-LICENSE.txt
COPY --from=builder /build/dist/notices-rust/ /opt/atlas/notices/rust/
COPY install/serve.py install/release.json /opt/atlas/
COPY install/atlas-source.json /opt/atlas/source-manifest.json
COPY engine/LICENSE /LICENSE
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
ENV RUST_LOG=info
EXPOSE 8894
# The image's identity is the git tree of install/ (its tag); label it last.
ARG INSTALL_TREE
ARG PRODUCT_VERSION
RUN python3 -c 'import json,sys; assert json.load(open("/opt/atlas/release.json"))["version"] == sys.argv[1]' "$PRODUCT_VERSION"
RUN test -n "$INSTALL_TREE"
LABEL org.opencontainers.image.version="$PRODUCT_VERSION" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.source="https://github.com/Enntity/sparkseek" \
      io.enntity.sparkseek.install-tree="$INSTALL_TREE"
ENTRYPOINT ["python3", "/opt/atlas/serve.py"]
