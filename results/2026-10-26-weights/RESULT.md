<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Which DeepSeek-V4.1-Flash weights fit four and two Sparks (2026-10-26)

Measured from the safetensors headers of each pinned revision, not from file
sizes or model cards. No weights were downloaded and no Spark was used: this
is a capacity plan, not a serving result.

```sh
python3 tools/memory_plan.py roles --hf REPO@REV --out results/2026-10-26-weights/NAME.roles.json
python3 tools/memory_plan.py plan results/2026-10-26-weights/NAME.roles.json --world 4 --engram nvme
```

`*.roles.json` holds each checkpoint's tensor bytes per placement role and its
`config.json`; `SHA256SUMS` covers them.

## Checkpoints

| Name | Revision | Tensors | Bytes (GB) | Format |
|---|---|---|---|---|
| `deepseek-fp8-mxfp4` | deepseek-ai/DeepSeek-V4.1-Flash-1017 @ `1bae0e10` | 96,085 | 510.29 | DeepSeek's release: FP8 dense (UE8M0, 32x32 blocks), MXFP4 routed experts, FP8 Engram tables |
| `nvidia-nvfp4` | nvidia/DeepSeek-V4.1-Flash-NVFP4 @ `c48dca01` | 188,245 | 428.97 | NVFP4 routed experts (W4A4, group 16, lossless from MXFP4 per NVIDIA), MXFP4 Engram tables (lossy), rest as released. Based on V4.1-Flash, not the 1017 update |
| `exl3-2.5bpw` | bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw @ `b8054f70` | 192,496 | 184.47 | EXL3 mul1, routed experts 2.5 bpw; no Engram tables |
| `exl3-eaq-2.9bpw` | Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw @ `45ee5d0f` | 192,495 | 210.58 | EXL3 mul1, 2.9 bpw body; no Engram tables |
| `exl3-4.0bpw` | bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw @ `f3657818` | 192,496 | 288.10 | EXL3 mul1, 4.0 bpw; no Engram tables |

## Bytes by placement role (GiB)

| Name | Engram tables | Routed experts | TP-split | Replicated | Rank 0 only (vision) |
|---|---|---|---|---|---|
| `deepseek-fp8-mxfp4` | 188.8 | 275.7 | 7.2 | 2.6 | 0.9 |
| `nvidia-nvfp4` | 97.3 | 291.5 | 7.2 | 2.6 | 0.9 |
| `exl3-2.5bpw` | 0 | 165.2 | 4.2 | 1.6 | 0.9 |
| `exl3-eaq-2.9bpw` | 0 | 188.9 | 4.6 | 1.7 | 0.9 |
| `exl3-4.0bpw` | 0 | 260.1 | 5.3 | 2.0 | 0.9 |

Routed experts (MTP/DSpark experts included) are split across ranks. TP-split
is the per-head attention projections, the indexer query heads, and the
vocabulary embedding and head. Replicated is everything else, shared experts
included (the reference keeps them whole on every rank).

## Per-rank residency

Budget per Spark: 119.7 GiB CUDA-visible x 0.88 utilization - 6 GiB reserve
for activations, workspace and graphs = **99.3 GiB**. KV positions are what
is left after the weights and eight sequences' window KV, at 890 bytes per
token (below).

| Name | World | Engram | Rank 0 (GiB) | Fits | KV positions left |
|---|---|---|---|---|---|
| `deepseek-fp8-mxfp4` | 4 | NVMe | 74.2 | yes | 30.2M |
| `deepseek-fp8-mxfp4` | 4 | sharded in memory | 121.5 | no | |
| `deepseek-fp8-mxfp4` | 2 | NVMe | 145.0 | no | |
| `nvidia-nvfp4` | 4 | NVMe | 78.2 | yes | 25.5M |
| `nvidia-nvfp4` | 4 | sharded in memory | 102.5 | no | |
| `nvidia-nvfp4` | 2 | NVMe | 152.9 | no | |
| `exl3-4.0bpw` | 4 | NVMe | 69.3 | yes | 36.2M |
| `exl3-4.0bpw` | 2 | NVMe | 135.6 | no | |
| `exl3-eaq-2.9bpw` | 4 | NVMe | 51.0 | yes | 58.3M |
| `exl3-eaq-2.9bpw` | 2 | NVMe | 99.3 | no at 0.88; 4.1M at 0.90, 8.4M at 0.93 (5 GiB reserve) | |
| `exl3-2.5bpw` | 4 | NVMe | 44.8 | yes | 65.8M |
| `exl3-2.5bpw` | 2 | NVMe | 87.1 | yes | 14.7M |

## KV cache

From the reference layout (inference/model.py) and the published config: only
the four kv_source_layers (2, 8, 14, 20) write compressed KV and index keys,
which the following layers share. A compressed position is 512 FP4 values with
an E4M3 scale per 16 (288 bytes) plus a 128-wide FP4 index key with an E8M0
scale per 32 (68 bytes); layers 2, 8 and 14 compress 2:1 and layer 20 1:1, so
a token costs (288 + 68) x 2.5 = **890 bytes**. Every layer also keeps a
128-slot FP8 sliding window, 2.77 MiB per sequence over 43 layers (40 plus
three DSpark). A 1M-token request needs about 0.9 GB of KV, on every rank,
since the single latent KV head is replicated under tensor parallelism.

## Caveats

- The reserve and utilization are SparkQwen's and SparkGLM's defaults, not
  measured for this model; Atlas may hold extra copies (for example a BF16
  `wo_a`, as the reference converter writes, adds 1.3 GiB per rank at TP2).
- "NVMe" means the Engram tables stay on each Spark's disk behind a row cache,
  as TensorFold serves them; the cache itself (2 GiB there) comes out of the
  reserve.
- Quality of the EXL3 packs is their authors' measurement. bertholomus reports
  KL 0.13 and 92% top-1 against the release on short rows for 2.5 bpw, and
  tool-call turns at KL 0.59 and 79% top-1.
