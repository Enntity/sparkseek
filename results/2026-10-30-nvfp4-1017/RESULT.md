<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# NVIDIA's NVFP4 pack of DeepSeek-V4.1-Flash-1017 (2026-10-30)

Measured from the safetensors headers of the pinned revision, the same way as
[2026-10-26-weights](../2026-10-26-weights/RESULT.md). No weights were
downloaded and no Spark was used: this is a capacity plan, not a serving
result.

```sh
python3 tools/memory_plan.py roles --hf nvidia/DeepSeek-V4.1-Flash-1017-NVFP4@480550d349ede47a048eb65a276c5c01a06256e2 --out nvidia-nvfp4-1017.roles.json
python3 tools/memory_plan.py plan nvidia-nvfp4-1017.roles.json --world 4 --engram nvme
```

## The checkpoint

`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4` @ `480550d3`: 188,245 tensors, 428.99 GB
in 48 shards. NVIDIA's card gives its source as DeepSeek's release at
`1bae0e10`, the revision SparkSeek pins, and every `config.json` field outside
`quantization_config` equals the release's.

Against the release's tensor names:

- Routed experts (384 x 40 x 3 projections) are NVFP4, group 16. Each `scale`
  becomes `weight_scale`, `weight_scale_2` and `input_scale`. NVIDIA reports the
  weight conversion lossless, and activation scales calibrated for W4A4, with
  24 of 46,080 entries on fallback scales.
- Engram tables are MXFP4 in blocks of 32 with UE8M0 scales, converted lossily
  from FP8: 136 bytes a row, 52 GB in each of `model-00047` and `model-00048`.
- DSpark experts keep MXFP4 and their `scale` tensors, so one model holds both
  expert formats. Attention, shared experts and vision are as released.

## Bytes by placement role (GiB)

| Engram tables | Routed experts | TP-split | Replicated | Rank 0 only (vision) |
|---|---|---|---|---|
| 97.3 | 291.5 | 7.2 | 2.6 | 0.9 |

## Per-rank residency

Budget per Spark as before: 99.3 GiB at utilization 0.88.

| World | Engram | Rank 0 (GiB) | Fits | KV positions left |
|---|---|---|---|---|
| 4 | NVMe | 78.2 | yes | 25.5M |
| 4 | sharded in memory | 102.5 | no at 0.88; yes at 0.93 | 3.4M at 0.93 |
| 2 | NVMe | 152.9 | no | |

Every figure equals the September pack's in the 2026-10-26 receipt, as the two
share their shapes and formats. Against the release at TP4 with Engram on NVMe,
this pack costs 4.0 GiB more per rank and 4.7M fewer KV positions, and holds
about twice the Engram rows in the same cache.

## Caveats

- Accuracy is NVIDIA's measurement on vLLM on GB300, not ours. Against the
  source, in percent: GPQA Diamond 92.93 / 92.77, AA-LCR 80.19 / 80.41, SciCode
  57.32 / 57.58, IFBench 76.47 / 76.33, MMMU-Pro 78.73 / 78.38, Terminal-Bench
  2.1 81.11 / 81.81.
- NVIDIA did not exercise speculative decoding with this pack.
- The sharded row needs a cross-rank Engram gather that no plan here includes,
  and the 0.93 budget is not a tested setting.
