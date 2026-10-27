<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# SparkSeek

DeepSeek-V4.1-Flash-1017 on NVIDIA DGX Sparks, served by the
[Atlas](https://github.com/Enntity/atlas) inference engine. It is the sibling
of [SparkGLM](https://github.com/Enntity/sparkglm) and
[SparkQwen](https://github.com/Enntity/sparkqwen), and follows their rules:
a pinned engine, weights loaded as published unless an option says otherwise,
and receipts for every claim.

**Status: groundwork. Nothing is served yet.** This repository has the
capacity plan and the engine plan; the Atlas branch has the CPU-tested config
and Engram hash work. GPU kernels, the attention cache, tensor parallelism and
the drafter come next, on Sparks ([docs/ENGINE.md](docs/ENGINE.md),
[docs/BRINGUP.md](docs/BRINGUP.md)).

## What fits

From the safetensors headers of each pinned revision
([results/2026-10-26-weights](results/2026-10-26-weights/RESULT.md)), at
SparkGLM's memory setting (99.3 GiB per Spark for weights and KV):

| Sparks | Weights | Per rank | Status |
|---|---|---|---|
| **4** | [DeepSeek's release](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (FP8 + MXFP4), as published | 74.2 GiB | the Strict target |
| **2** | [EXL3 2.5 bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw) by bertholomus, lossy | 87.1 GiB | the only published pack with KV room at TP2 |
| 2 | DeepSeek's release | 145.0 GiB | does not fit |
| 2 | EXL3 2.9 bpw (Mia-AiLab EAQ) | 99.3 GiB | no KV room at 0.88; 4-8M positions at 0.90-0.93 |
| 4 | NVIDIA NVFP4 | 78.2 GiB | fits, but quantizes the older September model |

In every case the two Engram tables (203 GB) stay on each Spark's NVMe behind
a row cache, as TensorFold serves them; held in memory they push four Sparks
to 121.5 GiB a rank. KV is not the constraint: 890 bytes a token, so a
1M-token request needs about 0.9 GB.

`tools/memory_plan.py` recomputes any row from a checkpoint's headers:

```sh
python3 tools/memory_plan.py roles --hf deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170 --out roles.json
python3 tools/memory_plan.py plan roles.json --world 4 --engram nvme
```

## The engine

Atlas already serves DeepSeek-V4-Flash, one request at a time on two Sparks
with expert parallelism. V4.1 reuses its MoE routing, MXFP4 expert kernels,
mHC and attention projections, but its attention cache is new: a 128-token
window on every layer, compressed KV and indexer top-k produced at a few layers
and shared by the layers after them, a two-level top-k, and Engram n-gram
tables. So it is its own model type, `deepseek_v41`, on the Enntity/atlas
branch `upstream/deepseek-v41-flash-20261026`
([docs/ENGINE.md](docs/ENGINE.md)).

## Credits

DeepSeek-V4.1-Flash and its reference implementation are DeepSeek's (MIT).
Capacity figures for other engines and the TP2 pack's quality are their
authors': MiaAI-Lab's TensorFold recipe for four Sparks, bertholomus's EXL3
packs and two-Spark TensorFold recipe, Mia-AiLab's EAQ packs.

## License

AGPL-3.0-only ([LICENSE](LICENSE)); see [NOTICE](NOTICE). The model weights
are not distributed here.
