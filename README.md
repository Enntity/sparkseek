<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# SparkSeek

DeepSeek-V4.1-Flash-1017 on NVIDIA DGX Sparks, served by the
[Atlas](https://github.com/Enntity/atlas) inference engine. It is the sibling
of [SparkGLM](https://github.com/Enntity/sparkglm) and
[SparkQwen](https://github.com/Enntity/sparkqwen), and follows their rules:
a pinned engine, weights loaded as published unless an option says otherwise,
and receipts for every claim.

**Status: serves on four Sparks.** `./start.sh` prepares the pinned engine and
serves DeepSeek's release as published on four Sparks joined in an RDMA ring:
prefill, and decode with the release's draft head. Three Sparks are
[not served yet](#three-sparks).

First speeds, one stream, through `./start.sh` at its defaults
([receipt](results/2026-11-17-four-sparks-base/RESULT.md)): a cold
32,000-token prefill in 4.87 to 5.45 s (5,875 to 6,567 tok/s), and replies at
53 tok/s on prose, 96 on Python and 86.5 on JSON with the release's draft head
on (46 tok/s with it off). With `--prefill-rows 16384 --drafts 0` the same
prefill takes 4.72 to 5.36 s.

That receipt was recorded one engine pin back (`8c71b62e`). The engine pinned
now runs a batch's attention faster and changes nothing else: on the same
four Sparks the 16,384-row prefill took 4.35 s (7,350 tok/s) on the engine's
own test harness, with the same tokens after a 32,000-token and a 9,000-token
prompt. It has not been started through `./start.sh` yet, so there is no
receipt for it here.

```sh
git clone https://github.com/Enntity/sparkseek.git
cd sparkseek
cp .env.example .env      # set WORKER to the other three Sparks' ssh destinations
./start.sh
```

## Requirements

- Four DGX Spark (GB10, 128 GB) systems cabled as a closed ring, each to two
  neighbours over ConnectX-7, with no switch. Every cabled port needs an IPv4
  address, and each pair of connected ports a subnet of its own: `start.sh`
  reads the ring from those subnets. A chain of four is refused, because this
  release has run on a ring only.
- The ranks come together over sockets on the rank 0-1 and rank 2-3 links
  before the ring carries the model's traffic, so every Spark must reach
  rank 0's address there, the far ones through routes on the hosts.
  `start.sh` configures no cables, addresses or routes; `./start.sh fabric`
  shows the ring it found and changes nothing.
- Docker with the NVIDIA runtime on every Spark, and passwordless `ssh` from
  the Spark that runs `start.sh` to the other three, whose users can run
  `docker`.
- About 510 GB free on each Spark's own NVMe for the checkpoint, or an
  existing copy at the same path on every Spark (`CHECKPOINT_DIR` in `.env`).
- Nothing else using the GPUs' memory. GB10 memory is shared with the host.
- The GPU clock capped on every Spark: run `./start.sh clock-cap` once (it
  asks for `sudo` on each). Uncapped, a long cold prefill can power a Spark
  off with nothing in any log; SparkGLM measured this on the same hardware
  ([its results](https://github.com/Enntity/sparkglm/tree/main/results/2026-10-13-clock-cap)),
  and a Spark it capped is capped here too.

## What `./start.sh` does

Run it on the Spark that will serve the API (rank 0). `WORKER` in `.env` names
the other three in any order; the cables decide the ranks. Each step is skipped
when already done, so running it again restarts the engine.

1. **Checks.** The options, the ring, and `gpu-clock-cap.service` on every
   Spark (`CLOCK_CAP=0` skips that one), before anything is pulled, downloaded
   or started.
2. **Image.** Pulls `ghcr.io/enntity/atlas-sparkseek:<tag>`, whose tag is the
   git tree of [`install/`](install/). If that image is not published, or
   `BUILD=1` is set, it builds it from source
   ([`install/build.sh`](install/build.sh): Atlas at the commit pinned in
   [`install/atlas-source.json`](install/atlas-source.json), kernel target
   `deepseek-v4.1-flash`; any other engine tree is refused). It then pulls or
   copies the image to the other Sparks.
3. **Weights.** Downloads
   [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
   at the revision pinned in [`install/checkpoints.json`](install/checkpoints.json)
   into `MODEL_ROOT` (default `~/models/sparkseek`) and copies it to the other
   Sparks. It is served as published: there is no conversion step. With
   `CHECKPOINT_DIR` set, nothing is downloaded: that directory must already
   hold the release on every Spark.
4. **Serve.** Starts ranks 3, 2 and 1 over ssh, then rank 0 here
   ([`install/start-node.sh`](install/start-node.sh)), waits for the API (up
   to 30 minutes), and asks it one smoke-test question.

Other commands: `./start.sh fabric | clock-cap | stop | status | logs [worker [HOST]] | build | download | --version`.

## Options

Each is a `start.sh` option and a line in `.env`; the command line wins.

| Option | `.env` | Default | |
|---|---|---|---|
| `--sparks N` | `SPARKS` | 4 | Sparks serving together. Only 4 is served so far. |
| `--drafts N` | `DRAFTS` | 5 | Tokens the release's draft head proposes each step, 0 to 15. 0 turns the draft head off. |
| `--prefill-rows N` | `PREFILL_ROWS` | 8192 | Rows a prefill pass runs together: 8192 or 16384. 16384 is quicker on long prompts and takes about 3 GB more a rank; with the draft head on it needs a Spark's memory to itself. |
| `--context N` | `CONTEXT` | 131072 | The longest sequence, prompt and reply, in tokens; at most the model's 1,048,576. |
| `--port N` | `PORT` | 8894 | The API's port on rank 0. Rank R also listens on this port + R, on its own loopback. |

[`.env.example`](.env.example) lists the rest: where the weights live, the
source build and the clock cap.

## Use it

The API listens on rank 0's loopback, `http://127.0.0.1:8894/v1`, model
`sparkseek`, with no authentication. Put your own authenticated proxy in front
of it before exposing it.

```sh
curl -s http://127.0.0.1:8894/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "sparkseek", "max_tokens": 512,
  "messages": [{"role": "user", "content": "Write a haiku about four small computers."}]}'
```

Limits of this base version:

- Only one sequence at a time is supported so far (`--max-num-seqs 1`).
- Four Sparks in a closed ring; no other number and no chain.
- Text only: the engine does not load the release's vision tower yet.
- No speed or quality figure is claimed for the served engine until its
  receipt is under [`results/`](results/).

## Three Sparks

Not yet. The pinned engine splits the release's weights four ways, one part a
rank; three Sparks need a three-way split, which it does not have.
`--sparks 3`, `SPARKS=3` and a `WORKER` list of two stop with that message
before anything starts. On two Sparks the release does not fit (the table
below).

## What fits

From the safetensors headers of each pinned revision
([results/2026-10-26-weights](results/2026-10-26-weights/RESULT.md),
[results/2026-10-30-nvfp4-1017](results/2026-10-30-nvfp4-1017/RESULT.md)), at
SparkGLM's memory setting (99.3 GiB per Spark for weights and KV):

| Sparks | Weights | Per rank | Status |
|---|---|---|---|
| **4** | [DeepSeek's release](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (FP8 + MXFP4), as published | 74.2 GiB | served by `./start.sh` |
| **2** | [EXL3 2.5 bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw) by bertholomus, lossy | 87.1 GiB | the only published pack with KV room at TP2; not served yet |
| 2 | DeepSeek's release | 145.0 GiB | does not fit |
| 2 | EXL3 2.9 bpw (Mia-AiLab EAQ) | 99.3 GiB | no KV room at 0.88; 4-8M positions at 0.90-0.93 |
| 4 | [NVIDIA NVFP4 of 1017](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4), lossy Engram tables | 78.2 GiB | candidate, to test against the release on Sparks |

In every case the two Engram tables stay on each Spark's NVMe behind a row
cache, as TensorFold serves them. They are 203 GB in DeepSeek's release and
104 GB in NVIDIA's pack; held in memory they push four Sparks to 121.5 and
102.5 GiB a rank. KV is not the constraint: 890 bytes a token, so a 1M-token
request needs about 0.9 GB.

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
branch `upstream/deepseek-v41-flash-20261114`, pinned in
[`install/atlas-source.json`](install/atlas-source.json)
([docs/ENGINE.md](docs/ENGINE.md) has the engine's history).

## Credits

`start.sh`, `fabric.py`, `gpu-clock-cap.sh` and the tooling in `install/` are
adapted from [SparkGLM](https://github.com/Enntity/sparkglm) (AGPL-3.0-only);
the container flags in `install/start-node.sh` follow
[LLooM](https://github.com/Enntity/lloom)'s Atlas recipe (MIT,
[`LICENSES/MIT-LLooM.txt`](LICENSES/MIT-LLooM.txt)).

DeepSeek-V4.1-Flash and its reference implementation are DeepSeek's (MIT).
Capacity figures for other engines and the TP2 pack's quality are their
authors': MiaAI-Lab's TensorFold recipe for four Sparks, bertholomus's EXL3
packs and two-Spark TensorFold recipe, Mia-AiLab's EAQ packs, NVIDIA's NVFP4
pack and its accuracy results.

## License

AGPL-3.0-only ([LICENSE](LICENSE)); see [NOTICE](NOTICE). The model weights
are not distributed here.
