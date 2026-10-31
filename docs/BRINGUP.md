<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Bring-up on Sparks

What to do when Sparks are available, in order. Reserve shared hardware
through LLooM first (`lloom reserve`), and never take a pair that serves
production without its owner's go-ahead.

## Storage, once per Spark

- DeepSeek's release is 510 GB. TP4 serving reads about 80 GB (74 GiB) of it per rank
  plus the two Engram shards (`model-00047`, `model-00048`, 101 GB each) on
  every Spark's own NVMe; TensorFold reads the rest over NFS from one Spark.
  Plan for the full checkpoint on one Spark and the Engram shards on each.
- The TP2 pack is 185 GB on each Spark, plus the same two Engram shards.
- NVIDIA's NVFP4 pack, if testing it, is 429 GB. Its own Engram shards (the same
  two file names, 52 GB each) go on every Spark's NVMe.
- O_DIRECT reads need ext4 or xfs. Check the NVMe's random 4 KiB read rate
  (`fio --rw=randread --bs=4k --direct=1 --iodepth=64`); the Engram row cache
  misses go there.

## Before any GPU code: reference activations

Run DeepSeek's reference `inference/` (torch, FP8/FP4 kernels) once on four
Sparks, or on any machine that holds it, for a short prompt set, and record
per-layer inputs and outputs of: layer 1 (Engram + window-only attention),
layer 2 (first compressor and indexer source), layer 3 (a consumer), layer 20
(1:1 source, candidate blocks), layer 24 (two-level top-k), and the DSpark
stages. These are the goldens for ENGINE.md steps 7-11. Store them with their
prompt, revision and SHA-256 under the research notes, not here.

The Engram hash ids in Atlas were checked against a transcription of the
reference, not the reference itself (running downloaded code was not allowed
in the session that wrote them). Cross-check `NgramHashState` on a Spark once:
same prompt, compare ids.

## Gates, per ENGINE.md step

| Step | Gate |
|---|---|
| 6 kernel target | `--check-kernels` resolves every handle on each rank; `expected_absent` entries name their fallbacks |
| 7 cache | one-layer outputs within BF16 tolerance of the reference across a chunked prefill and 64 decode steps, at batch 1 and 4 |
| 8 indexer | top-k sets equal to the reference at 4K, 32K and 128K; candidate blocks equal at layer 20 |
| 9 Engram | layer-1 output equal to the reference; NVMe row cache hit rate and decode stall reported |
| 10 TP | TP2/TP4 logits against TP1 on a 4-layer slice; then the full model on four Sparks: 9/9 RigMark gates, needle at 128K and 1M |
| 11 DSpark | greedy output identical with drafting on and off, at 1, 4 and 8 streams; acceptance per step |
| 12 EXL3 TP2 | KL and top-1 against the release on the same rows bertholomus published |
| NVFP4 lane | after step 11: KL and top-1 against the release, decode speed and Engram hit rate, with the drafter on; W4A16 and W4A4 experts both measured |

## Baselines on the same Sparks

- TP4: TensorFold with MiaAI-Lab's recipe
  (github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold),
  which reports 65.8 tok/s prose and 117.4 tok/s code at one stream, and about
  4,100 tok/s prefill.
- TP2: TensorFold with bertholomus's recipe
  (github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10, v0.7.3) on the
  2.5 bpw pack: 106.0 / 60.5 / 118.3 tok/s code / prose / structured.

Those are their numbers on their hardware. Measure both on ours before any
comparison, with every repetition and receipts under `results/`.
