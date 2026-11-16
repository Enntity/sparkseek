<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# DeepSeek's reference code, all 40 layers, on one Spark (2026-11-15)

The [three-layer recording](../2026-11-15-reference-layers/RESULT.md) covers
the kinds of cache the first layers build. This is the same prompt through
the whole model: every block's input, output, operands and caches, and what
comes out of the head. It is what the engine's deeper layers are checked
against one at a time, and what a full-depth run on four Sparks will be
compared with. No engine code ran.

The 40 blocks hold 296 GB of weights and one Spark has 121 GiB of
memory, so `record_deep.py` runs them one at a time. A block is built from
the reference's `Block`, loaded from its own file of the release through the
reference's `convert.py`, run over the whole prompt (one prefill of 1,401
tokens, then 48 decode steps), recorded, and freed. What blocks hand each
other is carried between them: every step's stream and Hyper-Connection mix,
and the reference's shared attention state (compressed keys, index keys,
top-k, candidates) as the block before left it at that step.

```sh
# REF = the release's inference/ folder, CKPT = the release, same container as the
# three-layer recording. --against names that recording.
python record_deep.py --ops --ref $REF --ckpt $CKPT --work scratch --out deep \
    --against rec --text $REF/README.md $REF/model.py
```

It took about 11 minutes on a GB10: the release has one file per layer, and a
block converts, loads and runs in 10 to 30 seconds.

## Is it the same computation?

| Check | Result |
|---|---|
| Layers 0 to 2 against the three-layer recording, which ran `Transformer.forward` | all 384 shared tensors bit-identical |
| Next token: the head's first choice against the prompt's own next token, prefill | 74.1% of 1,400 positions; in the top 8 on 92.0% |
| Same, the 47 decode steps that have a next token | 74.5% |

The first row says running blocks one at a time changes nothing where it can
be compared. The other two say the result is the working model: the prompt is
DeepSeek's README and source, which a model at chance would not predict.

## What differs from running the reference as shipped

The three differences of the three-layer recording (sparse Engram table,
sparse attention 16 heads at a time, TF32 off), and:

- **`Transformer.forward` is replaced by a loop over blocks** that follows
  its lines: Engram before a block that has one, then the block, then after
  the last block the collapse, norm and head.
- **Decode steps run block by block, not step by step.** Block 0 runs all 48
  steps, then block 1. A block at step `s` is given the shared attention
  state the block before it left at step `s`. A source's caches are read
  through a length, so entries written by later steps are not seen.
- **No draft layers, no vision.** Layers 40 to 42 are not run.

## What is in it

Per layer `L<nn>_prefill`, `L<nn>_decode` and `L<nn>_caches`, with the names
of the three-layer recording, including every projection, norm, routing and
Hyper-Connection step. `final` has the collapsed and normed stream at every
position, the eight highest logits and their token ids at every position, and
all 129,280 logits at the last prefill position and at each decode step.

The layers it adds a first recording of:

| Layers | What |
|---|---|
| 3 to 19 | read the compressed keys and the top-512 of the source below them (2, 8, 14) |
| 8, 14 | further key and index sources at ratio 2 |
| 14 | the second Engram layer |
| 20 | ratio 1: every token is a compression group. Candidate source |
| 21 to 39 | read layer 20's keys |
| 24, 28, 32, 36 | index sources that score layer 20's index keys with their own queries |
| after 39 | the final collapse, norm and head |

## Not covered

- **Candidate blocks never cut.** 2,048 blocks of 8 hold 16,384 compressed
  positions, and this prompt has 1,449, so every position is a candidate. A
  prompt past 16,384 tokens is needed, and the reference's single prefill of
  one needs more memory than one Spark's free share.
- **Decode steps on an odd number of index keys** carry the reference's
  library rounding, as described in the three-layer result. From layer 20 the
  key count is odd on every other step.
- The run was not repeated in a second process. Layers 0 to 2 matching the
  earlier recording is the evidence that it is deterministic.
- More than one rank.

## Files

`record_deep.py` is the script. `meta.json` has the token ids and the shape
of every saved tensor. The tensors are 34 GB and are not in the repository;
`tensors.sha256` lists their hashes from this run.
