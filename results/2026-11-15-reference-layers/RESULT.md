<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# DeepSeek's reference code, first three layers, on one Spark (2026-11-15)

The engine's attention cache ([docs/ENGINE.md](../../docs/ENGINE.md), step 7)
is accepted one layer at a time against DeepSeek's own forward pass. This is
that forward pass: `inference/model.py` from the pinned release
(`1bae0e10`), run on the release's weights for layers 0 to 2, with every
attention layer's input, output, operands and caches saved. No engine code ran.

Layers 0 and 1 attend over the 128-token window only. Layer 1 follows an
Engram lookup. Layer 2 is the first that compresses its keys and runs an
indexer, so the three cover every kind of cache the step builds.

```sh
# REF = the release's inference/ folder, CKPT = the release, in a container with
# nvcr.io/nvidia/pytorch:25.10-py3, tilelang==0.1.12, safetensors, tokenizers, sympy, tqdm
python $REF/convert.py --hf-ckpt-path <model-00002..00005 only> --save-path slice \
    --model-parallel 1 --expert-dtype fp4 --tokenizer-path $CKPT
python record.py --ops --ref $REF --slice slice/model0-mp1.safetensors --ckpt $CKPT --out rec \
    --prefill 1401 --decode 48 --text $REF/README.md $REF/model.py
python check.py --rec rec --slice slice/model0-mp1.safetensors --config $REF/config.json
```

The prompt is 1,449 tokens: BOS, then DeepSeek's own README and the start of
`model.py`. Run A prefills 1,401 tokens and decodes 48 one at a time. Run B
prefills all 1,449 at once. 1,401 is odd, so prefill ends inside a compression
group, and it is past 1,024, so layer 2's indexer has more than 512 compressed
positions to choose from.

## What differs from running the reference as shipped

- **Three layers, no head.** The model is built with `n_layers = 3` and no
  draft layers or vision. Nothing recorded depends on what is missing.
- **Engram table.** The released table for layer 1 is 98 GB. The 27,589 rows
  this prompt reads are gathered from the checkpoint file and looked up through
  an index; the dequantization after the lookup is the reference's.
- **Sparse attention runs 16 heads at a time.** At 64 heads of 512 the
  reference kernel asks for 141,312 bytes of shared memory per block, which a
  GB10 refuses. Heads are independent in the kernel, so the wrapper calls it
  four times. Everything else in `kernel.py` runs unchanged on a GB10, and
  `python model.py`, the reference's self-test, passes there.
- **PyTorch 2.9**, from the NVIDIA container; `requirements.txt` asks for 2.10.
- **TF32 off.** NVIDIA's container runs FP32 matrix products in TF32 unless
  told not to; stock PyTorch, which the reference is written for, does not.
  `record.py` turns it off. With it on, the reference's FP32 products
  (Hyper-Connection mixes, routing scores, the compressor's projections) are
  about 3e-4 from their exact values.

The first version of this result, published earlier the same day, was recorded
with TF32 on. Its tensors are replaced by the ones described here. The
findings did not change; the figures in the first table did.

## The reference against itself

| Check | Result |
|---|---|
| Run A repeated in the same process | every saved tensor bit-identical |
| Layer 0, positions 1,401 to 1,448: decoded (A) against prefilled (B), same inputs | attention output differs by 2.2e-4 relative; window ring bit-identical |
| Same positions, layer 0 block output | 5.9e-4 |
| Layer 1 attention input / output | 1.3e-3 / 1.7e-2 |
| Layer 2 attention input / output | 1.9e-2 / 5.3e-2 |

Differences are relative L2 ([stages.json](stages.json)). The first comes from
the order keys are read in: a decode step reads the ring, prefill reads
positions. It then grows at every step that quantizes activations to FP8 or
routes to experts. Two runs of DeepSeek's own code that split prefill and
decode differently are 5% apart by the third layer's attention.

## The recording against a transcription

`check.py` imports nothing from the reference. It recomputes three things from
the saved tensors ([check.json](check.json)):

| What | Layers | Result |
|---|---|---|
| Window ring: slot `pos % 128` holds position `pos`'s key; a decode step rewrites one slot; decode reads slots oldest first | 0, 1, 2 | exact, after prefill and after every decode step |
| Compressed KV: the compressor's latent, rotated at its group's first position (theta 160,000, YaRN), quantized to FP4 in groups of 16 under an E4M3 scale | 2 | all 700 prefill groups and 24 decode groups bit-identical |
| A group completes at decode when `pos + 1` divides by the ratio; the one prefill left half full completes on the first decode step | 2 | holds |
| Attention output, following the kernel's schedule: 64 keys at a time in index order, running maximum, weights rounded to BF16 before they multiply the keys | 0, 1, 2 | 99.88% to 99.99% of output bits equal; 2e-5 to 9e-5 relative |
| Attention output, plain softmax in float64 | 0, 1, 2 | 81% to 86% of output bits equal; 1.0e-3 to 1.3e-3 relative |

## What it means for the engine

- **Judge one layer at a time, on the reference's inputs.** Whole-model logits
  against the reference cannot be a tight check: the reference does not agree
  with itself past a few percent once prefill is split differently.
- **Attention tolerance.** An attention kernel that computes the softmax
  exactly lands about 1.3e-3 from the reference, the same order as BF16
  rounding of the output. Getting to 1e-4 means copying the reference's BF16
  weight rounding and its 64-key schedule.
- **The cache contents can be exact.** The window ring and the compressed KV
  are fixed by a short recipe and reproduce bit for bit, so the engine's caches
  can be compared for equality, not closeness.
- **Chunked prefill has no reference.** The reference prefills once from
  position 0 and then decodes token by token. A chunked prefill is checked
  against a single prefill of the same tokens.

## Not covered

- The indexer's scores and its top-512 choice are saved (`L2.indexer_topk`)
  but not recomputed; that is step 8.
- The Engram layer's input and output are saved but not recomputed; step 9.
- Layers past 2: candidate blocks start at layer 20, compression ratio 1 at
  layer 20, the draft layers after 39.
- More than one rank. This is the reference at `world_size = 1`.

## Files

`record.py` and `check.py` are the two scripts. `meta.json` has the token ids
and the shape of every saved tensor. With `--ops`, run A also saves the input
and output of every projection, norm and Hyper-Connection step inside a block,
and each MoE's input, routing and output (`L<layer>.op.<module>.in`, `.out`),
so one operation can be checked from its own recorded input. The tensors are
3.3 GB and are not in the repository; `tensors.sha256` lists their hashes from
this run, and `record.py` rebuilds them from the public release. A second run
in a new process gave the same hashes.
