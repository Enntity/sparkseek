<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# The engine: DeepSeek-V4.1-Flash on Atlas

SparkSeek serves DeepSeek-V4.1-Flash-1017 with
[Atlas](https://github.com/Enntity/atlas), like SparkGLM and SparkQwen. This
page records what Atlas already has, what V4.1 needs that it does not, and the
order the work lands in. Status as of 2026-11-15: steps 1 to 5 are done, and
step 6's kernel target compiles and passes its GPU test on a Spark. No model
loads or serves yet.

## Branch

`upstream/deepseek-v41-flash-20261114`, cut from Atlas-Inf `main` at
`d080d6b7d`, so the series stays proposable upstream. It replaces
`upstream/deepseek-v41-flash-20261026`, which was cut at `504063ad`. The product
build will stack it on the shared engine layer SparkGLM and SparkQwen use
(`upstream/atlas-shared-*`) once there is something to serve. Pinned in `install/atlas-source.json`.

## What V4.1 changes from V4

From DeepSeek's reference `inference/model.py` and the published config:

| | DeepSeek-V4-Flash (in Atlas) | DeepSeek-V4.1-Flash-1017 |
|---|---|---|
| Shape | 43 layers, hidden 4096, 256 experts, q_lora 1024 | 40 layers, hidden 5120, 384 experts (inter 2304), q_lora 1280 |
| Routing | first 3 layers hash-routed | no hash layers; separate VL bias for image tokens |
| Attention | CSA 4:1 / HCA 128:1, a compressor and indexer per layer | 128-token window everywhere; compressed 2:1 (layers 2-19) and 1:1 (20-39) |
| KV sharing | none | compressed KV and index keys only at layers 2, 8, 14, 20, reused by the layers after each |
| Indexer | per CSA layer | at layers 2, 8, 14, 20, 24, 28, 32, 36 (32 heads x 128), reused in between; a two-level top-k from layer 20 (2048 candidate blocks of 8) |
| KV dtype | FP8 | compressed KV FP4 + E4M3/16; index keys FP4 + E8M0/32; window FP8 + E8M0/32 |
| FP8 weight blocks | 128 x 128 | 32 x 32 (UE8M0) |
| Engram | none | n-gram hash tables at layers 1 and 14 (2 x 384M rows x 256 FP8), a gated add into the mHC stream |
| Drafter | one MTP module, or DSpark stages | three DSpark stages: block of 5, Markov head, confidence head, 128 experts top-3 |
| Vision | none | 32-layer ViT and aligner; image spans skip Engram |
| Prompt format | DSML `tool_calls` blocks | DSML ` calls` / ` invoke` / ` parameter` (leading space), numeric reasoning effort 1-100, mid-conversation `<｜System｜>`, `namespace::name` tools, image placeholders |

KV per token is 890 bytes (results/2026-10-26-weights), so memory goes to
weights, not context.

## What Atlas already has

A read of Atlas-Inf `main` at `504063ad`:

- **Reusable as is or nearly:** sqrtsoftplus/noaux routing with
  `routed_scaling_factor` (`common/moe_topk_sqrtsoftplus.cu`, up to 512
  experts); routed-expert GEMMs on native MXFP4/E8M0 and NVFP4
  (`moe_w4a16_grouped_gemm.cu`, `*_e8m0`); mHC with Sinkhorn
  (`deepseek-v4-flash/nvfp4/hyper_connection.cu`, `ops/hyper_connection*.rs`);
  V4's attention projections, grouped `wo_a`, inverse RoPE on the output and
  attention sink (`decode/attention_forward_v4.rs`); the weight loader's naming
  and expert formats (`weight_loader/deepseek_v4/`); the DSML tokenizer and tool
  parser (`tokenizer/deepseek_v4*`, `tool_parser/deepseek_v4_dsml.rs`).
- **Close templates:** qwen4_exp's PLE layer (`layers/ple/`, `ple.cu`) is
  Engram's shape: host-side hash ids, a cached gather, a key/value projection,
  a gate against each mHC stream, an add into the FP32 highway. The NVMe row
  cache `spark-storage/src/ngram_cache.rs` (pinned GPU-addressable arena,
  CLOCK eviction, O_DIRECT reads straight from safetensors offsets) fits the
  384M-row tables. GLM-5.3's pooled indexer and qwen4_exp's QSA top-k
  (`qsa_radix.cuh`) are the closest working top-k machinery. Lightning's DSpark
  head (`layers/dflash_head/`, with `markov.rs`) is the closest drafter.
- **Not usable for V4.1:** V4's compressed state is one pool per layer shared
  by all requests (batch size 1); the verify and batched paths skip the
  compressor; chunked prefill attends only within the chunk; there is no
  indexer; DSpark is config and preflight only; TP is refused
  (`DeepSeekV4WeightLoader::supports_tp`); the FP8 GEMV hardcodes 128-wide
  blocks (`common/w8a16_gemv.cu`); EXL3 has dense kernels only and is
  allowed for qwen4_exp alone.

So V4.1 is a new `deepseek_v41` model type that reuses V4's MoE, mHC, attention
projections and loader pieces, and brings its own cache, indexer, Engram,
drafter and TP layout.

## Done (CPU-tested on macOS, `cargo test -p atlas-core`)

1. **Config.** `parsers/deepseek_v41.rs`: unwraps `text_config`, shares V4's
   parse body, and refuses a config whose sharing would read a wrong cache:
   a compressing layer without an earlier source of the same ratio, DSpark
   stages with compressed attention, a candidate source that runs no indexer.
   `ModelConfig::kv_source_layer` / `index_source_layer` answer which cache and
   top-k a layer reads. Dispatch accepts `model_type` or the architecture name.
   Pinned to the published 1017 `config.json` (`test_data/`).
2. **Engram hash ids.** `config/engram_dsv41.rs`: bucket primes, offsets,
   numpy's `default_rng` multipliers (SeedSequence, PCG64, Lemire), and row
   ids with sequence-start and image-span breaks. The prime buckets sum to the
   config's `engram_num_embeddings` exactly, which `validate` requires; the
   draws match numpy; row ids match a transcription of the reference
   `NgramHashState` over the 1017 tokenizer.

3. **Compressed vocab.** `tokenizer/engram_dsv41.rs` builds Engram's token
   map with the `tokenizers` crate's normalizers. On the 1017 tokenizer:
   129,280 ids to 99,092, SHA-256 `c60a8632…59fe46481`, as the reference. A
   tokenizer that compresses to another size is refused at serve start.
4. **V4.1 prompt encoding and DSML parser.** `Dialect::V41` inside the V4
   encoder (`tokenizer/deepseek_v4/`). Byte-exact on DeepSeek's five fixture
   pairs and eight more cases from its `encoding.py`. The repository's later
   `chat_template.jinja` (revision `08c8657f`) renders the same bytes on the
   nine cases it accepts; it refuses string tool-call arguments. OpenAI effort
   names map as that template maps them.
5. **Weight manifest.** `preflight/deepseek_v41.rs` equals the headers of the
   release (96,085 tensors), NVIDIA's NVFP4 pack (188,245) and the EXL3
   2.5 bpw pack (192,496). Nothing calls it at load yet. The downloaded
   release's headers equal the pinned ones.

## Next, in order

Each step names the check that accepts it. "Spark" marks a check that needs
one.

6. **Kernel target.** `kernels/gb10/deepseek-v4.1-flash/` compiles
   deepseek-v4-flash's kernels (`kernel_source`) plus an overlay. Done so far,
   on a Spark (2026-11-15):
   - The target compiles: 216 kernels.
   - FP8 needs no new kernel. The release scales FP8 per 32 x 32 block with
     UE8M0; `weight_map::fp8_block_scales_to_rows` repeats each scale row, and
     decode then runs on GLM's exact tensor-core MXFP8 tiers (`mxfp8_gemv`),
     which the overlay includes. V4's own FP8 kernels assume 128 x 128 blocks.
   - GPU test `block32`: parity with the BF16 GEMV and row invariance at nine
     V4.1 shapes, and on eleven of the release's own projections (attention,
     shared experts, indexer, Engram, DSpark).
   - Every UE8M0 scale in the release is inside the range where the BF16
     expansion is exact: FP8 dense 2^-13 to 2^-6, routed experts 2^-16 to 2^0,
     Engram 2^-18 to 2^-4.

   Not done: `--check-kernels` resolving every handle. It boots `build_model`,
   which loads weights, so it needs the loader below and, for the full model,
   all four Sparks. Prefill needs a 32-block FP8 GEMM; V4's are 128-block.

   What a `deepseek_v41` loader needs (a read of the V4 path, 2026-11-15):
   - Reusable as they are: embed, norm and head names; the attention
     projections, norms and attention sink; the mHC tensors; the router; routed
     experts (V4's native MXFP4 path, `quantized_mxfp4_e8m0`, reads the same
     I8 + E8M0 layout).
   - Decisions on `model_type`: the factory arm, vanilla norm weights, the
     E8M0 expert kernel series, the KV-only prefix cache rule.
   - Different in V4.1: the compressor exists only on KV source layers, has no
     `ape`, and `wgate` only where the ratio is above 1; there is no model-level
     `hc_head`; FP8 projections must go to the MXFP8 path, not V4's FP8 twins,
     which would load and compute wrong values.
   - No V4 equivalent: indexer weights, Engram, the image routing bias, the
     DSpark stages, the vision tower.
7. **Per-sequence attention cache.** Paged window ring (128 slots) per layer,
   compressed FP4 KV and index keys per source layer per sequence, compressor
   partial-group state per sequence; correct across chunked prefill, verify
   and batching. Spark: one-layer numerics against the reference forward on
   recorded activations.

   The reference side is recorded
   ([results/2026-11-15-reference-layers](../results/2026-11-15-reference-layers/RESULT.md)):
   layers 0 to 2 of DeepSeek's code on the release's weights, 1,401 tokens
   prefilled and 48 decoded. What it fixes for this step:
   - The window ring and the compressed KV reproduce bit for bit from a short
     recipe, so the engine's caches are compared for equality.
   - Attention output is accepted within about 1.3e-3 relative of the
     reference on the reference's own inputs. Tighter needs the reference's
     BF16 weight rounding and 64-key schedule.
   - The reference has no chunked prefill, and its own decode and prefill
     paths are 5% apart by layer 2. Chunked prefill is checked against a
     single prefill, and no check compares whole-model logits tightly.
   - The reference's 64-head attention kernel does not fit a GB10's shared
     memory; it ran 16 heads at a time.
8. **Indexer and candidate blocks.** Sources compute scores (TP-split heads,
   summed across ranks), consumers reuse top-k; layer 20 picks candidate
   blocks. Spark: top-k agreement with the reference past 32K tokens.
9. **Engram layer.** Generalize the PLE layer to two Engram layers, gather FP8
   rows with E8M0 group scales from the NVMe row cache, gate per mHC stream
   (signed-sqrt sigmoid), zero the gate on image tokens. Spark: layer output
   against the reference; cache hit rate on agent traffic.
10. **TP=2 and TP=4.** Head-split attention and indexer, EP experts, vocab-split
    embed/head; latent KV replicated. Spark: TP2 and TP4 logits against TP1 of
    a 4-layer slice; full model on four Sparks.
11. **DSpark drafter.** Three stages over target layers 37-39, Markov head,
    confidence head; exact verify. Spark: greedy output identical with and
    without drafting; acceptance length.
12. **EXL3 MoE for TP2.** Grouped EXL3 expert GEMM (no requant to NVFP4, which
    would not fit), and TP slicing of EXL3 dense tensors. Spark: KL against the
    release on bertholomus's row set.
13. **Vision.** ViT, aligner, VL routing bias.

The Strict TP4 lane (steps 3-11, 13) does not depend on step 12.

## NVIDIA's NVFP4 pack: an optional TP4 lane

`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4` is the same 1017 revision with NVFP4
routed experts and lossy MXFP4 Engram tables
([results/2026-10-30-nvfp4-1017](../results/2026-10-30-nvfp4-1017/RESULT.md)).
It costs 4.0 GiB more per rank than the release, and its Engram tables are half
the size, so the same cache holds about twice the rows. The lossy tables keep it
out of the Strict lane: it stays off by default and is adopted only if it wins.

Beyond the Strict steps it needs:

- **Experts.** Load `weight_scale`, `weight_scale_2` and `input_scale` for the
  main model while DSpark experts keep MXFP4 and `scale`. Atlas has both V4's
  NVFP4 W4A16 kernels and a W4A4 grouped kernel
  (`nemotron-labs-3-puzzle-75b-a9b/nvfp4/moe_w4a4_grouped.cu`). NVIDIA calibrated
  and evaluated W4A4, so that is the mode its figures describe. Compare both
  against the release's logits before choosing.
- **Engram.** Step 9 over MXFP4 rows: 136 bytes each, 256 FP4 values with an
  E8M0 scale per 32.

Accept: on the same Sparks, KL and top-1 against the release, decode speed, and
Engram cache hit rate, with the drafter on. NVIDIA did not exercise it.
