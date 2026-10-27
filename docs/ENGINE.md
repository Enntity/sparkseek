<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# The engine: DeepSeek-V4.1-Flash on Atlas

SparkSeek serves DeepSeek-V4.1-Flash-1017 with
[Atlas](https://github.com/Enntity/atlas), like SparkGLM and SparkQwen. This
page records what Atlas already has, what V4.1 needs that it does not, and the
order the work lands in. Status as of 2026-10-26: CPU-side groundwork only; no
GPU code has been written or run.

## Branch

`upstream/deepseek-v41-flash-20261026` in Enntity/atlas, cut from Atlas-Inf
`main` at `504063ad`, so the series stays proposable upstream. The product
build will stack it on the shared engine layer SparkGLM and SparkQwen use
(`upstream/atlas-shared-*`) once there is something to serve. Not pushed yet.

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

## Next, in order

Each step names the check that accepts it. "Spark" marks the first step that
needs one.

3. **Compressed vocab.** Build Engram's token map from `tokenizer.json` with
   the `tokenizers` crate's normalizers (NFKC, NFD, strip accents, lowercase,
   whitespace collapse) in `spark-server`. Accept: 99,092 ids and SHA-256
   `c60a8632…59fe46481` of the little-endian i32 map for the 1017 tokenizer.
4. **V4.1 prompt encoding and DSML parser.** Port `encoding/encoding.py` beside
   the V4 renderer, select it on `deepseek_v41`. Accept: DeepSeek's five
   `encoding/tests` pairs byte for byte, and the tool parser on the new tag
   names.
5. **Weight manifest and preflight.** Expected names, dtypes and shapes for the
   release and the EXL3 2.5 bpw pack, checked against the headers; the 32x32
   FP8 scale shapes; DSpark stage tensors. Accept: the manifest matches both
   checkpoints' headers with no extras.
6. **Kernel target.** `kernels/gb10/deepseek-v4.1-flash/` (`model_type
   deepseek_v41`, `hidden_size 5120`) layering V4's MoE and mHC kernels, plus a
   32-wide-block FP8 GEMV. Spark: `--check-kernels` resolves every handle.
7. **Per-sequence attention cache.** Paged window ring (128 slots) per layer,
   compressed FP4 KV and index keys per source layer per sequence, compressor
   partial-group state per sequence; correct across chunked prefill, verify
   and batching. Spark: one-layer numerics against the reference forward on
   recorded activations.
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
