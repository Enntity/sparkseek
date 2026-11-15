# SPDX-License-Identifier: AGPL-3.0-only
"""Record what DeepSeek's reference V4.1 code computes in its first layers, on real weights.

Runs `inference/model.py` (unmodified, imported from --ref) as a model truncated to the first
--layers blocks, on real tokens, and saves each attention layer's input, output, sparse-attention
operands and caches. Two runs over the same tokens: A prefills P tokens then decodes one token
at a time; B prefills all of them at once.

The only change to the reference is the Engram table: the released table is 98 GB, so the rows
this prompt reads are gathered from the checkpoint file and looked up through an index. The
dequantization after the lookup is the reference's.

  python record.py --ref /ref --slice /slice/model0-mp1.safetensors --ckpt /ckpt --out /out
"""

import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_model, save_file
from tokenizers import Tokenizer
from torch import nn

p = argparse.ArgumentParser()
p.add_argument("--ref", required=True)
p.add_argument("--slice", required=True, help="convert.py output for the layers' files, mp=1")
p.add_argument("--ckpt", required=True, help="the released checkpoint directory")
p.add_argument("--out", required=True)
p.add_argument("--layers", type=int, default=3)
p.add_argument("--prefill", type=int, default=1401)
p.add_argument("--decode", type=int, default=48)
p.add_argument("--no-engram", action="store_true")
p.add_argument("--text", nargs="+", required=True, help="text files that supply the tokens")
a = p.parse_args()

sys.path.insert(0, a.ref)
import model as M  # noqa: E402

ENGRAM_FILE = "model-00047-of-00048.safetensors"


class Tok:
    """What NgramHashState reads from a transformers tokenizer."""

    def __init__(self, path):
        self.backend_tokenizer = Tokenizer.from_file(path)

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


class SparseEngramEmbedding(nn.Module):
    """ParallelEngramEmbedding holding only the rows that are read. `forward` is the reference's
    after the row lookup."""

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.num_embeddings, self.dim, self.block_size = num_embeddings, dim, M.fp8_block_size
        self.ids = self.weight = self.scale = None

    def set_rows(self, ids, weight, scale):
        self.ids, self.weight, self.scale = ids, weight, scale

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        at = torch.searchsorted(self.ids, indices).clamp_max(self.ids.numel() - 1)
        assert (self.ids[at] == indices).all(), "an Engram row that was not gathered"
        values = F.embedding(at, self.weight)
        scales = F.embedding(at, self.scale)
        values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16)


def tensor_span(path, name):
    """(dtype, shape, absolute byte offset) of one tensor in a safetensors file."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(n))[name]
    return meta["dtype"], meta["shape"], 8 + n + meta["data_offsets"][0]


def gather_rows(path, name, rows):
    dtype, shape, offset = tensor_span(path, name)
    table = np.memmap(path, dtype=np.uint8, mode="r", offset=offset, shape=tuple(shape))
    return dtype, torch.from_numpy(np.ascontiguousarray(table[rows]))


torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(0)
torch.cuda.set_device(0)

cfg = json.load(open(os.path.join(a.ref, "config.json")))
cfg.update(
    n_layers=a.layers,
    n_mtp_layers=0,
    dspark_block_size=0,
    dspark_target_layer_ids=[],
    vision_n_layers=0,
    max_batch_size=1,
    max_seq_len=4096,
    temperature=0,
)
if a.no_engram:
    cfg["engram_layer_ids"] = []
args = M.ModelArgs(**cfg)
tok = Tok(os.path.join(a.ckpt, "tokenizer.json"))
M.ParallelEngramEmbedding = SparseEngramEmbedding
with torch.device("cuda"):
    net = M.Transformer(args, tok)
missing, unexpected = load_model(net, a.slice, strict=False, device="cuda")
torch.set_default_device("cuda")

# Tokens: BOS, then the text files' own tokens.
total = a.prefill + a.decode
text = "\n\n".join(open(f).read() for f in a.text)
ids = tok.backend_tokenizer.encode(text, add_special_tokens=False).ids
bos = tok.backend_tokenizer.token_to_id("<｜begin▁of▁sentence｜>")
assert bos is not None and len(ids) >= total - 1, (bos, len(ids))
ids = torch.tensor([[bos] + ids[: total - 1]], dtype=torch.long)

engram_layers = [i for i, layer in enumerate(net.layers) if layer.engram is not None]
if engram_layers:
    path = os.path.join(a.ckpt, ENGRAM_FILE)
    hashes = net.engram_hash(ids, 0)
    with safe_open(path, framework="pt", device="cuda") as f:
        for i in engram_layers:
            engram = net.layers[i].engram
            prefix = f"layers.{i}.engram."
            for key in ("wkv.weight", "wkv.scale", "q_weight", "k_weight"):
                target = engram.get_parameter(key)
                source = f.get_tensor(prefix + key)
                assert source.shape == target.shape and source.dtype == target.dtype, key
                target.data.copy_(source)
                missing.remove(prefix + key)
            rows = hashes[0, :, engram.layer_hash_index, :].unique()
            order = rows.cpu().numpy()
            wd, weight = gather_rows(path, prefix + "embed.weight", order)
            sd, scale = gather_rows(path, prefix + "embed.scale", order)
            assert (wd, sd) == ("F8_E4M3", "F8_E8M0"), (wd, sd)
            engram.embed.set_rows(
                rows, weight.cuda().view(torch.float8_e4m3fn), scale.cuda().view(torch.float8_e8m0fnu)
            )
            print(f"layer {i}: {rows.numel()} Engram rows gathered")

# Nothing a recorded value depends on may be left unloaded.
allowed = ("head.", "norm.")
assert all(k.startswith(allowed) for k in missing), [k for k in missing if not k.startswith(allowed)]
assert all(k.endswith("gate.bias_vl") or int(k.split(".")[1]) >= a.layers for k in unexpected if k.startswith("layers.")), unexpected[:5]
print("left unloaded (never recorded):", sorted(missing))

rec, cur = {}, {"layer": None}


def keep(name, t):
    rec.setdefault(name, []).append(None if t is None else t.detach().clone().cpu())


for i, layer in enumerate(net.layers):
    L = f"L{i}."
    layer.register_forward_pre_hook(lambda m, x, L=L: (keep(L + "block_in", x[0]), keep(L + "pre_mix_in", x[2])) and None)
    layer.register_forward_hook(lambda m, x, y, L=L: (keep(L + "block_out", y[0]), keep(L + "pre_mix_out", y[1])) and None)
    layer.attn.register_forward_pre_hook(lambda m, x, L=L, i=i: (cur.update(layer=i), keep(L + "attn_in", x[0])) and None)
    layer.attn.register_forward_hook(lambda m, x, y, L=L: keep(L + "attn_out", y))
    if layer.engram is not None:
        layer.engram.register_forward_pre_hook(lambda m, x, L=L: (keep(L + "engram_in", x[0]), keep(L + "engram_hash_ids", x[1])) and None)
        layer.engram.register_forward_hook(lambda m, x, y, L=L: keep(L + "engram_out", y))
    if layer.attn.compressor is not None:
        layer.attn.compressor.register_forward_hook(lambda m, x, y, L=L: keep(L + "compressor_latent", y))
    if layer.attn.indexer is not None:
        layer.attn.indexer.register_forward_hook(lambda m, x, y, L=L: keep(L + "indexer_topk", y))

reference_sparse_attn = M.sparse_attn
operands = {"on": True}


# The reference kernel asks for 141 KB of shared memory per block at 64 heads of 512, more than a
# GB10 allows. Heads are independent in it, so it runs here on HEADS_PER_CALL heads at a time.
HEADS_PER_CALL = 16


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    o = torch.cat(
        [
            reference_sparse_attn(
                q[:, :, h : h + HEADS_PER_CALL].contiguous(),
                kv,
                attn_sink[h : h + HEADS_PER_CALL].contiguous(),
                topk_idxs,
                softmax_scale,
            )
            for h in range(0, q.size(2), HEADS_PER_CALL)
        ],
        dim=2,
    )
    L = f"L{cur['layer']}."
    keep(L + "sa_idxs", topk_idxs)
    if operands["on"]:
        keep(L + "sa_q", q), keep(L + "sa_kv", kv), keep(L + "sa_o", o)
    return o


M.sparse_attn = sparse_attn


def reset():
    """Every cache back to its initial value, so a run does not see the one before it."""
    for layer in net.layers:
        attn = layer.attn
        attn.window_kv_cache.zero_()
        if attn.compressor is not None:
            attn.compress_kv_cache.zero_()
            if attn.compress_ratio > 1:
                attn.compressor.kv_state.zero_()
                attn.compressor.score_state.fill_(-torch.inf)
        if attn.indexer is not None and attn.indexer.owns_k:
            attn.indexer.k_cache.zero_()


def caches(tag, length):
    """The caches as they stand after `length` tokens."""
    out = {}
    for i, layer in enumerate(net.layers):
        attn = layer.attn
        out[f"L{i}.window_kv_cache"] = attn.window_kv_cache
        if attn.compressor is not None:
            n = length // attn.compress_ratio
            out[f"L{i}.compress_kv_cache"] = attn.compress_kv_cache[:, :n]
            if attn.compress_ratio > 1:
                out[f"L{i}.compressor_kv_state"] = attn.compressor.kv_state
                out[f"L{i}.compressor_score_state"] = attn.compressor.score_state
            if attn.indexer is not None and attn.indexer.owns_k:
                out[f"L{i}.indexer_k_cache"] = attn.indexer.k_cache[:, :n]
    return {f"{tag}.{k}": v.detach().clone().cpu() for k, v in out.items()}


def flush():
    """Recorded lists as tensors: one entry stays as it is, several of one shape are stacked."""
    out = {}
    for name, items in rec.items():
        if any(t is None for t in items):
            out[name + ".present"] = torch.tensor([t is not None for t in items])
            items = [t for t in items if t is not None]
        if not items:
            continue
        if len(items) == 1:
            out[name] = items[0]
        elif all(t.shape == items[0].shape for t in items):
            out[name] = torch.stack(items)
        else:
            for step, t in enumerate(items):
                out[f"{name}.{step}"] = t
    rec.clear()
    return {k: v.contiguous() for k, v in out.items()}


def run_a():
    reset()
    net(ids[:, : a.prefill], 0)
    prefill, after_prefill = flush(), caches("after_prefill", a.prefill)
    for pos in range(a.prefill, total):
        net(ids[:, pos : pos + 1], pos)
    return prefill, flush(), {**after_prefill, **caches("after_decode", total)}


os.makedirs(a.out, exist_ok=True)
with torch.inference_mode():
    prefill, decode, cache = run_a()
    again = run_a()
    same = all(
        x.keys() == y.keys() and all(torch.equal(x[k].view(torch.uint8), y[k].view(torch.uint8)) for k in x)
        for x, y in zip((prefill, decode, cache), again)
    )
    del again
    operands["on"] = False
    reset()
    net(ids, 0)
    full, full_cache = flush(), caches("after_prefill", total)

# The reference against itself: positions decoded one at a time in A, prefilled in B.
spread = {}
for i in range(a.layers):
    one = decode[f"L{i}.attn_out"].float().flatten(0, -2)
    all_ = full[f"L{i}.attn_out"].float()[0, a.prefill :]
    spread[f"L{i}"] = {
        "attn_out_rel_l2_decode_vs_prefill": ((one - all_).norm() / all_.norm()).item(),
        "attn_in_bit_equal": bool(torch.equal(decode[f"L{i}.attn_in"].flatten(0, -2), full[f"L{i}.attn_in"][0, a.prefill :])),
    }

    spread[f"L{i}"]["caches_bit_equal_decode_vs_prefill"] = {
        k.split(".", 2)[2]: bool(torch.equal(v.view(torch.uint8), full_cache[k.replace("after_decode", "after_prefill")].view(torch.uint8)))
        for k, v in cache.items()
        if k.startswith(f"after_decode.L{i}.")
    }

files = {"A_prefill": prefill, "A_decode": decode, "A_caches": cache, "B_prefill": full, "B_caches": full_cache}
shapes = {}
for name, tensors in files.items():
    save_file(tensors, os.path.join(a.out, name + ".safetensors"))
    shapes[name] = {k: [str(v.dtype).removeprefix("torch."), list(v.shape)] for k, v in tensors.items()}
meta = {
    "layers": a.layers,
    "prefill": a.prefill,
    "decode": a.decode,
    "engram": bool(engram_layers),
    "token_ids": ids[0].tolist(),
    "token_ids_sha256": hashlib.sha256(np.asarray(ids[0].tolist(), dtype="<i4").tobytes()).hexdigest(),
    "text_files": [os.path.basename(f) for f in a.text],
    "run_a_repeats_bit_for_bit": same,
    "reference_decode_vs_prefill": spread,
    "left_unloaded": sorted(missing),
    "sparse_attn_heads_per_call": HEADS_PER_CALL,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "tensors": shapes,
}
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print(json.dumps({k: meta[k] for k in ("run_a_repeats_bit_for_bit", "reference_decode_vs_prefill")}, indent=1))
print("RECORDED")
