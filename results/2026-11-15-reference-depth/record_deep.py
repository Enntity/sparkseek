# SPDX-License-Identifier: AGPL-3.0-only
"""Record what DeepSeek's reference V4.1 code computes at full depth, on one Spark.

The 40 blocks do not fit one Spark's memory together, so they run one at a time: a block is built,
loaded from its own file of the release (through the reference's convert.py), run over the whole
prompt -- one prefill, then the decode steps one token at a time -- recorded, and freed. What the
blocks hand each other is kept between them: the stream and Hyper-Connection mix of every step,
and the reference's shared attention runtime (compressed keys, index keys, top-k and candidates)
as it stood after the block before, per step.

The blocks, their code and the order of their calls are the reference's (`inference/model.py`,
imported from --ref). Changed from running it as shipped: the same three things as record.py
(sparse Engram table, sparse attention 16 heads at a time, TF32 off), and this loop in place of
`Transformer.forward`, whose lines it follows.

With --against, every tensor that an earlier record.py recording also holds is compared with it
bit for bit: layers 0 to 2 here must be that recording.

  python record_deep.py --ref /ref --ckpt /ckpt --work /scratch --out /out --ops \
      --text /ref/README.md /ref/model.py [--layers 0:40] [--against rec] [--resume]
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_model, save_file
from tokenizers import Tokenizer
from torch import nn

p = argparse.ArgumentParser()
p.add_argument("--ref", required=True)
p.add_argument("--ckpt", required=True, help="the released checkpoint directory")
p.add_argument("--work", required=True, help="scratch space for one converted layer at a time")
p.add_argument("--out", required=True)
p.add_argument("--layers", default="0:40", help="first:end of the blocks to run; first > 0 needs --resume")
p.add_argument("--prefill", type=int, default=1401)
p.add_argument("--decode", type=int, default=48)
p.add_argument("--ops", action="store_true", help="also save every projection, norm, gate and Hyper-Connection step")
p.add_argument("--against", help="a record.py recording to compare the layers it holds with")
p.add_argument("--resume", action="store_true", help="continue from the state the last finished block left in --work")
p.add_argument("--text", nargs="+", required=True, help="text files that supply the tokens")
a = p.parse_args()

sys.path.insert(0, a.ref)
import convert  # noqa: E402
import model as M  # noqa: E402

# See record.py: NVIDIA's containers default to TF32, the reference is written for stock PyTorch.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(0)
torch.cuda.set_device(0)


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


class Stack(nn.Module):
    """One block under the name it has in a checkpoint: `layers.<i>`."""

    def __init__(self, i: int, block: nn.Module):
        super().__init__()
        self.layers = nn.ModuleDict({str(i): block})


def gather_rows(path, name, rows):
    """Rows of one tensor of a safetensors file, read from disk."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(n))[name]
    table = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n + meta["data_offsets"][0], shape=tuple(meta["shape"]))
    return meta["dtype"], torch.from_numpy(np.ascontiguousarray(table[rows]))


cfg = json.load(open(os.path.join(a.ref, "config.json")))
n_layers = cfg["n_layers"]
cfg.update(
    n_mtp_layers=0,
    dspark_block_size=0,
    dspark_target_layer_ids=[],
    vision_n_layers=0,
    max_batch_size=1,
    max_seq_len=4096,
    temperature=0,
)
args = M.ModelArgs(**cfg)
assert args.dtype == "fp8" and M.world_size == 1 and M.default_dtype == torch.float8_e4m3fn
first, end = (int(v) for v in a.layers.split(":"))
assert 0 <= first < end <= n_layers and (first == 0 or a.resume), a.layers
weight_map = json.load(open(os.path.join(a.ckpt, "model.safetensors.index.json")))["weight_map"]
tok = Tok(os.path.join(a.ckpt, "tokenizer.json"))
M.ParallelEngramEmbedding = SparseEngramEmbedding
torch.set_default_device("cuda")
os.makedirs(a.out, exist_ok=True)
os.makedirs(a.work, exist_ok=True)


def released(name):
    """One tensor of the release, on the GPU."""
    with safe_open(os.path.join(a.ckpt, weight_map[name]), framework="pt", device="cuda") as f:
        return f.get_tensor(name)


# Tokens: BOS, then the text files' own tokens (record.py's).
total = a.prefill + a.decode
text = "\n\n".join(open(f).read() for f in a.text)
ids = tok.backend_tokenizer.encode(text, add_special_tokens=False).ids
bos = tok.backend_tokenizer.token_to_id("<｜begin▁of▁sentence｜>")
assert bos is not None and len(ids) >= total - 1, (bos, len(ids))
ids = torch.tensor([[bos] + ids[: total - 1]], dtype=torch.long)
# Step 0 is the prefill; step s > 0 decodes the token at position prefill + s - 1.
spans = [(0, a.prefill)] + [(pos, pos + 1) for pos in range(a.prefill, total)]

layout = M.EngramLayout.from_args(args)
hasher = M.NgramHashState(args, layout, tok)
all_hashes = hasher(ids, 0)
hashes = [hasher(ids[:, lo:hi], lo) for lo, hi in spans]
assert torch.equal(torch.cat(hashes, dim=1), all_hashes), "n-gram hashes depend on how the prompt is split"

rec, cur = {}, {"layer": None}


def keep(name, t):
    rec.setdefault(name, []).append(None if t is None else t.detach().clone().cpu())


def flush():
    """Recorded lists as tensors: one entry stays as it is, several of one shape are stacked."""
    out = {}
    for name, items in rec.items():
        if any(t is None for t in items):
            out[name + ".present"] = torch.tensor([t is not None for t in items], device="cpu")
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


# The reference kernel asks for 141 KB of shared memory per block at 64 heads of 512, more than a
# GB10 allows. Heads are independent in it, so it runs here on HEADS_PER_CALL heads at a time.
HEADS_PER_CALL = 16
reference_sparse_attn = M.sparse_attn


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
    keep(L + "sa_idxs", topk_idxs), keep(L + "sa_q", q), keep(L + "sa_kv", kv), keep(L + "sa_o", o)
    return o


M.sparse_attn = sparse_attn

# Hyper-Connection sites in the order a block reaches them.
SITES = ("attn", "ffn")


def hook(i, layer):
    """record.py's hooks, on one block."""
    L = f"L{i}."
    layer.register_forward_pre_hook(lambda m, x: (keep(L + "block_in", x[0]), keep(L + "pre_mix_in", x[2])) and None)
    layer.register_forward_hook(lambda m, x, y: (keep(L + "block_out", y[0]), keep(L + "pre_mix_out", y[1])) and None)
    layer.attn.register_forward_pre_hook(lambda m, x: (cur.update(layer=i), keep(L + "attn_in", x[0])) and None)
    layer.attn.register_forward_hook(lambda m, x, y: keep(L + "attn_out", y))
    if layer.engram is not None:
        layer.engram.register_forward_pre_hook(lambda m, x: (keep(L + "engram_in", x[0]), keep(L + "engram_hash_ids", x[1])) and None)
        layer.engram.register_forward_hook(lambda m, x, y: keep(L + "engram_out", y))
    if layer.attn.compressor is not None:
        layer.attn.compressor.register_forward_hook(lambda m, x, y: keep(L + "compressor_latent", y))
    if layer.attn.indexer is not None:
        layer.attn.indexer.register_forward_hook(lambda m, x, y: keep(L + "indexer_topk", y))
    if not a.ops:
        return
    O = L + "op."
    for path, module in layer.named_modules():
        if ".experts." in path or not path:
            continue
        if isinstance(module, (M.Linear, M.RMSNorm, SparseEngramEmbedding, M.MoE)):
            module.register_forward_pre_hook(lambda m, x, n=O + path: keep(n + ".in", x[0]))
            module.register_forward_hook(lambda m, x, y, n=O + path: keep(n + ".out", y))
        elif isinstance(module, M.Gate):
            module.register_forward_hook(lambda m, x, y, n=O + path: (keep(n + ".weights", y[0]), keep(n + ".indices", y[1])) and None)
    site = {"n": 0}
    layer.register_forward_pre_hook(lambda m, x: site.update(n=0))

    def mixes(*args, inner=layer.hc_mixes):
        pre, post, comb = inner(*args)
        name = f"{O}hc.{SITES[site['n']]}."
        keep(name + "pre", pre), keep(name + "post", post), keep(name + "comb", comb)
        return pre, post, comb

    def collapse(*args, inner=layer.hc_pre):
        y = inner(*args)
        keep(f"{O}hc.{SITES[site['n']]}.collapsed", y)
        return y

    def expand(*args, inner=layer.hc_post):
        y = inner(*args)
        keep(f"{O}hc.{SITES[site['n']]}.expanded", y)
        site["n"] += 1
        return y

    layer.hc_mixes, layer.hc_pre, layer.hc_post = mixes, collapse, expand


def caches(i, layer, tag, length):
    """The block's caches as they stand after `length` tokens."""
    attn, out = layer.attn, {}
    out[f"L{i}.window_kv_cache"] = attn.window_kv_cache
    if attn.compressor is not None:
        n = length // attn.compress_ratio
        out[f"L{i}.compress_kv_cache"] = attn.compress_kv_cache[:, :n]
        if attn.compress_ratio > 1:
            out[f"L{i}.compressor_kv_state"] = attn.compressor.kv_state
            out[f"L{i}.compressor_score_state"] = attn.compressor.score_state
        if attn.indexer is not None and attn.indexer.owns_k:
            out[f"L{i}.indexer_k_cache"] = attn.indexer.k_cache[:, :n]
    return {f"{tag}.{k}": v.detach().clone().cpu().contiguous() for k, v in out.items()}


def build(i):
    """Block `i` with the release's weights."""
    shard = weight_map[f"layers.{i}.attn_norm.weight"]
    source, converted = os.path.join(a.work, f"in{i}"), os.path.join(a.work, f"conv{i}")
    for d in (source, converted):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(source)
    os.symlink(os.path.join(a.ckpt, shard), os.path.join(source, shard))
    convert.main(source, converted, 1, "fp4", a.ckpt)
    layer = M.Block(i, args, layout)
    missing, unexpected = load_model(Stack(i, layer), os.path.join(converted, "model0-mp1.safetensors"), strict=False, device="cuda")
    shutil.rmtree(source), shutil.rmtree(converted)
    if layer.engram is not None:
        prefix = f"layers.{i}.engram."
        for key in ("wkv.weight", "wkv.scale", "q_weight", "k_weight"):
            target, value = layer.engram.get_parameter(key), released(prefix + key)
            assert value.shape == target.shape and value.dtype == target.dtype, key
            target.data.copy_(value)
            missing.remove(prefix + key)
        rows = all_hashes[0, :, layer.engram.layer_hash_index, :].unique()
        path = os.path.join(a.ckpt, weight_map[prefix + "embed.weight"])
        wd, weight = gather_rows(path, prefix + "embed.weight", rows.cpu().numpy())
        sd, scale = gather_rows(path, prefix + "embed.scale", rows.cpu().numpy())
        assert (wd, sd) == ("F8_E4M3", "F8_E8M0"), (wd, sd)
        layer.engram.embed.set_rows(rows, weight.cuda().view(torch.float8_e4m3fn), scale.cuda().view(torch.float8_e8m0fnu))
        print(f"layer {i}: {rows.numel()} Engram rows gathered", flush=True)
    # Nothing the block computes with may be left unloaded, and its file holds nothing else's.
    assert not missing, missing[:5]
    assert all(k.startswith(f"layers.{i}.") and k.endswith("gate.bias_vl") for k in unexpected), unexpected[:5]
    return layer


SHARED = ("compress_kv", "index_k", "topk_idxs", "candidates")
state_path = os.path.join(a.work, "state.pt")
if a.resume:
    state = torch.load(state_path, map_location="cuda")
    assert state["next"] == first and state["ids"] == ids[0].tolist(), (state["next"], first)
    streams, mixes, shared = state["streams"], state["mixes"], state["shared"]
else:
    embed = M.ParallelEmbedding(args.vocab_size, args.dim)
    embed.weight.data.copy_(released("embed.weight"))
    with torch.inference_mode():
        streams = [embed(ids[:, lo:hi]).unsqueeze(2).repeat(1, 1, args.hc_mult, 1) for lo, hi in spans]
        mixes = [M.make_identity_pre_mix(h, args.hc_mult) for h in streams]
    shared = [dict.fromkeys(SHARED) for _ in spans]
    del embed

against = {}
if a.against:
    against = {
        part: safe_open(os.path.join(a.against, f"A_{part}.safetensors"), framework="pt", device="cpu")
        for part in ("prefill", "decode", "caches")
    }


def compare(part, tensors):
    """How many of `tensors` the earlier recording holds, and that each is the same bytes."""
    if not against:
        return 0
    theirs, held = against[part], 0
    for name, mine in tensors.items():
        if name in theirs.keys():
            other = theirs.get_tensor(name)
            assert other.dtype == mine.dtype and other.shape == mine.shape, name
            assert torch.equal(other.contiguous().view(torch.uint8), mine.view(torch.uint8)), f"{part} {name} differs"
            held += 1
    return held


shapes, compared = {}, 0
for i in range(first, end):
    began = time.time()
    layer = build(i)
    hook(i, layer)
    parts = {}
    with torch.inference_mode():
        for step, (lo, hi) in enumerate(spans):
            for field in SHARED:
                setattr(M.shared_attn, field, shared[step][field])
            h, pre_mix = streams[step], mixes[step]
            # Transformer.forward's loop body, for a text-only prompt.
            if layer.engram is not None:
                h = layer.engram(h, hashes[step][:, :, layer.engram.layer_hash_index, :], None)
            h, pre_mix = layer(h, lo, pre_mix, None)
            streams[step], mixes[step] = h, pre_mix
            shared[step] = {field: getattr(M.shared_attn, field) for field in SHARED}
            if step == 0:
                parts["prefill"] = flush()
                parts["caches"] = caches(i, layer, "after_prefill", a.prefill)
        parts["decode"] = flush()
        parts["caches"].update(caches(i, layer, "after_decode", total))
        if i == n_layers - 1:
            # After the last block: collapse with the last mix, norm, head.
            norm, head = M.RMSNorm(args.dim, args.norm_eps), M.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
            norm.weight.data.copy_(released("norm.weight"))
            head.weight.data.copy_(released("head.weight"))
            final = {}
            for step in range(len(spans)):
                collapsed = M.Block.hc_pre(layer, streams[step], mixes[step])
                normed = norm(collapsed)
                logits = head(normed, full_logits=True)
                top = logits.topk(8, dim=-1)
                tag = "prefill" if step == 0 else "decode"
                for name, t in (("collapsed", collapsed), ("normed", normed), ("top_logits", top.values), ("top_ids", top.indices), ("last_logits", logits[:, -1])):
                    final.setdefault(f"{tag}.final.{name}", []).append(t.detach().clone().cpu())
            parts["final"] = {k: (v[0] if len(v) == 1 else torch.stack(v)).contiguous() for k, v in final.items()}
            del norm, head
    for part, tensors in parts.items():
        name = "final" if part == "final" else f"L{i:02d}_{part}"
        save_file(tensors, os.path.join(a.out, name + ".safetensors"))
        os.chmod(os.path.join(a.out, name + ".safetensors"), 0o644)
        shapes[name] = {k: [str(v.dtype).removeprefix("torch."), list(v.shape)] for k, v in tensors.items()}
        if part != "final":
            compared += compare(part, tensors)
    del layer, parts
    torch.cuda.empty_cache()
    torch.save({"next": i + 1, "ids": ids[0].tolist(), "streams": streams, "mixes": mixes, "shared": shared}, state_path + ".tmp")
    os.replace(state_path + ".tmp", state_path)
    print(f"layer {i}: recorded in {time.time() - began:.0f} s; {compared} tensors so far equal the earlier recording's", flush=True)

meta_path = os.path.join(a.out, "meta.json")
meta = json.load(open(meta_path)) if a.resume and os.path.exists(meta_path) else {"tensors": {}}
meta["tensors"].update(shapes)
meta.update(
    layers=[0, end],
    prefill=a.prefill,
    decode=a.decode,
    token_ids=ids[0].tolist(),
    token_ids_sha256=hashlib.sha256(np.asarray(ids[0].tolist(), dtype="<i4").tobytes()).hexdigest(),
    text_files=[os.path.basename(f) for f in a.text],
    ops=a.ops,
    sparse_attn_heads_per_call=HEADS_PER_CALL,
    tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
    torch=torch.__version__,
    gpu=torch.cuda.get_device_name(0),
    equal_to_earlier_recording=compared if a.against else None,
)
json.dump(meta, open(meta_path, "w"), indent=1)
print("RECORDED")
