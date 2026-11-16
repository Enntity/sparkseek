# SPDX-License-Identifier: AGPL-3.0-only
"""Replay the reference indexer's scoring on a recording, and save what it scores.

`record.py --ops` saves the indexer's projections and its top-k, not the scores between them.
This runs the reference's own lines (model.py, Indexer.forward) on the recorded projections,
checks that the top-k they give is the recorded one, and saves the quantized queries, the head
weights and the scores. It also checks a statement of the arithmetic in which each rounding step
is explicit, and how equal scores at the cut are resolved.

  python index_replay.py --ref /ref --rec /rec --layer 2
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

p = argparse.ArgumentParser()
p.add_argument("--ref", required=True)
p.add_argument("--rec", required=True)
p.add_argument("--layer", type=int, default=2)
a = p.parse_args()

sys.path.insert(0, a.ref)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_default_dtype(torch.bfloat16)
torch.cuda.set_device(0)
import model as M  # noqa: E402
from kernel import fp4_act_quant  # noqa: E402

cfg = json.load(open(os.path.join(a.ref, "config.json")))
meta = json.load(open(os.path.join(a.rec, "meta.json")))
ratio = cfg["compress_ratios"][a.layer]
heads, width, rope, topk_limit = cfg["index_n_heads"], cfg["index_head_dim"], cfg["rope_head_dim"], cfg["index_topk"]
window = cfg["window_size"]
L = f"L{a.layer}."
OP = L + "op.attn.indexer."
freqs = M.precompute_freqs_cis(
    rope, 4096, cfg["original_seq_len"], cfg["compress_rope_theta"], cfg["rope_factor"], cfg["beta_fast"], cfg["beta_slow"]
).cuda()
weight_scale = width**-0.5 * heads**-0.5

report, out = {}, {}


def reference_scores(q, index_k, weights, start_pos):
    """model.py Indexer.forward from the query projection to the top-k, at world size 1."""
    seqlen, end_pos = q.size(1), start_pos + q.size(1)
    q = q.clone().unflatten(-1, (heads, width))
    M.apply_rotary_emb(q[..., -rope:], freqs[start_pos:end_pos])
    fp4_act_quant(q, M.fp4_block_size, True)
    weights = weights * weight_scale
    per_head = torch.einsum("bshd,btd->bsht", q, index_k)
    index_score = (per_head.clone().relu_() * weights.unsqueeze(-1)).sum(dim=2)
    if start_pos == 0:
        compress_lens = (torch.arange(1, seqlen + 1, device=q.device) // ratio).unsqueeze(-1)
        index_score.masked_fill_(torch.arange(seqlen // ratio, device=q.device) >= compress_lens, -torch.inf)
        offset = seqlen
    else:
        compress_lens = end_pos // ratio
        offset = window
    topk = min(topk_limit, end_pos // ratio)
    idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    idxs = torch.where(idxs < compress_lens, idxs + offset, -1).int()
    return q, weights, per_head, index_score, idxs, compress_lens, offset


def explicit(q, index_k, weights, per_head, index_score):
    """The same scores with every rounding step written out. Returns how many values differ."""
    dot = torch.einsum("bshd,btd->bsht", q.double(), index_k.double())
    exact_in_f32 = bool((dot.float().double() == dot).all())
    head = dot.float().to(torch.bfloat16)
    head_diff = int((head != per_head).sum())
    weighted = (head.float().relu() * weights.float().unsqueeze(-1)).to(torch.bfloat16)
    total = weighted.float().sum(dim=2).to(torch.bfloat16)
    seen = torch.isfinite(index_score)
    return exact_in_f32, head_diff, int((total != index_score)[seen].sum()), int(seen.sum())


def ties(index_score, idxs, compress_lens, offset):
    """Rows whose cut falls among equal scores, and which of the equal ones were kept."""
    score = index_score.float().flatten(0, -2)
    picked = idxs.flatten(0, -2)
    lens = compress_lens.flatten().tolist() if torch.is_tensor(compress_lens) else [compress_lens] * score.size(0)
    rows = lowest = highest = 0
    for r, n in enumerate(lens):
        if n <= topk_limit:
            continue
        s = score[r, :n]
        cut = s.topk(topk_limit).values.min()
        equal = (s == cut).nonzero().flatten()
        above = int((s > cut).sum())
        need = topk_limit - above
        if equal.numel() == need:
            continue
        rows += 1
        kept = set((picked[r][picked[r] >= 0] - offset).tolist()) & set(equal.tolist())
        lowest += kept == set(equal[:need].tolist())
        highest += kept == set(equal[-need:].tolist())
    return {"rows_with_a_tie_at_the_cut": rows, "kept_lowest_positions": lowest, "kept_highest_positions": highest}


def keys_from_latents(normed, first_positions):
    """model.py Indexer.forward's index keys from `k_norm`'s output."""
    k = normed.clone()
    M.apply_rotary_emb(k[..., -rope:], freqs[first_positions])
    fp4_act_quant(k, M.fp4_block_size, True)
    return k


with torch.inference_mode():
    P = load_file(os.path.join(a.rec, "A_prefill.safetensors"))
    C = load_file(os.path.join(a.rec, "A_caches.safetensors"))
    D = load_file(os.path.join(a.rec, "A_decode.safetensors"))
    prefill = meta["prefill"]

    # Index keys.
    cache = C[f"after_decode.{L}indexer_k_cache"].cuda()
    groups = prefill // ratio
    k = keys_from_latents(P[OP + "k_norm.out"].cuda(), torch.arange(groups, device="cuda") * ratio)
    report["prefill_keys_bit_equal"] = bool(torch.equal(k, cache[:, :groups]))

    # Prefill.
    q, w, per_head, score, idxs, lens, offset = reference_scores(
        P[OP + "wq_b.out"].cuda(), cache[:, :groups], P[OP + "weights_proj.out"].cuda(), 0
    )
    recorded = P[L + "indexer_topk"].cuda()
    report["prefill"] = {
        "queries": q.size(1),
        "topk_rows_equal_recorded": int((idxs == recorded).all(dim=-1).sum()),
        "ties": ties(score, idxs, lens, offset),
    }
    ok, head_diff, diff, seen = explicit(q, cache[:, :groups], w, per_head, score)
    report["prefill"]["explicit"] = {"dot_exact_in_f32": ok, "per_head_differ": head_diff, "scores_differ": diff, "scores": seen}
    out["prefill.q"], out["prefill.weights"], out["prefill.score"] = q[0].cpu(), w[0].cpu(), score[0].cpu()

    # Decode, one token at a time.
    steps = D[OP + "wq_b.out"].size(0)
    rows_equal, tie = 0, {}
    stats = {"per_head_differ": 0, "scores_differ": 0, "scores": 0, "dot_exact_in_f32": True}
    padded = torch.full((steps, cache.size(1)), -torch.inf)
    qs, ws = [], []
    for step in range(steps):
        pos = prefill + step
        n = (pos + 1) // ratio
        q, w, per_head, score, idxs, lens, offset = reference_scores(
            D[OP + "wq_b.out"][step].cuda(), cache[:, :n], D[OP + "weights_proj.out"][step].cuda(), pos
        )
        rows_equal += int((idxs == D[L + "indexer_topk"][step].cuda()).all())
        for name, v in ties(score, idxs, lens, offset).items():
            tie[name] = tie.get(name, 0) + v
        ok, head_diff, diff, seen = explicit(q, cache[:, :n], w, per_head, score)
        stats["dot_exact_in_f32"] &= ok
        stats["per_head_differ"] += head_diff
        stats["scores_differ"] += diff
        stats["scores"] += seen
        padded[step, :n] = score[0, 0].cpu()
        qs.append(q[0, 0].cpu()), ws.append(w[0, 0].cpu())
    report["decode"] = {"queries": steps, "topk_rows_equal_recorded": rows_equal, "ties": tie, "explicit": stats}
    out["decode.q"], out["decode.weights"], out["decode.score"] = torch.stack(qs), torch.stack(ws), padded

save_file({k: v.contiguous() for k, v in out.items()}, os.path.join(a.rec, "A_index.safetensors"))
report["tensors"] = {k: [str(v.dtype).removeprefix("torch."), list(v.shape)] for k, v in out.items()}
json.dump(report, open(os.path.join(a.rec, "index.json"), "w"), indent=1)
print(json.dumps(report, indent=1))
print("REPLAYED")
