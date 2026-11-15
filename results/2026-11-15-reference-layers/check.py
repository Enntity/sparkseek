# SPDX-License-Identifier: AGPL-3.0-only
"""Recompute parts of a `record.py` recording without DeepSeek's code, and compare.

Nothing here imports the reference. Each check is a transcription of what one piece of the V4.1
attention cache is meant to hold, run against the tensors the reference produced:

  1. sparse attention: output from the recorded q, KV, indices and the layer's sink, two ways --
     plain softmax in float64, and the kernel's schedule (64 keys at a time, running maximum,
     weights rounded to BF16 before they multiply the keys);
  2. window ring: slot `pos % window` holds position `pos`'s key, after prefill and after decode;
  3. compressed KV: a compressor latent, rotated at its group's first position and quantized to
     FP4 in groups of 16 under an E4M3 scale, is what the cache holds.

  python check.py --rec /out --slice /slice/model0-mp1.safetensors --config /ref/config.json
"""

import argparse
import json
import math
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file

p = argparse.ArgumentParser()
p.add_argument("--rec", required=True)
p.add_argument("--slice", required=True)
p.add_argument("--config", required=True)
p.add_argument("--queries", type=int, default=256, help="prefill queries to recompute per layer")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.load(open(a.config))
meta = json.load(open(os.path.join(a.rec, "meta.json")))
P, D, LAYERS = meta["prefill"], meta["decode"], meta["layers"]
WIN, HEAD, ROPE = cfg["window_size"], cfg["head_dim"], cfg["rope_head_dim"]
SCALE = HEAD**-0.5
BF16 = torch.bfloat16
pre = load_file(os.path.join(a.rec, "A_prefill.safetensors"))
dec = load_file(os.path.join(a.rec, "A_decode.safetensors"))
cache = load_file(os.path.join(a.rec, "A_caches.safetensors"))
with safe_open(a.slice, framework="pt") as f:
    sinks = [f.get_tensor(f"layers.{i}.attn.attn_sink").float() for i in range(LAYERS)]


def at_step(name, step):
    """One decode step's tensor: steps of one shape were saved stacked, others one by one."""
    return dec[name][step] if name in dec else dec[f"{name}.{step}"]


def bits(t):
    return t.contiguous().view(torch.int16 if t.dtype == BF16 else torch.uint8)


def rel(x, y):
    x, y = x.double(), y.double()
    return ((x - y).norm() / y.norm()).item()


def attend_exact(q, kv, idxs, sink):
    """q [m,h,d], kv [n,d], idxs [m,k] (-1 = nothing), sink [h] -> o [m,h,d], in float64."""
    q, kv, sink = q.double(), kv.double(), sink.double()
    keys = kv[idxs.clamp_min(0).long()]  # [m,k,d]
    s = torch.einsum("mhd,mkd->mhk", q, keys) * SCALE
    s = s.masked_fill((idxs == -1).unsqueeze(1), -torch.inf)
    top = torch.maximum(s.amax(-1), sink)  # [m,h]
    w = torch.exp(s - top.unsqueeze(-1))
    return torch.einsum("mhk,mkd->mhd", w, keys) / (w.sum(-1) + torch.exp(sink - top)).unsqueeze(-1)


def attend_as_kernel(q, kv, idxs, sink, block=64):
    """The reference kernel's schedule in float32: keys in index order, `block` at a time, with a
    running maximum; the weights that multiply the keys are rounded to BF16, the ones summed for
    the denominator are not."""
    q, kv, sink = q.float(), kv.float(), sink.float()
    m, h, d = q.shape
    acc = q.new_zeros(m, h, d)
    total = q.new_zeros(m, h)
    top = q.new_full((m, h), -1e30)
    for t in range(0, idxs.size(1), block):
        ids = idxs[:, t : t + block]
        live = ids != -1
        keys = kv[ids.clamp_min(0).long()] * live.unsqueeze(-1)
        s = torch.einsum("mhd,mkd->mhk", q, keys).masked_fill(~live.unsqueeze(1), -torch.inf) * SCALE
        before, top = top, torch.maximum(top, s.amax(-1))
        shrink = torch.exp(before - top)
        w = torch.exp(s - top.unsqueeze(-1))
        total = total * shrink + w.sum(-1)
        acc = acc * shrink.unsqueeze(-1) + torch.einsum("mhk,mkd->mhd", w.to(BF16).float(), keys)
    return acc / (total + torch.exp(sink - top)).unsqueeze(-1)


def compare(o_ref, o_exact, o_kernel):
    """Both recomputations rounded to BF16, as the reference's output is."""
    out = {}
    for name, mine in (("exact_math", o_exact.float().to(BF16)), ("kernel_schedule", o_kernel.to(BF16))):
        out[name + "_bf16_bits_equal"] = (bits(mine) == bits(o_ref)).float().mean().item()
        out[name + "_rel_l2"] = rel(mine, o_ref)
    return out


def attention_checks(i):
    L, sink = f"L{i}.", sinks[i].to(dev)
    q, kv, o, idxs = (pre[L + k][0].to(dev) for k in ("sa_q", "sa_kv", "sa_o", "sa_idxs"))
    # spread over the prompt, and always its last positions, where the context is longest
    pick = torch.unique(torch.cat([torch.linspace(0, P - 1, a.queries // 2).long(), torch.arange(P - a.queries // 2, P).clamp_min(0)])).to(dev)
    out = {"prefill_queries": pick.numel()}
    parts_exact, parts_kernel = [], []
    for chunk in pick.split(32):
        parts_exact.append(attend_exact(q[chunk], kv, idxs[chunk], sink))
        parts_kernel.append(attend_as_kernel(q[chunk], kv, idxs[chunk], sink))
    out["prefill"] = compare(o[pick], torch.cat(parts_exact), torch.cat(parts_kernel))

    exact, kernel, ref = [], [], []
    for step in range(D):
        q1, o1, ids1, kv1 = (at_step(L + k, step)[0].to(dev) for k in ("sa_q", "sa_o", "sa_idxs", "sa_kv"))
        exact.append(attend_exact(q1, kv1, ids1, sink))
        kernel.append(attend_as_kernel(q1, kv1, ids1, sink))
        ref.append(o1)
    out["decode"] = compare(torch.cat(ref), torch.cat(exact), torch.cat(kernel))
    return out


def window_checks(i):
    """Slot pos % window holds position pos's key."""
    L = f"L{i}."
    keys = pre[L + "sa_kv"][0, :P]  # prefill keys in position order (compressed ones follow them)
    ring = cache[f"after_prefill.{L}window_kv_cache"][0]
    held = range(max(0, P - WIN), P)
    out = {"after_prefill_bits_equal": all(torch.equal(bits(ring[pos % WIN]), bits(keys[pos])) for pos in held)}
    # decode: each step's ring is the one before it with one slot replaced
    rings = [at_step(L + "sa_kv", step)[0, :WIN] for step in range(D)]
    before, steps_ok = ring, True
    for step, now in enumerate(rings):
        slot = (P + step) % WIN
        others = torch.arange(WIN) != slot
        steps_ok &= torch.equal(bits(now[others]), bits(before[others]))
        before = now
    out["decode_writes_only_slot_pos_mod_window"] = bool(steps_ok)
    out["after_decode_bits_equal_last_step"] = torch.equal(bits(cache[f"after_decode.{L}window_kv_cache"][0]), bits(rings[-1]))
    # the indices: prefill lists positions, decode lists slots oldest first
    ids = pre[L + "sa_idxs"][0, P - 1, : min(P, WIN)]
    out["prefill_last_query_window"] = ids.tolist() == list(range(max(0, P - WIN), P))
    pos = P + D - 1
    want = [(pos + 1 + k) % WIN for k in range(WIN)]
    out["decode_last_query_slots_oldest_first"] = at_step(L + "sa_idxs", D - 1)[0, 0, :WIN].tolist() == want
    return out


def rope_freqs(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """Rotary frequencies with YaRN, as (cos, sin) per position and pair."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return torch.outer(torch.arange(seqlen), freqs)


def rotate(x, angles):
    """Rotate adjacent pairs of BF16 `x` [n, dim] by `angles` [n, dim/2]; the result is BF16."""
    pairs = x.float().unflatten(-1, (-1, 2))
    re, im = pairs[..., 0], pairs[..., 1]
    cos, sin = torch.cos(angles), torch.sin(angles)
    return torch.stack([re * cos - im * sin, re * sin + im * cos], dim=-1).flatten(-2).to(BF16)


def round_half_even(v, step):
    return torch.round(v / step) * step  # torch.round is half-to-even


def to_e4m3(v):
    """Positive float32 to the nearest E4M3 value (3 mantissa bits, least exponent -6)."""
    exponent = torch.floor(torch.log2(v)).clamp(min=-6.0)
    return round_half_even(v, torch.exp2(exponent - 3)).clamp(max=448.0)


def to_fp4(v):
    """float32 in [-6, 6] to the nearest E2M1 value (0, 0.5, 1, 1.5, 2, 3, 4, 6), ties to the even
    mantissa."""
    mag = v.abs()
    exponent = torch.floor(torch.log2(mag.clamp_min(1e-30))).clamp(min=0.0, max=2.0)  # 0.5 is subnormal
    return torch.copysign(round_half_even(mag, torch.exp2(exponent - 1)), v)


def fp4_groups_of_16(x):
    """BF16 `x` [n, d] quantized to FP4 in groups of 16 under an E4M3 scale, back as BF16."""
    groups = x.float().unflatten(-1, (-1, 16))
    amax = groups.abs().amax(-1, keepdim=True).clamp_min(6 * 2.0**-9)
    scale = to_e4m3(amax / 6.0)
    return (to_fp4((groups / scale).clamp(-6.0, 6.0)) * scale).flatten(-2).to(BF16)


def compressed_checks(i):
    L, ratio = f"L{i}.", cfg["compress_ratios"][i]
    latent = pre[L + "compressor_latent"][0].to(dev)  # [P // ratio, head_dim], before rotation
    angles = rope_freqs(ROPE, 4096, cfg["original_seq_len"], cfg["compress_rope_theta"], cfg["rope_factor"], cfg["beta_fast"], cfg["beta_slow"]).to(dev)
    held = cache[f"after_prefill.{L}compress_kv_cache"][0].to(dev)

    def expect(latents, first_positions):
        rotated = torch.cat([latents[:, :-ROPE], rotate(latents[:, -ROPE:], angles[first_positions.to(dev)])], dim=-1)
        return fp4_groups_of_16(rotated)

    mine = expect(latent, torch.arange(0, P - P % ratio, ratio))
    out = {
        "groups_after_prefill": held.size(0),
        "prefill_bf16_bits_equal": (bits(mine) == bits(held)).float().mean().item(),
        "prefill_rel_l2": rel(mine, held),
    }
    # decode: a latent appears each time a group completes, and takes the group's first position
    present = dec[L + "compressor_latent.present"]
    done = [P + step for step in range(D) if present[step]]
    out["decode_groups_complete_at"] = done[:4]
    out["decode_groups_complete_when_pos_plus_1_divides"] = all((pos + 1) % ratio == 0 for pos in done) and len(done) == sum((P + s + 1) % ratio == 0 for s in range(D))
    late = dec[L + "compressor_latent"].flatten(0, -2).to(dev)
    mine = expect(late, torch.tensor([pos + 1 - ratio for pos in done]))
    held = cache[f"after_decode.{L}compress_kv_cache"][0, P // ratio :].to(dev)
    out["decode_bf16_bits_equal"] = (bits(mine) == bits(held)).float().mean().item()
    out["decode_rel_l2"] = rel(mine, held)
    return out


result = {"prefill": P, "decode": D}
for i in range(LAYERS):
    entry = {"attention": attention_checks(i), "window": window_checks(i)}
    if f"L{i}.compressor_latent" in pre:
        entry["compressed_kv"] = compressed_checks(i)
    result[f"L{i}"] = entry
json.dump(result, open(os.path.join(a.rec, "check.json"), "w"), indent=1)
print(json.dumps(result, indent=1))
print("CHECKED")
