# SPDX-License-Identifier: AGPL-3.0-only
"""A small fixture for the engine's host-side index rules, from the reference's own lines run on
recorded index scores: the top-k (model.py Indexer.forward) and the candidate blocks
(model.py select_candidate_blocks), at a small k and block count so a few hundred scores exercise
them.

  python index_fixture.py --ref /ref --rec /rec --out index_fixture.json
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file

p = argparse.ArgumentParser()
p.add_argument("--ref", required=True)
p.add_argument("--rec", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
sys.path.insert(0, a.ref)
torch.set_default_dtype(torch.bfloat16)
torch.cuda.set_device(0)
import model as M  # noqa: E402

K, BLOCKS, BLOCK = 64, 16, 8
scores = load_file(os.path.join(a.rec, "A_index.safetensors"))["prefill.score"].cuda()
ratio = 2


def tie_at_cut(s):
    cut = s.float().topk(K).values.min()
    return int((s.float() >= cut).sum()) > K


rows, tied = [], 0
for query in range(scores.size(0)):
    visible = (query + 1) // ratio
    if visible <= K:
        continue
    s = scores[query, :visible]
    wanted = query in (199, 665, 1400)
    if not wanted and not (tied < 3 and tie_at_cut(s) and query % 97 == 0):
        continue
    tied += tie_at_cut(s)
    chosen = s.topk(K, dim=-1, sorted=False).indices.sort(dim=-1).values
    kept = M.select_candidate_blocks(s, visible, BLOCKS, BLOCK)
    masked = s.masked_fill(~kept, -torch.inf)
    second = masked.topk(K, dim=-1, sorted=False).indices.sort(dim=-1).values
    rows.append(
        {
            "query": query,
            "tie_at_the_cut": tie_at_cut(s),
            "scores": s.cpu().view(torch.int16).numpy().astype("<u2").tobytes().hex(),
            "chosen": chosen.tolist(),
            "candidates": "".join("1" if v else "0" for v in kept.tolist()),
            "chosen_among_candidates": second.tolist(),
        }
    )
json.dump({"k": K, "topk_blocks": BLOCKS, "block_size": BLOCK, "rows": rows}, open(a.out, "w"), indent=1)
print([(r["query"], r["tie_at_the_cut"], sum(c == "1" for c in r["candidates"])) for r in rows])
print("FIXTURE")
