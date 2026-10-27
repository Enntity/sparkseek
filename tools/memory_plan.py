#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Per-Spark memory plan for a DeepSeek-V4.1-Flash checkpoint.

Reads only safetensors headers (from a local directory or, by HTTP range
requests, from a Hugging Face revision), sorts every tensor into a placement
role and prints what one rank holds at a given world size.

Placement follows DeepSeek's reference implementation (inference/model.py):
routed experts are split across ranks (expert parallel); the per-head attention
projections, the indexer query heads and the vocabulary embedding and head are
split (tensor parallel); the vision tower lives on rank 0; everything else is
replicated. The two Engram tables are a lookup, not a matmul weight, so where
they live is a choice: on each node's NVMe behind a row cache (`nvme`), split
by rows across ranks (`sharded`, the reference layout) or whole on every rank
(`replicated`).

  memory_plan.py roles  --hf REPO@REV [--out roles.json]
  memory_plan.py roles  --dir CHECKPOINT_DIR [--out roles.json]
  memory_plan.py plan   roles.json [--world 2|4] [--engram nvme|sharded|replicated]
"""
import argparse
import json
import re
import struct
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GIB = 1024**3

# Order matters: the first matching pattern wins.
ROLES = [
    ("engram_table", r"\.engram\.embed\."),
    ("routed_experts", r"^(layers|mtp)\.\d+\.ffn\.experts\."),
    ("rank0_only", r"^(vision|aligner)\.|^image_(start|end|newline)$"),
    ("tp_split", r"\.attn\.(wq_b|wo_a|wo_b|attn_sink|indexer\.wq_b|indexer\.weights_proj)\b"
                 r"|^(embed|head)\.(weight|trellis|suh|svh|mul1)$"),
    ("replicated", r""),
]


def role_of(name):
    return next(role for role, pattern in ROLES if re.search(pattern, name))


def roles_from_headers(headers):
    """{tensor name: header entry} -> {role: bytes}."""
    out = {role: 0 for role, _ in ROLES}
    for name, entry in headers.items():
        start, end = entry["data_offsets"]
        out[role_of(name)] += end - start
    return out


def read_header(open_range):
    (n,) = struct.unpack("<Q", open_range(0, 8))
    header = json.loads(open_range(8, 8 + n))
    header.pop("__metadata__", None)
    return header


def local_headers(directory):
    headers = {}
    for path in sorted(Path(directory).glob("*.safetensors")):
        with open(path, "rb") as f:
            def rng(a, b, f=f):
                f.seek(a)
                return f.read(b - a)
            headers.update(read_header(rng))
    return headers


def hf_get(repo, rev, path, start=None, end=None):
    req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/{rev}/{path}")
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end - 1}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def hf_headers(repo, rev):
    index = json.loads(hf_get(repo, rev, "model.safetensors.index.json"))
    files = sorted(set(index["weight_map"].values()))
    headers = {}
    with ThreadPoolExecutor(16) as pool:
        for h in pool.map(lambda f: read_header(lambda a, b: hf_get(repo, rev, f, a, b)), files):
            headers.update(h)
    return headers, json.loads(hf_get(repo, rev, "config.json"))


def kv_shape(config):
    """Bytes per compressed-KV token and per sequence window, from the model config.

    Compressed KV is FP4 with an E4M3 scale per 16 values; index keys are FP4 with
    an E8M0 scale per 32; window KV is FP8 with an E8M0 scale per 32 (model.py).
    Only kv_source_layers write compressed KV and index keys; the layers after
    them read the same cache.
    """
    t = config.get("text_config", config)
    head, idx, ratios = t["head_dim"], t["index_head_dim"], t["compress_ratios"]
    per_pos = head // 2 + head // 16 + idx // 2 + idx // 32
    per_token = sum(per_pos / ratios[layer] for layer in t["kv_source_layer_ids"])
    layers = t["num_hidden_layers"] + t.get("num_nextn_predict_layers", 0)
    per_seq = layers * t["sliding_window"] * (head + head // 32)
    return per_token, per_seq


def plan(roles, world, engram, spark_gib, util, reserve_gib, seqs):
    split = roles["routed_experts"] + roles["tp_split"]
    engram_bytes = {"nvme": 0, "sharded": roles["engram_table"] / world,
                    "replicated": roles["engram_table"]}[engram]
    rank0 = roles["replicated"] + split / world + engram_bytes + roles["rank0_only"]
    other = rank0 - roles["rank0_only"]
    budget = spark_gib * GIB * util - reserve_gib * GIB
    per_token, per_seq = kv_shape(roles["config"])
    kv_room = budget - rank0 - seqs * per_seq
    return {
        "world": world, "engram": engram,
        "rank0_gib": rank0 / GIB, "other_rank_gib": other / GIB, "budget_gib": budget / GIB,
        "kv_bytes_per_token": per_token, "window_mib_per_seq": per_seq / 2**20,
        "kv_positions": max(0, int(kv_room // per_token)), "fits": kv_room > 0,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("roles", help="sum tensor bytes per placement role")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", metavar="REPO@REV")
    src.add_argument("--dir")
    r.add_argument("--out")
    q = sub.add_parser("plan", help="per-rank residency at a world size")
    q.add_argument("roles")
    q.add_argument("--world", type=int, choices=(1, 2, 4), default=4)
    q.add_argument("--engram", choices=("nvme", "sharded", "replicated"), default="nvme")
    q.add_argument("--spark-gib", type=float, default=119.7, help="CUDA-visible memory per GB10")
    q.add_argument("--util", type=float, default=0.88, help="share of it the engine may use")
    q.add_argument("--reserve-gib", type=float, default=6.0, help="activations, workspace, graphs")
    q.add_argument("--seqs", type=int, default=8, help="concurrent sequences (window KV each)")
    a = p.parse_args(argv)

    if a.cmd == "roles":
        if a.hf:
            repo, _, rev = a.hf.partition("@")
            if not re.fullmatch(r"[0-9a-f]{40}", rev):
                p.error("--hf needs REPO@<40-hex revision>")
            headers, config = hf_headers(repo, rev)
            source = {"repo": repo, "revision": rev}
        else:
            headers = local_headers(a.dir)
            config = json.loads((Path(a.dir) / "config.json").read_text())
            source = {"dir": str(a.dir)}
        doc = {"source": source, "tensors": len(headers), **roles_from_headers(headers), "config": config}
        text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
        Path(a.out).write_text(text) if a.out else sys.stdout.write(text)
        return 0

    roles = json.loads(Path(a.roles).read_text())
    result = plan(roles, a.world, a.engram, a.spark_gib, a.util, a.reserve_gib, a.seqs)
    json.dump(result, sys.stdout, indent=1)
    print()
    return 0 if result["fits"] else 1


if __name__ == "__main__":
    sys.exit(main())
