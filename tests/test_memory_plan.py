# SPDX-License-Identifier: AGPL-3.0-only
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import memory_plan as mp  # noqa: E402

# The fields kv_shape reads, as published in DeepSeek-V4.1-Flash-1017's config.json.
CONFIG = {"text_config": {
    "head_dim": 512, "index_head_dim": 128, "sliding_window": 128,
    "num_hidden_layers": 40, "num_nextn_predict_layers": 3,
    "kv_source_layer_ids": [2, 8, 14, 20],
    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
}}


class Roles(unittest.TestCase):
    def test_names_from_every_published_format(self):
        cases = {
            # DeepSeek FP8 / MXFP4
            "layers.1.engram.embed.weight": "engram_table",
            "layers.14.engram.embed.scale": "engram_table",
            "layers.3.ffn.experts.17.w1.weight": "routed_experts",
            "mtp.2.ffn.experts.127.w2.scale": "routed_experts",
            "layers.3.attn.wq_b.weight": "tp_split",
            "layers.3.attn.wo_a.weight": "tp_split",
            "layers.3.attn.wo_b.scale": "tp_split",
            "layers.3.attn.attn_sink": "tp_split",
            "layers.8.attn.indexer.wq_b.weight": "tp_split",
            "layers.8.attn.indexer.weights_proj.weight": "tp_split",
            "embed.weight": "tp_split",
            "head.weight": "tp_split",
            "vision.blocks.0.mlp.w1.weight": "rank0_only",
            "aligner.w1.weight": "rank0_only",
            "image_newline": "rank0_only",
            "layers.3.attn.wq_a.weight": "replicated",
            "layers.3.attn.wkv.weight": "replicated",
            "layers.8.attn.indexer.wk.weight": "replicated",
            "layers.2.attn.compressor.wgate.weight": "replicated",
            "layers.3.ffn.shared_experts.w1.weight": "replicated",
            "layers.3.ffn.gate.weight": "replicated",
            "layers.3.hc_attn_fn": "replicated",
            "layers.1.engram.wkv.weight": "replicated",
            "mtp.0.main_proj.weight": "replicated",
            # NVIDIA NVFP4
            "layers.3.ffn.experts.17.w3.weight_scale": "routed_experts",
            # EXL3
            "layers.3.ffn.experts.17.w1.trellis": "routed_experts",
            "layers.3.ffn.experts.17.w2.svh": "routed_experts",
            "layers.3.attn.wo_a.slice.5.trellis": "tp_split",
            "head.trellis": "tp_split",
            "layers.1.engram.wkv.trellis": "replicated",
            "layers.1.engram.q_weight": "replicated",
        }
        for name, role in cases.items():
            with self.subTest(name=name):
                self.assertEqual(mp.role_of(name), role)

    def test_local_directory_headers(self):
        with tempfile.TemporaryDirectory() as d:
            header = {"__metadata__": {"format": "pt"},
                      "layers.3.ffn.experts.0.w1.weight": {"dtype": "I8", "shape": [4, 2], "data_offsets": [0, 8]},
                      "embed.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [8, 16]}}
            raw = json.dumps(header).encode()
            Path(d, "model-00001-of-00001.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(16))
            roles = mp.roles_from_headers(mp.local_headers(d))
        self.assertEqual((roles["routed_experts"], roles["tp_split"], roles["replicated"]), (8, 8, 0))


class Plan(unittest.TestCase):
    def test_kv_shape_matches_the_reference_layout(self):
        per_token, per_seq = mp.kv_shape(CONFIG)
        # (256 + 32 + 64 + 4) bytes per compressed position, at ratios 2, 2, 2 and 1
        self.assertEqual(per_token, 356 * 2.5)
        # 43 layers x 128 window slots x (512 FP8 + 16 E8M0) bytes
        self.assertEqual(per_seq, 43 * 128 * 528)

    def test_split_and_engram_placement(self):
        roles = {"engram_table": 40 * mp.GIB, "routed_experts": 80 * mp.GIB, "tp_split": 8 * mp.GIB,
                 "rank0_only": 1 * mp.GIB, "replicated": 4 * mp.GIB, "config": CONFIG}
        args = dict(spark_gib=100, util=1.0, reserve_gib=0, seqs=0)
        nvme = mp.plan(roles, world=4, engram="nvme", **args)
        self.assertAlmostEqual(nvme["rank0_gib"], 4 + 88 / 4 + 1)
        self.assertAlmostEqual(nvme["other_rank_gib"], 4 + 88 / 4)
        sharded = mp.plan(roles, world=4, engram="sharded", **args)
        self.assertAlmostEqual(sharded["rank0_gib"] - nvme["rank0_gib"], 10)
        replicated = mp.plan(roles, world=2, engram="replicated", **args)
        self.assertAlmostEqual(replicated["rank0_gib"], 4 + 44 + 40 + 1)
        self.assertEqual(replicated["kv_positions"], int((100 - 89) * mp.GIB // (356 * 2.5)))

    def test_a_plan_that_does_not_fit_says_so(self):
        roles = {"engram_table": 0, "routed_experts": 300 * mp.GIB, "tp_split": 0,
                 "rank0_only": 0, "replicated": 0, "config": CONFIG}
        result = mp.plan(roles, world=2, engram="nvme", spark_gib=119.7, util=0.88, reserve_gib=6, seqs=8)
        self.assertFalse(result["fits"])
        self.assertEqual(result["kv_positions"], 0)


if __name__ == "__main__":
    unittest.main()
