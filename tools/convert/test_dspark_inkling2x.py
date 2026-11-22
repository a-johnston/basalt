"""Inkling 2's DSpark drafter (a tiny synthetic mtp.safetensors and its config.json) added to a tiny inkling2x pack with
convert.py --dspark --into: the section, its record, every tensor against the checkpoint (the folded Markov head against
the pack's own head, in double), the refusals, and the roll back."""
import json
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import convert  # noqa: E402
import dspark_inkling2x as DI  # noqa: E402
import model_sections as S  # noqa: E402
import test_convert_inkling2x as base  # noqa: E402
from model_file import SECTION_ALIGN, TENSOR_ALIGN, ModelFile  # noqa: E402

E, LAYERS, HEADS, KV, HD, FF, RANK, VOCAB_PADDED = base.N_EMBD, 2, 2, 1, 128, 32, 256, 8
N_VOCAB = len(base.TOKENS)
CONFIG = {"hidden_size": E, "num_hidden_layers": LAYERS, "num_attention_heads": HEADS, "num_key_value_heads": KV,
          "head_dim": HD, "intermediate_size": FF, "rope_theta": 1e6, "rms_norm_eps": 1e-6, "local_layer_ids": [0],
          "sliding_window_size": 16, "block_size": 4, "markov_rank": RANK, "markov_head_type": "rnn_hidden",
          "target_layer_ids": [1, 5, 8]}   # the tiny pack's visits: 4 layers, 2 of them looped 3 times, 8 in all


def bf16(x):
    """float32 -> bf16 bits, round to nearest even."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


class DsparkInkling2Test(unittest.TestCase):
    def setUp(self):
        self.pack = base.Inkling2Test()
        self.pack.setUp()
        self.t = self.pack.t
        self.rng = np.random.default_rng(9)

    def tearDown(self):
        self.pack.tearDown()

    def checkpoint(self, config=None, drop=()):
        """mtp.safetensors (BF16) and the config.json beside it; returns the tensors' values (float64, as stored)."""
        shapes = {"norm.weight": [E], "mask_embed": [E], "fuse_weight": [LAYERS + 1, len(CONFIG["target_layer_ids"])],
                  "markov_head.dflare_norm.weight": [E], "markov_head.rnn_proj.weight": [3 * RANK, 2 * RANK + 2 * E],
                  "markov_head.rnn_proj.bias": [3 * RANK], "markov_head.markov_w1.weight": [VOCAB_PADDED, RANK],
                  "markov_head.hidden_out.weight": [E, RANK], "markov_head.hidden_out.bias": [E]}
        for i in range(LAYERS):
            p = f"layers.{i}."
            shapes.update({p + "dflare_norm.weight": [E], p + "input_layernorm.weight": [E], p + "post_attention_layernorm.weight": [E],
                           p + "self_attn.q_proj.weight": [HEADS * HD, E], p + "self_attn.k_proj.weight": [KV * HD, E],
                           p + "self_attn.v_proj.weight": [KV * HD, E], p + "self_attn.o_proj.weight": [E, HEADS * HD],
                           p + "self_attn.k_proj_ctx.weight": [KV * HD, E], p + "self_attn.v_proj_ctx.weight": [KV * HD, E],
                           p + "self_attn.q_norm.weight": [HD], p + "self_attn.k_norm.weight": [HD],
                           p + "self_attn.k_norm_ctx.weight": [HD], p + "mlp.gate_proj.weight": [FF, E],
                           p + "mlp.up_proj.weight": [FF, E], p + "mlp.down_proj.weight": [E, FF]})
        values, head, blob = {}, {}, bytearray()
        for name, shape in shapes.items():
            if name in drop:
                continue
            raw = bf16(self.rng.standard_normal(int(np.prod(shape))) * 0.1)
            values[name] = S._bf16_to_f32(raw).astype(np.float64).reshape(shape)
            head[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [len(blob), len(blob) + raw.nbytes]}
            blob += raw.tobytes()
        h = json.dumps(head).encode()
        mtp = self.t / "mtp.safetensors"
        mtp.write_bytes(struct.pack("<Q", len(h)) + h + bytes(blob))
        (self.t / "config.json").write_text(json.dumps({"mtp_config": config or CONFIG}))
        return values

    def into(self, **kw):
        d = dict(into=str(self.t / "out.basalt"), dspark=str(self.t / "mtp.safetensors"), undo_dir=str(self.t / "undo"), roll_back=False)
        d.update(kw)
        return type("A", (), d)()

    def target(self):
        self.pack.write(head=True)
        self.pack.convert()
        return self.t / "out.basalt"

    def test_the_drafter_becomes_its_section(self):
        pack = self.target()
        before = pack.read_bytes()
        old = ModelFile(str(pack))
        v = self.checkpoint()
        convert.add_dspark(self.into())
        m = ModelFile(str(pack))
        for name in old.names():
            self.assertTrue(m.verify(name), name)
            if name != "meta":
                self.assertEqual(m.read(name), old.read(name), name)
        self.assertTrue(m.verify("dspark/gguf"))
        self.assertEqual(m.section("dspark/gguf")[0] % SECTION_ALIGN, 0)
        meta = json.loads(m.read("meta"))
        rec = meta["dspark"]
        self.assertEqual((rec["section"], rec["architecture"], rec["config"]), ("dspark/gguf", "inkling2-dspark", CONFIG))
        self.assertIn("mtp.safetensors", [Path(x["path"]).name for x in meta["sources"]])
        sec = np.frombuffer(m.read("dspark/gguf"), dtype=np.uint8)
        self.assertEqual(rec["bytes"], len(sec))

        def tensor(name):
            t = rec["tensors"][name]
            self.assertEqual(t["offset"] % TENSOR_ALIGN, 0, name)
            raw = sec[t["offset"]:t["offset"] + t["bytes"]].copy()
            return t, raw

        for i in range(LAYERS):
            for gg, ck in DI.LAYER_Q8.items():
                t, raw = tensor(f"dspark.blk.{i}.{gg}")
                src = v[f"layers.{i}.{ck}"]
                self.assertEqual((t["type"], t["shape"]), ("Q8_0", [src.shape[1], src.shape[0]]))
                got = S.q8_0_values(raw).reshape(src.shape)
                # Q8_0: half a step of each block's scale (amax / 127), and the fp16 scale's rounding
                self.assertLess(np.abs(got - src).max(), np.abs(src).max() / 127 * 0.51 + 1e-3, gg)
            for gg, ck in DI.LAYER_F32.items():
                t, raw = tensor(f"dspark.blk.{i}.{gg}")
                self.assertEqual(t["type"], "F32")
                np.testing.assert_array_equal(raw.view("<f4"), v[f"layers.{i}.{ck}"].astype(np.float32).reshape(-1))
        for gg, ck in DI.TOP_F32.items():
            t, raw = tensor(gg)
            self.assertEqual((t["type"], t["shape"]), ("F32", list(reversed(v[ck].shape))))
            np.testing.assert_array_equal(raw.view("<f4"), v[ck].astype(np.float32).reshape(-1))
        t, raw = tensor("dspark.markov_token_embd")   # the vocabulary's rows of markov_w1, as stored
        self.assertEqual((t["type"], t["shape"]), ("BF16", [RANK, N_VOCAB]))
        np.testing.assert_array_equal(raw.view("<u2"), bf16(v["markov_head.markov_w1.weight"][:N_VOCAB].reshape(-1)))
        # the fold: the pack's own head (its first n_vocab rows) times hidden_out, in double, rounded once
        head = S.q8_0_values(np.frombuffer(self.pack.raw["output.weight"], dtype=np.uint8)).astype(np.float64).reshape(VOCAB_PADDED, E)[:N_VOCAB]
        t, raw = tensor("dspark.markov_fold")
        self.assertEqual((t["type"], t["shape"]), ("BF16", [RANK, N_VOCAB]))
        np.testing.assert_array_equal(raw.view("<u2"), bf16(head @ v["markov_head.hidden_out.weight"]).reshape(-1))
        t, raw = tensor("dspark.markov_fold_bias")
        np.testing.assert_allclose(raw.view("<f4"), head @ v["markov_head.hidden_out.bias"], rtol=1e-6, atol=1e-7)
        # the roll back puts the pack back
        convert.add_dspark(self.into(roll_back=True))
        self.assertEqual(pack.read_bytes(), before)

    def test_drafters_this_engine_does_not_run_are_refused(self):
        self.target()
        for change in ({"markov_rank": 128}, {"target_layer_ids": [1, 5, 7]}, {"head_dim": 64}, {"block_size": 32},
                       {"markov_head_type": "linear"}, {"q_bias": True},
                       {"sliding_window_size": 0}):
            with self.subTest(change=change), self.assertRaises(SystemExit):
                self.checkpoint(config={**CONFIG, **change})
                convert.add_dspark(self.into())
        self.assertFalse((self.t / "undo").exists())

    def test_a_missing_tensor_or_config_is_refused(self):
        self.target()
        self.checkpoint(drop=("markov_head.hidden_out.bias",))
        with self.assertRaises(SystemExit):
            convert.add_dspark(self.into())
        self.checkpoint()
        (self.t / "config.json").unlink()
        with self.assertRaises(SystemExit):
            convert.add_dspark(self.into())
        self.assertFalse((self.t / "undo").exists())


if __name__ == "__main__":
    unittest.main()
