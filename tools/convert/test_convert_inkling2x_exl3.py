"""convert.py --exl3 over an Inkling 2 base pack: a tiny synthetic inkling2_loop GGUF made into the base, and a tiny EXL3
checkpoint whose experts are at two rate pairs; the v8 layout, the blobs, the tensors' forms and the refusals."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import convert  # noqa: E402
import exl3 as X  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import ModelFile  # noqa: E402
from test_checkpoint import write_safetensors  # noqa: E402

P = "inkling2_loop."
N_EXPERT, N_LAYERS, H, FF, VOCAB = 4, 2, 128, 128, 8
TOKENS = ["a", "b", "c", "d"]
TEMPLATE = "{{ messages }}"
TOKENIZER = {"tokenizer.json": b"{}", "vocab.json": json.dumps({t: i for i, t in enumerate(TOKENS)}).encode(),
             "merges.txt": b"", "token_type.json": b"[1,1,1,1]", "chat_template.jinja": TEMPLATE.encode()}
RATES = {0: (5, 6), 1: (6, 7)}   # expert parity -> gate/up and down k2 (2.5/3 and 3/3.5 bpw)


def bf16(x: np.ndarray) -> bytes:
    return (np.ascontiguousarray(x, dtype=np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()


def q8_0_values(raw: bytes, n: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    d = b[:, :2].copy().view("<f2").astype(np.float32)
    return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)[:n]


class Inkling2Exl3Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.rng = np.random.default_rng(11)
        self._orig = convert.export_tokenizer

        def tokenizer_files(first, out_dir):
            d = Path(out_dir) / "tokenizer"
            d.mkdir(parents=True, exist_ok=True)
            for k, v in TOKENIZER.items():
                (d / k).write_bytes(v)
            return sorted(d.iterdir())
        convert.export_tokenizer = tokenizer_files
        self.addCleanup(setattr, convert, "export_tokenizer", self._orig)
        self.addCleanup(self.tmp.cleanup)
        self.base = self.base_pack()

    def base_pack(self) -> Path:
        """The base: a GGUF-route pack with one tensor of each form the EXL3 path writes."""
        g = GGUFWriter()
        g.add("general.architecture", "inkling2_loop")
        kv = {"block_count": (N_LAYERS, "u32"), "embedding_length": (H, "u32"), "expert_count": (N_EXPERT, "u32"),
              "expert_used_count": (2, "u32"), "expert_feed_forward_length": (FF, "u32"),
              "attention.head_count": ([1] * N_LAYERS, "array:u32"), "attention.head_count_kv": ([1] * N_LAYERS, "array:u32"),
              "attention.key_length": (128, "u32"), "attention.sliding_window": (16, "u32"),
              "attention.sliding_window_pattern": ([True, False], "array:bool"), "attention.relative_length": (4, "u32"),
              "attention.relative_extent": (16, "u32"), "shortconv.l_cache": (4, "u32"), "attn_res.block_size": (1, "u32"),
              "loop.start": (1, "u32"), "loop.end": (2, "u32"), "loop.count": (2, "u32"), "loop.gate_rank": (8, "u32"),
              "vocab_size": (VOCAB, "u32"), "vocab_size_unpadded": (len(TOKENS), "u32")}
        for k, (v, ty) in kv.items():
            g.add(P + k, v, ty)
        g.add("tokenizer.ggml.tokens", TOKENS)
        g.add("tokenizer.chat_template", TEMPLATE)
        q8 = lambda shape: self.rng.bytes(int(np.prod(shape)) // 32 * 34)
        g.add_raw("token_embd.weight", [H, VOCAB], "Q8_0", q8([H, VOCAB]))
        g.add_f32("token_embd_norm.weight", np.ones(H, dtype=np.float32))
        g.add_raw("output.weight", [H, VOCAB], "Q8_0", q8([H, VOCAB]))
        for l in range(N_LAYERS):
            g.add_f32(f"blk.{l}.ffn_gate_inp.weight", np.zeros((H, N_EXPERT), dtype=np.float32))
            g.add_f32(f"blk.{l}.attn_post_conv.weight", np.zeros((4, H), dtype=np.float32))
            g.add_f32(f"blk.{l}.attn_rel_bias.weight", np.zeros((4, 16), dtype=np.float32))
            g.add_raw(f"blk.{l}.attn_q.weight", [H, 128], "Q8_0", q8([H, 128]))
            g.add_raw(f"blk.{l}.ffn_gate_up_exps.weight", [H, 2 * FF, N_EXPERT], "Q8_0", q8([H, 2 * FF, N_EXPERT]))
            g.add_raw(f"blk.{l}.ffn_down_exps.weight", [FF, H, N_EXPERT], "Q8_0", q8([FF, H, N_EXPERT]))
        g.add_f32("blk.1.attn_k_conv.weight", np.zeros((4, 128), dtype=np.float32))   # the full layer: static taps
        gguf = g.write(self.t / "Inkling-2-Mini-Q8_0.gguf")
        out = self.t / "base.basalt"
        a = type("A", (), dict(gguf=[str(gguf)], mmproj=None, out=str(out), mtp_source=None, mtp_dense=None, draft_vocab=None,
                               profile=None, profile_from=None, hc8=None, name="Base", compat_bf16=False, no_vision=False,
                               force=False, sections_from=None))()
        convert.convert(a)
        return out

    def checkpoint(self, gate_up_differ=False, bad_mul1=False, drop=None) -> Path:
        """The EXL3 checkpoint: each base tensor's counterpart under the official names."""
        d = self.t / "exl3"
        d.mkdir(exist_ok=True)
        (d / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "exl3"}, "text_config": {}}))
        ts, self.src = {}, {}

        def plain(name, x, dt="BF16"):
            x = np.asarray(x, dtype=np.float32)
            if dt == "BF16":
                x = ((x.view(np.uint32) >> 16) << 16).view(np.float32)   # exactly BF16
            self.src[name] = x
            ts[name] = (dt, x.shape, bf16(x) if dt == "BF16" else x.astype("<f4").tobytes())

        def exl3(stem, k, n, k2):
            words = 8 * k2
            tr = self.rng.integers(0, 1 << 16, size=(k // 16, n // 16, words), dtype=np.uint32).astype("<u2")
            suh = (self.rng.choice([-1.0, 1.0], k) * self.rng.uniform(0.5, 1.5, k)).astype("<f2")
            svh = (self.rng.choice([-1.0, 1.0], n) * self.rng.uniform(0.5, 1.5, n)).astype("<f2")
            self.src[stem] = (tr, suh, svh)
            ts[stem + ".trellis"] = ("I16", tr.shape, tr.tobytes())
            ts[stem + ".suh"] = ("F16", suh.shape, suh.tobytes())
            ts[stem + ".svh"] = ("F16", svh.shape, svh.tobytes())
            ts[stem + ".mul1"] = ("I32", [], struct.pack("<I", 0x12345678 if bad_mul1 else 0x83DCD12D))

        L = "model.llm."
        plain(L + "embed.weight", self.rng.standard_normal((VOCAB, H)))
        plain(L + "embed_norm.weight", self.rng.standard_normal(H))
        exl3(L + "unembed", H, 128, 8)   # 128 padded outputs, VOCAB kept
        for l in range(N_LAYERS):
            p = f"{L}layers.{l}."
            plain(p + "mlp.gate.weight", self.rng.standard_normal((N_EXPERT, H)))
            plain(p + "attn_sconv.weight", self.rng.standard_normal((H, 1, 4)))
            plain(p + "attn.rel_logits_proj.proj", self.rng.standard_normal((4, 16)))
            exl3(p + "attn.wq_du", H, 128, 10)
            for e in range(N_EXPERT):
                gu, dn = RATES[e % 2]
                for r, (k, n, k2) in (("gate_proj", (H, FF, gu)), ("up_proj", (H, FF, gu + (1 if gate_up_differ and e == 1 else 0))),
                                      ("down_proj", (FF, H, dn))):
                    exl3(f"{p}mlp.experts.{e}.{r}", k, n, k2)
        plain(L + "layers.1.attn.k_sconv.static_w", self.rng.standard_normal((4, 128)))
        if drop:
            ts = {k: v for k, v in ts.items() if not k.startswith(drop)}
        write_safetensors(d / "model-00001-of-00001.safetensors", ts)
        return d

    def convert(self, d) -> ModelFile:
        out = self.t / "out.basalt"
        a = type("A", (), dict(sections_from=str(self.base), exl3=str(d), out=str(out), dense_q8_0=False, name="Tiny EXL3",
                               workers=2, force=True))()
        convert.from_exl3(a)
        return ModelFile(str(out))

    def test_sections_layout_and_blobs(self):
        m = self.convert(self.checkpoint())
        self.assertEqual(set(m.names()), {"tokenizer/" + k for k in TOKENIZER} | {"gguf", "experts", "experts.layout", "meta"})
        for n in m.names():
            self.assertTrue(m.verify(n), n)
        lines = m.read("experts.layout").decode().splitlines()
        self.assertTrue(lines[0].startswith("# experts v8: layer expert gu_type d_type offset blob_bytes (n_expert 4"))
        rows = [[int(v) for v in x.split()] for x in lines[1:]]
        self.assertEqual(len(rows), N_LAYERS * N_EXPERT)
        experts = m.read("experts")
        for l, e, gt, dt, off, nb in rows:
            gu, dn = RATES[e % 2]
            self.assertEqual((gt, dt), (160 + gu, 160 + dn))
            want = b""
            vec = b""
            for r in ("gate_proj", "up_proj", "down_proj"):
                tr, suh, svh = self.src[f"model.llm.layers.{l}.mlp.experts.{e}.{r}"]
                want += np.ascontiguousarray(tr.transpose(1, 0, 2)).tobytes()   # tiles output-major
                vec += suh.tobytes() + svh.tobytes()
            self.assertEqual(experts[off:off + nb], want + vec)
        meta = json.loads(m.read("meta"))
        self.assertEqual(meta["arch"], "inkling2x")
        self.assertEqual(meta["gguf_kv"]["general.architecture"], "inkling2x")
        self.assertEqual(meta["exl3"]["experts"]["rates gate-up/down bpw -> experts"], {"2.5/3": 4, "3/3.5": 4})

    def test_tensor_forms(self):
        m = self.convert(self.checkpoint())
        meta = json.loads(m.read("meta"))
        g = m.read("gguf")
        T = meta["tensors"]
        def raw(n):
            return g[T[n]["offset"]:T[n]["offset"] + T[n]["bytes"]]
        self.assertEqual({n: T[n]["type"] for n in ("token_embd.weight", "output.weight", "blk.0.attn_q.weight", "blk.0.ffn_gate_inp.weight")},
                         {"token_embd.weight": "BF16", "output.weight": "Q8_0", "blk.0.attn_q.weight": "Q8_0", "blk.0.ffn_gate_inp.weight": "F32"})
        self.assertEqual(raw("token_embd.weight"), bf16(self.src["model.llm.embed.weight"]))
        f32 = lambda n: np.frombuffer(raw(n), dtype="<f4")
        np.testing.assert_array_equal(f32("blk.0.ffn_gate_inp.weight"), self.src["model.llm.layers.0.mlp.gate.weight"].reshape(-1))
        np.testing.assert_array_equal(f32("blk.0.attn_post_conv.weight"), self.src["model.llm.layers.0.attn_sconv.weight"].reshape(-1))
        np.testing.assert_array_equal(f32("blk.0.attn_rel_bias.weight"), self.src["model.llm.layers.0.attn.rel_logits_proj.proj"].T.reshape(-1))
        np.testing.assert_array_equal(f32("blk.1.attn_k_conv.weight"), self.src["model.llm.layers.1.attn.k_sconv.static_w"][::-1].T.reshape(-1))
        # an EXL3 projection: W^T rows (outputs), within one Q8_0 rounding; the head keeps the vocabulary's rows only
        for name, stem, n_out in (("blk.1.attn_q.weight", "model.llm.layers.1.attn.wq_du", 128), ("output.weight", "model.llm.unembed", VOCAB)):
            tr, suh, svh = self.src[stem]
            w = X.weight(tr, suh, svh).T[:n_out].reshape(-1)
            got = q8_0_values(raw(name), w.size)
            self.assertLess(np.abs(got - w).max(), np.abs(w).max() / 127 * 0.51 + 1e-6, name)

    def test_refusals(self):
        for kw, what in ((dict(gate_up_differ=True), "gate is"), (dict(bad_mul1=True), "only mul1"),
                         (dict(drop="model.llm.unembed.mul1"), "has no model.llm.unembed.mul1"),
                         (dict(drop="model.llm.layers.0.attn.wq_du"), "has no model.llm.layers.0.attn.wq_du")):
            with self.subTest(what=what):
                (self.t / "exl3" / "model-00001-of-00001.safetensors").unlink(missing_ok=True)
                d = self.checkpoint(**kw)
                with self.assertRaisesRegex(SystemExit, what):
                    self.convert(d)


if __name__ == "__main__":
    unittest.main()
