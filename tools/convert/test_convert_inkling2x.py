"""A tiny synthetic inkling2_loop GGUF converted end to end: sections, meta, the experts from the fused gate_up, and
tools/convert/verify_pack.py on the result."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import convert  # noqa: E402
import model_sections as S  # noqa: E402
import verify_pack  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import SECTION_ALIGN, ModelFile  # noqa: E402

P = "inkling2_loop."
N_EXPERT, N_LAYERS, N_EMBD, N_FF = 4, 4, 64, 32
TOKENS = ["a", "b", "c", "d"]
BLOCK = {"IQ4_NL": (18, 32), "Q8_0": (34, 32)}   # bytes, values
TEMPLATE = "{{ messages }}"
# what the tokenizer export writes for TOKENS and TEMPLATE (verify_pack compares them with the GGUF)
TOKENIZER = {"tokenizer.json": b"{}", "vocab.json": json.dumps({t: i for i, t in enumerate(TOKENS)}).encode(),
             "merges.txt": b"", "token_type.json": b"[1,1,1,1]", "chat_template.jinja": TEMPLATE.encode()}


def nbytes(ty, shape):
    b, v = BLOCK[ty]
    return int(np.prod(shape)) // v * b


class Inkling2Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.rng = np.random.default_rng(5)
        self.kv = {
            "block_count": (N_LAYERS, "u32"), "embedding_length": (N_EMBD, "u32"), "expert_count": (N_EXPERT, "u32"),
            "expert_used_count": (2, "u32"), "expert_feed_forward_length": (N_FF, "u32"),
            "attention.head_count": ([2] * N_LAYERS, "array:u32"), "attention.head_count_kv": ([1] * N_LAYERS, "array:u32"),
            "attention.key_length": (32, "u32"), "attention.sliding_window": (16, "u32"),
            "attention.sliding_window_pattern": ([True, False] * (N_LAYERS // 2), "array:bool"),
            "attention.relative_length": (4, "u32"), "attention.relative_extent": (16, "u32"),
            "shortconv.l_cache": (4, "u32"), "attn_res.block_size": (2, "u32"),
            "loop.start": (2, "u32"), "loop.end": (4, "u32"), "loop.count": (3, "u32"), "loop.gate_rank": (8, "u32"),
            "vocab_size": (8, "u32"), "vocab_size_unpadded": (len(TOKENS), "u32"),
        }
        self.raw = {}
        self._orig = convert.export_tokenizer

        def tokenizer_files(first, out_dir):
            d = Path(out_dir) / "tokenizer"
            d.mkdir(parents=True, exist_ok=True)
            for k, v in TOKENIZER.items():
                (d / k).write_bytes(v)
            return sorted(d.iterdir())
        convert.export_tokenizer = tokenizer_files

    def tearDown(self):
        convert.export_tokenizer = self._orig
        self.tmp.cleanup()

    def write(self, drop=(), split_layer=None, kv=None, head=False):
        """The GGUF: fused experts in every layer (separate gate and up as well in `split_layer`), `drop` keys left out,
        with `head` an output.weight."""
        g = GGUFWriter()
        g.add("general.architecture", "inkling2_loop")
        g.add("general.quantized_by", "Someone")
        for k, (v, ty) in {**self.kv, **(kv or {})}.items():
            if k not in drop:
                g.add(P + k, v, ty)
        g.add("tokenizer.ggml.tokens", TOKENS)
        g.add("tokenizer.chat_template", TEMPLATE)

        def raw(name, shape, ty):
            data = self.rng.bytes(nbytes(ty, shape))
            self.raw[name] = data
            g.add_raw(name, shape, ty, data)

        raw("token_embd.weight", [N_EMBD, 8], "Q8_0")
        if head:   # real Q8_0 blocks: a drafter folds it
            self.raw["output.weight"] = S.q8_0(self.rng.standard_normal(N_EMBD * 8).astype(np.float32))
            g.add_raw("output.weight", [N_EMBD, 8], "Q8_0", self.raw["output.weight"])
        g.add_f32("token_embd_norm.weight", self.rng.standard_normal(N_EMBD))
        for l in range(N_LAYERS):
            g.add_f32(f"blk.{l}.ffn_gate_inp.weight", self.rng.standard_normal((N_EMBD, N_EXPERT)))
            g.add_f32(f"blk.{l}.attn_post_conv.weight", self.rng.standard_normal((4, N_EMBD)))
            raw(f"blk.{l}.attn_q.weight", [N_EMBD, 64], "Q8_0")
            raw(f"blk.{l}.ffn_gate_up_exps.weight", [N_EMBD, 2 * N_FF, N_EXPERT], "IQ4_NL")
            raw(f"blk.{l}.ffn_down_exps.weight", [N_FF, N_EMBD, N_EXPERT], "Q8_0")
            if l == split_layer:
                raw(f"blk.{l}.ffn_gate_exps.weight", [N_EMBD, N_FF, N_EXPERT], "IQ4_NL")
        self.gguf = g.write(self.t / "Inkling-2-Mini-IQ4_NL.gguf")

    def args(self, **kw):
        d = dict(gguf=[str(self.gguf)], mmproj=None, out=str(self.t / "out.basalt"), mtp_source=None, mtp_dense=None,
                 draft_vocab=None, profile=None, profile_from=None, hc8=None, name=None, compat_bf16=False,
                 no_vision=False, force=False, sections_from=None)
        d.update(kw)
        return type("A", (), d)()

    def convert(self, **kw):
        a = self.args(**kw)
        a.name = a.name or convert.A.for_architecture("inkling2_loop").default_name(a.gguf[0], {})
        convert.convert(a)
        return ModelFile(str(self.t / "out.basalt"))

    def test_sections_and_meta(self):
        self.write()
        m = self.convert()
        self.assertEqual(set(m.names()), {"tokenizer/" + k for k in TOKENIZER} |
                         {"dense", "dense.index", "experts", "experts.layout", "gguf", "meta"})
        for n in m.names():
            self.assertTrue(m.verify(n), n)
            self.assertEqual(m.section(n)[0] % SECTION_ALIGN, 0, n)
        meta = json.loads(m.read("meta"))
        self.assertEqual(meta["arch"], "inkling2x")
        self.assertEqual(meta["gguf_kv"]["general.architecture"], "inkling2x")
        self.assertEqual(meta["name"], "Inkling-2-Mini IQ4_NL")
        self.assertEqual((meta["license"]["name"], meta["license"]["quantized_by"]), ("Apache-2.0", "Someone"))
        self.assertIn("TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION", meta["license"]["text"])
        self.assertEqual((meta["geometry"]["loop.start"], meta["geometry"]["loop.end"], meta["geometry"]["loop.count"]), (2, 4, 3))
        self.assertIsNone(meta["vision"])
        self.assertIsNone(meta["mtp"])

    def test_experts_from_the_fused_gate_up(self):
        self.write()
        m = self.convert()
        gu, dn = nbytes("IQ4_NL", [N_EMBD, 2 * N_FF]), nbytes("Q8_0", [N_FF, N_EMBD])
        lines = m.read("experts.layout").decode().splitlines()
        self.assertEqual(lines[0], "# experts v5: layer gu_type d_type offset blob_bytes (n_expert %d, total %d)"
                         % (N_EXPERT, N_LAYERS * N_EXPERT * (gu + dn)))
        self.assertEqual(lines[1:], ["%d 20 8 %d %d" % (l, l * N_EXPERT * (gu + dn), gu + dn) for l in range(N_LAYERS)])
        want = b""
        for l in range(N_LAYERS):
            g, d = self.raw[f"blk.{l}.ffn_gate_up_exps.weight"], self.raw[f"blk.{l}.ffn_down_exps.weight"]
            want += b"".join(g[e * gu:(e + 1) * gu] + d[e * dn:(e + 1) * dn] for e in range(N_EXPERT))
        self.assertEqual(m.read("experts"), want)

    def test_projections_stay_in_gguf_and_floats_in_dense(self):
        self.write()
        m = self.convert()
        rows = {r.split()[0]: r.split() for r in m.read("dense.index").decode().splitlines() if not r.startswith("#")}
        pool_bytes = lambda n: int(rows[n][6])
        self.assertEqual(pool_bytes("blk.0.attn_q.weight"), 0)          # served from `gguf` as it is
        self.assertEqual(pool_bytes("blk.0.attn_post_conv.weight"), 4 * N_EMBD * 4)
        self.assertFalse(any("_exps" in n for n in rows))

    def test_verify_agrees_and_catches_a_flipped_down_byte(self):
        self.write()
        m = self.convert()
        pack = self.t / "out.basalt"
        self.assertTrue(verify_pack.verify(str(pack), str(self.gguf)))
        gu = nbytes("IQ4_NL", [N_EMBD, 2 * N_FF])
        off, _ = m.section("experts")
        with open(pack, "r+b") as f:          # a byte of expert 0's down, after its fused gate_up
            f.seek(off + gu + 3)
            b = f.read(1)
            f.seek(off + gu + 3)
            f.write(bytes([b[0] ^ 1]))
        self.assertFalse(verify_pack.verify(str(pack), str(self.gguf)))

    def test_fused_and_separate_experts_are_refused(self):
        self.write(split_layer=1)
        with self.assertRaises(SystemExit):
            self.convert()

    def test_missing_loop_is_refused(self):
        self.write(drop=("loop.count",))
        with self.assertRaises(SystemExit):
            self.convert()

    def test_loop_off_the_block_grid_is_refused(self):
        self.write(kv={"loop.start": (1, "u32")})
        with self.assertRaises(SystemExit):
            self.convert()

    def test_side_sections_are_refused(self):
        self.write()
        with self.assertRaises(SystemExit):
            self.convert(mtp_source="x.safetensors")

    def test_unpaired_gate_up_is_refused(self):
        rng = np.random.default_rng(0)
        g = GGUFWriter()
        g.add("general.architecture", "inkling2_loop")
        g.add_f32("blk.0.ffn_gate_inp.weight", rng.standard_normal((N_EMBD, N_EXPERT)))
        g.add_raw("blk.0.ffn_gate_up_exps.weight", [N_EMBD, N_FF, N_EXPERT], "IQ4_NL", rng.bytes(nbytes("IQ4_NL", [N_EMBD, N_FF, N_EXPERT])))
        g.add_raw("blk.0.ffn_down_exps.weight", [N_FF, N_EMBD, N_EXPERT], "Q8_0", rng.bytes(nbytes("Q8_0", [N_FF, N_EMBD, N_EXPERT])))
        p = g.write(self.t / "bad.gguf")
        with self.assertRaisesRegex(S.ConvertError, "does not pair"):
            S.Experts(S.SourceModel([p]))


if __name__ == "__main__":
    unittest.main()
