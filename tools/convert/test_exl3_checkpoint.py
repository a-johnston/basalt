"""Tests for exl3_checkpoint.py on a small synthetic EXL3 checkpoint.

    python3 -m unittest discover -s tools/convert -p 'test_exl3_checkpoint.py'"""
import json
import multiprocessing
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exl3 as X  # noqa: E402
import exl3_checkpoint as E  # noqa: E402
import model_sections as S  # noqa: E402
from test_checkpoint import write_safetensors  # noqa: E402

H, FF, NL, NE = 256, 128, 2, 3


def make(d: Path, k2=(4, 3), drop=None, extra=None, mismatch=None):
    """A checkpoint of NL layers of NE experts, layer l at k2[l] half-bits; returns name -> array."""
    rng = np.random.default_rng(7)
    cfg = {"text_config": {"num_hidden_layers": NL, "num_experts": NE, "hidden_size": H, "moe_intermediate_size": FF},
           "quantization_config": {"quant_method": "exl3", "bits": 2.0, "codebook": "mul1"}}
    (d / "config.json").write_text(json.dumps(cfg))
    t, arrays = {}, {}
    for l in range(NL):
        for e in range(NE):
            for r, (k, n) in (("gate_proj", (H, FF)), ("up_proj", (H, FF)), ("down_proj", (FF, H))):
                p = f"model.language_model.layers.{l}.mlp.experts.{e}.{r}."
                words = 8 * (mismatch[1] if mismatch and mismatch[0] == (l, e, r) else k2[l])
                tr = rng.integers(0, 65536, size=(k // 16, n // 16, words), dtype=np.uint16)
                suh = (rng.standard_normal(k) * 0.02).astype("<f2")
                svh = rng.standard_normal(n).astype("<f2")
                arrays[p + "trellis"], arrays[p + "suh"], arrays[p + "svh"] = tr, suh, svh
                t[p + "trellis"] = ("I16", tr.shape, tr.tobytes())
                t[p + "suh"] = ("F16", suh.shape, suh.tobytes())
                t[p + "svh"] = ("F16", svh.shape, svh.tobytes())
                t[p + "mul1"] = ("I32", (), np.array(X.MUL1_MULT, dtype="<u4").tobytes())
    t["lm_head.suh"] = ("F16", (4,), np.zeros(4, "<f2").tobytes())   # non-expert tensors are ignored here
    for k in drop or ():
        del t[k]
    t.update(extra or {})
    names = list(t)
    half = len(names) // 2
    write_safetensors(d / "model-00001-of-00002.safetensors", {k: t[k] for k in names[:half]})
    write_safetensors(d / "model-00002-of-00002.safetensors", {k: t[k] for k in names[half:]})
    return arrays


NG_KV = {"general.architecture": "m", "m.ple.head_offsets": [0, 5], "m.ple.head_vocab_sizes": [5, 7],
         "m.ple.layer_multipliers": [3, 5, 7]}


def make_ngram(path: Path, k=3, shard_rows=(5, 5, 3), gap=False, offsets=(0, 5), fmt="exl3_ngram_trellis"):
    """An n-gram table file of the given shards; returns (rows uint16 [n, 1 + 10 k], bias fp16 [2, 160])."""
    rng = np.random.default_rng(11)
    stem = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
    bias = (rng.standard_normal((2, X.NGRAM_DIM)) * 0.01).astype("<f2")
    rows = rng.integers(0, 65536, size=(sum(shard_rows), 1 + 10 * k), dtype=np.uint16)
    t = {stem + "head_bias": ("F16", bias.shape, bias.tobytes()),
         stem + "head_offsets": ("I64", (2,), np.array(offsets, "<i8").tobytes()),
         stem + "head_vocab_sizes": ("I64", (2,), np.array([5, 7], "<i8").tobytes()),
         stem + "layer_multipliers": ("I64", (3,), np.array([3, 5, 7], "<i8").tobytes())}
    at = 0
    for i, n in enumerate(shard_rows):
        if gap and i == 1:
            t[stem + "pad"] = ("I16", (1,), b"\0\0")
        t[stem + f"shard_{i}.trellis"] = ("I16", (n, rows.shape[1]), rows[at:at + n].tobytes())
        at += n
    write_safetensors(path, t)
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    head = json.loads(raw[8:8 + n])
    head["__metadata__"] = {"format": fmt, "codebook": "mul1", "row_dim": "160", "K": str(k)}
    hb = json.dumps(head).encode()
    path.write_bytes(len(hb).to_bytes(8, "little") + hb + raw[8 + n:])
    return rows, bias


class NgramTableTest(unittest.TestCase):
    def test_the_table_is_the_files_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / E.NGRAM_FILE
            rows, bias = make_ngram(p, k=3)
            ng = E.NgramTable(p, NG_KV, 13)
            self.assertEqual((ng.k, ng.rows, ng.row_bytes, ng.type_id, ng.type_name), (3, 13, 62, 183, "EXL3R3"))
            src, off, size = ng.part()
            self.assertEqual(Path(src).read_bytes()[off:off + size], rows.tobytes())
            self.assertEqual(ng.bias, bias.tobytes())
            self.assertEqual(ng.bias_shape, [160, 2])

    def test_other_tables_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / E.NGRAM_FILE
            for kw, rows in (({"gap": True}, 13), ({"offsets": (0, 6)}, 13), ({"fmt": "other"}, 13), ({}, 14)):
                make_ngram(p, **kw)
                with self.assertRaises(S.ConvertError, msg=str(kw)):
                    E.NgramTable(p, NG_KV, rows)


class DenseJobTest(unittest.TestCase):
    def test_rows_are_the_weight_as_q8_0_or_q6_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(5)
            k_in, n_out = 256, 256
            tr = rng.integers(0, 65536, size=(k_in // 16, n_out // 16, 32), dtype=np.uint16)
            suh = (rng.standard_normal(k_in) * 0.02).astype("<f2")
            svh = rng.standard_normal(n_out).astype("<f2")
            p = Path(tmp) / "m.safetensors"
            write_safetensors(p, {"trellis": ("I16", tr.shape, tr.tobytes()), "suh": ("F16", suh.shape, suh.tobytes()),
                                  "svh": ("F16", svh.shape, svh.tobytes())})
            head, base = S.read_safetensors_header(p)
            part = lambda k: (p, base + head[k]["data_offsets"][0], head[k]["dtype"], head[k]["shape"], 0)
            parts = (part("trellis"), part("suh"), part("svh"))
            want = X.weight(tr, suh, svh).T[128:256].astype(np.float32)
            for ty, tol in (("Q8_0", 1 / 127), ("Q6_K", 1 / 31)):
                raw = np.frombuffer(E._dense_job(("rows", parts, 128, 128, None, ty)), dtype=np.uint8)
                got = S._dequantize(raw, ty).reshape(128, k_in)
                self.assertLess(np.abs(got - want).max(), tol * np.abs(want).max(), ty)
            self.assertEqual(len(E._dense_job(("rows", parts, 128, 128, None, "Q6_K"))), 128 * k_in // 256 * 210)


class Exl3CheckpointTest(unittest.TestCase):
    def test_blobs_hold_the_checkpoint_weights(self):
        with tempfile.TemporaryDirectory() as tmp, multiprocessing.Pool(2) as pool:
            d = Path(tmp)
            arrays = make(d)
            ex = E.Exl3Experts(E.Exl3Checkpoint(d))
            data = b"".join(ex.chunks(pool, group=2))
            self.assertEqual(len(data), ex.total)
            text = ex.layout_text().decode()
            self.assertTrue(text.startswith("# experts v5: "))
            self.assertIn("(n_expert 3, total %d)" % ex.total, text)
            rows = [[int(v) for v in ln.split()] for ln in text.splitlines()[1:]]
            self.assertEqual([r[1:3] for r in rows], [[164, 164], [163, 163]])
            for l, gt, dt, off, blob in rows:
                self.assertEqual(blob, 3 * H * FF * (gt - 160) // 16 + 6 * (H + FF))
                for e in range(NE):
                    b = np.frombuffer(data, np.uint8, blob, off + e * blob)
                    at = 0
                    vec = b[blob - 6 * (H + FF):].view("<f2")
                    vat = 0
                    for r, (k, n) in (("gate_proj", (H, FF)), ("up_proj", (H, FF)), ("down_proj", (FF, H))):
                        p = f"model.language_model.layers.{l}.mlp.experts.{e}.{r}."
                        nb = k * n * (gt - 160) // 16
                        tr = b[at:at + nb].view("<u2").reshape(n // 16, k // 16, -1).transpose(1, 0, 2)
                        at += nb
                        self.assertTrue(np.array_equal(tr, arrays[p + "trellis"]), p)
                        suh, svh = vec[vat:vat + k], vec[vat + k:vat + k + n]
                        vat += k + n
                        self.assertTrue(np.array_equal(X.weight(tr, suh, svh),
                                                       X.weight(arrays[p + "trellis"], arrays[p + "suh"], arrays[p + "svh"])))
            self.assertEqual(ex.bitrates(), {"2/2": [0], "1.5/1.5": [1]})

    def refused(self, why, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            make(d, **kw)
            with self.assertRaises(S.ConvertError) as cm:
                E.Exl3Experts(E.Exl3Checkpoint(d))
            self.assertIn(why, str(cm.exception))

    def test_refusals(self):
        p = "model.language_model.layers.1.mlp.experts.2."
        self.refused("only the mul1 codebook", drop=[p + "up_proj.mul1"])
        self.refused("has no", drop=[p + "down_proj.svh"])
        self.refused("unexpected expert tensor", extra={p + "gate_proj.mcg": ("I32", (), b"\0" * 4)})
        self.refused("unexpected expert tensor", extra={p + "gate_proj.su": ("I16", (16,), b"\0" * 32)})
        self.refused("unexpected expert tensor",
                     extra={"model.language_model.layers.2.mlp.experts.0.gate_proj.suh": ("F16", (H,), b"\0" * 2 * H)})
        self.refused("gate is 2.0 bpw and up 1.5", mismatch=((0, 1, "up_proj"), 3))
        self.refused("one bitrate per layer", mismatch=((0, 1, "down_proj"), 6))
        self.refused("not a known EXL3 bitrate", mismatch=((0, 0, "gate_proj"), 9))

    def test_a_missing_shard_or_other_format_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            make(d)
            (d / "model-00002-of-00002.safetensors").unlink()
            with self.assertRaises(S.ConvertError):
                E.Exl3Checkpoint(d)
            make(d)
            cfg = json.loads((d / "config.json").read_text())
            cfg["quantization_config"]["quant_method"] = "awq"
            (d / "config.json").write_text(json.dumps(cfg))
            with self.assertRaises(S.ConvertError):
                E.Exl3Checkpoint(d)


if __name__ == "__main__":
    unittest.main()
