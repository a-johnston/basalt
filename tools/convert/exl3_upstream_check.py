"""Checks tools/convert/exl3.py against ExLlamaV3 itself on a real checkpoint.  Run it with a Python that has torch
and an ExLlamaV3 checkout built in place (not Basalt's environment):

    python tools/convert/exl3_upstream_check.py --exllamav3 DIR --checkpoint DIR [--tensor NAME ...]

The first form checks the token-embedding rows and, when the directory holds ngram_embedding.safetensors, the n-gram
rows against `ngram_codec.dequant_rows`, then each matrix (by default one of every (kind, bitrate) pair) against
the extension: the inner matrix bit for bit against `reconstruct`, the full weight against
`LinearEXL3.get_weight_tensor` and a forward against the extension's GEMV.  With --tensor only the named matrices
are checked.
"""
import argparse
import json
import pathlib
import re
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exl3 as X  # noqa: E402

DEV = "cuda:0"
NGRAM_FILE = "ngram_embedding.safetensors"


def header(p: pathlib.Path):
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def tensors(d: pathlib.Path) -> dict:
    """name -> (path, offset, dtype, shape, bytes) over the model shards."""
    t = {}
    for p in sorted(d.glob("model-*.safetensors")):
        head, base = header(p)
        for k, v in head.items():
            if k != "__metadata__":
                a, b = v["data_offsets"]
                t[k] = (p, base + a, v["dtype"], v["shape"], b - a)
    return t


def load(t, name, dtype):
    p, off, _, shape, n = t[name]
    return np.fromfile(p, dtype=dtype, count=n // np.dtype(dtype).itemsize, offset=off).reshape(shape)


def linear(torch, LinearEXL3, t, stem):
    tr = load(t, stem + ".trellis", "<i2")
    return LinearEXL3(None, tr.shape[0] * 16, tr.shape[1] * 16,
                      suh=torch.from_numpy(load(t, stem + ".suh", "<f2").copy()).to(DEV),
                      svh=torch.from_numpy(load(t, stem + ".svh", "<f2").copy()).to(DEV),
                      trellis=torch.from_numpy(tr.copy()).to(DEV),
                      mul1=torch.tensor(0, dtype=torch.int, device=DEV), key=stem)


def check_embeddings(torch, N, t) -> int:
    """Rows from the start, middle and end of each EXL3 embedding table."""
    bad = 0
    for stem in (k[:-len(".trellis")] for k in t if k.endswith(".trellis") and k[:-len(".trellis")] + ".signs" in t):
        signs = load(t, stem + ".signs", "<f2")
        p, off, _, shape, _ = t[stem + ".trellis"]
        rows = np.memmap(p, dtype="<i2", mode="r", offset=off, shape=tuple(shape))
        pick = np.r_[0:64, shape[0] // 2:shape[0] // 2 + 64, shape[0] - 64:shape[0]]
        k = X.ring_bits(shape[1], signs.shape[0])
        packed = torch.from_numpy(np.ascontiguousarray(rows[pick])).reshape(len(pick) * signs.shape[0], -1)
        up = N.dequant_rows(packed, k, N.mul1_codebook("cpu"),
                            signs=torch.from_numpy(signs.astype(np.float32)).repeat(len(pick), 1))
        up = up.reshape(len(pick), -1).double().numpy()
        got = X.embedding_rows(np.ascontiguousarray(rows[pick]).view("<u2"), signs).astype(np.float64)
        err = float(np.abs(got - up).max() / np.abs(up).max())
        ok = err < 1e-5
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {stem}: {len(pick)} rows of {signs.shape[0]} rings at {k} bpw, max rel err "
              f"{err:.2e} against ngram_codec.dequant_rows")
    return bad


def check_ngram(torch, N, path: pathlib.Path) -> int:
    """The first and last 32 rows of every head and the table's last 32 against ngram_codec.dequant_rows, exactly."""
    head, base = header(path)
    get = lambda suffix, dt: next(np.fromfile(path, dtype=dt, count=int(np.prod(v["shape"])),
                                              offset=base + v["data_offsets"][0]).reshape(v["shape"])
                                  for k, v in head.items() if k.endswith(suffix))
    bias, offsets = get(".head_bias", "<f2"), get(".head_offsets", "<i8")
    shards = sorted((int(re.search(r"shard_(\d+)\.trellis$", k).group(1)), v)
                    for k, v in head.items() if k.endswith(".trellis"))
    words = shards[0][1]["shape"][1]
    total = sum(v["shape"][0] for _, v in shards)
    table = np.memmap(path, dtype="<i2", mode="r", offset=base + shards[0][1]["data_offsets"][0], shape=(total, words))
    pick = np.unique(np.concatenate([np.arange(o, o + 32) for o in offsets] + [np.arange(o - 32, o) for o in offsets[1:]]
                                    + [np.arange(total - 32, total)]))
    heads = np.clip(np.searchsorted(offsets, pick, side="right") - 1, 0, len(offsets) - 1)
    packed = np.ascontiguousarray(table[pick])
    k = X.ngram_bits(words)
    up = N.dequant_rows(torch.from_numpy(packed), k, N.mul1_codebook("cpu"), torch.from_numpy(bias[heads].astype(np.float32)))
    got = X.ngram_rows(packed.view("<u2"), bias[heads])
    err = float(np.abs(got.astype(np.float64) - up.double().numpy()).max())
    print(f"{'ok  ' if err == 0 else 'FAIL'} n-gram table: {len(pick)} rows over the {len(offsets)} heads at {k} bpw, "
          f"max abs diff {err:.2e} against ngram_codec.dequant_rows")
    return int(err != 0)


def check_matrices(torch, ext, LinearEXL3, t, names) -> int:
    if not names:   # one tensor of every (kind, bitrate) pair
        seen = {}
        for k in t:
            stem = k[:-len(".trellis")]
            if k.endswith(".trellis") and stem + ".suh" in t:
                seen.setdefault((re.sub(r"\d+", "N", stem), t[k][3][-1]), stem)
        names = list(seen.values())
    bad = 0
    for stem in names:
        if stem + ".mul1" not in t:
            print(f"{stem}: not mul1, skipped")
            continue
        tr = load(t, stem + ".trellis", "<i2")
        suh, svh = load(t, stem + ".suh", "<f2"), load(t, stem + ".svh", "<f2")
        k2 = X.k2_of_words(tr.shape[-1])
        K = k2 // 2 if k2 % 2 == 0 else k2 / 2
        k, n = tr.shape[0] * 16, tr.shape[1] * 16
        lin = linear(torch, LinearEXL3, t, stem)
        inner_up = torch.empty((k, n), dtype=torch.half, device=DEV)
        ext.reconstruct(inner_up, torch.from_numpy(tr.copy()).to(DEV), K, False, True)
        same = np.array_equal(X.reconstruct(tr).view(np.uint16), inner_up.cpu().numpy().view(np.uint16))
        w_up = lin.get_weight_tensor().double().cpu().numpy()
        w = X.weight(tr, suh, svh)
        werr = float(np.abs(w - w_up).max() / np.abs(w).max())
        x = np.random.default_rng(1).standard_normal((4, k)).astype(np.float16)
        y_up = lin.forward(torch.from_numpy(x).to(DEV), {}, torch.float).double().cpu().numpy()
        y = x.astype(np.float64) @ w
        yerr = float(np.abs(y - y_up).max() / np.abs(y).max())
        ok = same and werr < 2e-3 and yerr < 2e-2
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {stem}: K {K} [{k} x {n}] inner bit-identical {same}, "
              f"weight max rel err {werr:.2e} (upstream rounds to fp16), forward max rel err {yerr:.2e}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exllamav3", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tensor", action="append", default=[])
    a = ap.parse_args()
    sys.path.insert(0, a.exllamav3)
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.quant.exl3 import LinearEXL3
    from exllamav3.modules.quant.exl3_lib import ngram_codec as N

    d = pathlib.Path(a.checkpoint)
    t = tensors(d)
    ngram = d / NGRAM_FILE
    bad = 0
    if not a.tensor:
        bad += check_embeddings(torch, N, t)
        if ngram.is_file():
            bad += check_ngram(torch, N, ngram)
    bad += check_matrices(torch, ext, LinearEXL3, t, a.tensor)
    print("FAILED" if bad else "all checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
