"""The language model of a .basalt pack read from an EXL3 checkpoint (ExLlamaV3's safetensors, as in
turboderp/Qwen3.8-Flash-Next-exl3), for tools/convert/convert.py --exl3.  The format is in exl3.py.

  experts          per expert [gate trellis | up trellis | down trellis | suh_g svh_g suh_u svh_u suh_d svh_d]:
                   the trellis words as stored, tiles reordered output-major (tile (kt, nt) of the checkpoint at
                   nt * k/16 + kt), so 16 output rows are one contiguous run; the six fp16 vectors unchanged
  experts.layout   v5: `layer gu_type d_type offset blob_bytes`, the types 160 + 2 K (K the bits per weight)

  dense, gguf      every other language-model tensor under checkpoint.CheckpointDense's names, order and forms:
                   an EXL3 matrix's weight computed in double and stored once as Q6_K or Q8_0, an EXL3 embedding
                   table as Q8_0, the rest as BF16, F32 or F16

  gguf             the n-gram table, when the checkpoint directory holds ngram_embedding.safetensors: its rows
                   (an fp16 scale and a ring of 160 steps of K bits, 2 + 20 K bytes) byte for byte as
                   per_layer_token_embd.weight, type 180 + K, and the 16 heads' biases as
                   per_layer_token_embd.head_bias (F16)

Every expert of a layer must share one bitrate, gate and up one bitrate, and every matrix the mul1 codebook with
fp16 suh / svh; anything else is refused.  Needs numpy.
"""
from __future__ import annotations

import json
import pathlib
import re

import numpy as np

import checkpoint as C
import exl3 as X
import model_sections as S

TYPE_BASE = 160          # engine type id of an EXL3 mul1 matrix: TYPE_BASE + 2 K
ROLES = ("gate_proj", "up_proj", "down_proj")
PARTS = ("trellis", "suh", "svh", "mul1")
EXPERT = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)\.(\w+)")


class Exl3Checkpoint:
    """The model shards of an EXL3 checkpoint directory: name -> (path, absolute offset, dtype, shape, bytes)."""

    def __init__(self, d):
        self.dir = pathlib.Path(d)
        cfg = self.dir / "config.json"
        if not cfg.is_file():
            raise S.ConvertError(f"{cfg} not found")
        self.config = json.loads(cfg.read_text(encoding="utf-8"))
        self.text = self.config.get("text_config", self.config)
        q = self.config.get("quantization_config") or {}
        if q.get("quant_method") != "exl3":
            raise S.ConvertError(f"{cfg}: quantization_config.quant_method is {q.get('quant_method')!r}, expected exl3")
        self.quant = q
        self.shards = sorted(p for p in self.dir.glob("*.safetensors") if re.fullmatch(r"model-\d+-of-\d+\.safetensors", p.name))
        if not self.shards:
            raise S.ConvertError(f"{self.dir}: no model-NNNNN-of-NNNNN.safetensors shards")
        m = re.fullmatch(r"model-\d+-of-(\d+)\.safetensors", self.shards[0].name)
        if int(m.group(1)) != len(self.shards):
            raise S.ConvertError(f"{self.dir}: {len(self.shards)} of {int(m.group(1))} model shards")
        self.t = {}
        for p in self.shards:
            head, base = S.read_safetensors_header(p)
            for k, v in head.items():
                if k == "__metadata__":
                    continue
                if k in self.t:
                    raise S.ConvertError(f"{p.name}: duplicate tensor {k}")
                a, b = v["data_offsets"]
                self.t[k] = (p, base + a, v["dtype"], list(v["shape"]), b - a)


def _expert_job(args):
    """Blobs of some experts of one layer: per expert nine (path, offset, shape) reads, role-major, then part-major."""
    out = []
    for ex in args:
        tr, vec = [], []
        for (tp, to, tshape), (up, uo, un), (vp, vo, vn) in ex:
            t = np.fromfile(tp, dtype="<u2", count=int(np.prod(tshape)), offset=to).reshape(tshape)
            tr.append(np.ascontiguousarray(t.transpose(1, 0, 2)).tobytes())
            for p, o, n in ((up, uo, un), (vp, vo, vn)):
                v = np.fromfile(p, dtype="<f2", count=n, offset=o)
                if not np.isfinite(v).all():
                    return f"a scale vector with a non-finite value at {p.name}+{o}"
                vec.append(v.tobytes())
        out += tr + vec
    return b"".join(out)


NGRAM_FILE = "ngram_embedding.safetensors"
NGRAM_TYPE_BASE = 180    # engine type id of an n-gram row of K bits per value: NGRAM_TYPE_BASE + K
NGRAM_NAME = "per_layer_token_embd.weight"
NGRAM_BIAS_NAME = "per_layer_token_embd.head_bias"


class NgramTable:
    """The n-gram table of an EXL3 checkpoint (format exl3_ngram_trellis): shards of rows that follow one another
    in the file, copied as they are, and the per-head bias.  `kv` is the pack's GGUF metadata, whose hash
    constants the table's must equal."""

    def __init__(self, path, kv: dict, rows: int):
        self.path = pathlib.Path(path)
        head, base = S.read_safetensors_header(self.path)
        md = head.get("__metadata__") or {}
        if (md.get("format"), md.get("codebook"), md.get("row_dim")) != ("exl3_ngram_trellis", "mul1", str(X.NGRAM_DIM)):
            raise S.ConvertError(f"{self.path}: not an exl3_ngram_trellis table of mul1 rows of {X.NGRAM_DIM} values")

        def one(suffix, dtype, shape):
            found = [(k, v) for k, v in head.items() if k.endswith(suffix)]
            if len(found) != 1 or found[0][1]["dtype"] != dtype or list(found[0][1]["shape"]) != shape:
                raise S.ConvertError(f"{self.path}: expected one {dtype} {shape} tensor *{suffix}")
            a, b = found[0][1]["data_offsets"]
            return found[0][0], np.fromfile(self.path, dtype={"I64": "<i8", "F16": "<f2"}[dtype], count=(b - a) // (8 if dtype == "I64" else 2),
                                            offset=base + a)

        arch = kv.get("general.architecture", "")
        heads = len(kv.get(arch + ".ple.head_offsets", []))
        for suffix, key, n in ((".head_offsets", "head_offsets", heads), (".head_vocab_sizes", "head_vocab_sizes", heads),
                               (".layer_multipliers", "layer_multipliers", len(kv.get(arch + ".ple.layer_multipliers", [])))):
            _, v = one(suffix, "I64", [n])
            if [int(x) for x in v] != [int(x) for x in kv.get(f"{arch}.ple.{key}", [])]:
                raise S.ConvertError(f"{self.path}: {key} differs from the pack's {arch}.ple.{key}")
        name, bias = one(".head_bias", "F16", [heads, X.NGRAM_DIM])
        self.bias = bias.tobytes()
        self.bias_shape = [X.NGRAM_DIM, heads]   # GGUF order: the fast axis first
        stem = name[:-len(".head_bias")]
        shards = {}
        for k, v in head.items():
            m = re.fullmatch(re.escape(stem) + r"\.shard_(\d+)\.trellis", k)
            if m:
                shards[int(m.group(1))] = v
        if not shards or sorted(shards) != list(range(len(shards))):
            raise S.ConvertError(f"{self.path}: the shards of {stem} are not numbered 0 .. n - 1")
        words = shards[0]["shape"][1]
        self.k = X.ngram_bits(words)
        self.row_bytes = 2 * words
        self.rows = 0
        end = shards[0]["data_offsets"][0]
        for i in range(len(shards)):
            v = shards[i]
            a, b = v["data_offsets"]
            if v["dtype"] != "I16" or len(v["shape"]) != 2 or v["shape"][1] != words or a != end or b - a != v["shape"][0] * self.row_bytes:
                raise S.ConvertError(f"{self.path}: shard {i} is not int16 rows of {words} that follow shard {i - 1}")
            end = b
            self.rows += v["shape"][0]
        if self.rows != rows:
            raise S.ConvertError(f"{self.path}: {self.rows} rows, the pack's table has {rows}")
        self.offset = base + shards[0]["data_offsets"][0]
        self.nbytes = self.rows * self.row_bytes
        self.type_id = NGRAM_TYPE_BASE + self.k
        self.type_name = f"EXL3R{self.k}"

    def part(self) -> tuple:
        return (str(self.path), self.offset, self.nbytes)


class Exl3Experts:
    """The `experts` section of an EXL3 checkpoint and its `experts.layout`."""

    def __init__(self, ck: Exl3Checkpoint):
        tc = ck.text
        self.ck = ck
        self.n_layers, self.n_expert = int(tc["num_hidden_layers"]), int(tc["num_experts"])
        h, ff = int(tc["hidden_size"]), int(tc["moe_intermediate_size"])
        if h % 128 or ff % 128:
            raise S.ConvertError(f"expert widths {h} x {ff} are not whole 128-value Hadamard blocks")
        self.h, self.ff = h, ff
        dims = {"gate_proj": (h, ff), "up_proj": (h, ff), "down_proj": (ff, h)}   # inputs, outputs
        for k in ck.t:
            m = EXPERT.fullmatch(k)
            if k.startswith("model.language_model.layers.") and ".mlp.experts." in k and (
                    not m or int(m.group(1)) >= self.n_layers or int(m.group(2)) >= self.n_expert
                    or m.group(3) not in ROLES or m.group(4) not in PARTS):
                raise S.ConvertError(f"unexpected expert tensor {k} (the model has {self.n_layers} layers of "
                                     f"{self.n_expert} experts; only mul1 trellis, suh and svh are read)")
        self.entries, self.layout, self.k2 = [], [], []
        self.vec_bytes = 6 * (h + ff)
        offset = 0
        for l in range(self.n_layers):
            layer, k2s = [], None
            for e in range(self.n_expert):
                ex, got = [], []
                for r in ROLES:
                    p = f"model.language_model.layers.{l}.mlp.experts.{e}.{r}."
                    k, n = dims[r]
                    for part in PARTS:
                        if p + part not in ck.t:
                            raise S.ConvertError(f"the checkpoint has no {p}{part}" + (
                                " (only the mul1 codebook is read)" if part == "mul1" else ""))
                    tp, to, tdt, tshape, _ = ck.t[p + "trellis"]
                    if tdt != "I16" or len(tshape) != 3 or tshape[:2] != [k // 16, n // 16]:
                        raise S.ConvertError(f"{p}trellis is {tdt} {tshape}, expected I16 [{k // 16}, {n // 16}, 16 K]")
                    try:
                        got.append(X.k2_of_words(tshape[2]))
                    except X.Exl3Error as err:
                        raise S.ConvertError(f"{p}trellis: {err}") from None
                    vs = []
                    for part, n_v in (("suh", k), ("svh", n)):
                        vp, vo, vdt, vshape, _ = ck.t[p + part]
                        if vdt != "F16" or vshape != [n_v]:
                            raise S.ConvertError(f"{p}{part} is {vdt} {vshape}, expected F16 [{n_v}]")
                        vs.append((vp, vo, n_v))
                    ex.append(((tp, to, tshape), vs[0], vs[1]))
                if got[0] != got[1]:
                    raise S.ConvertError(f"layer {l} expert {e}: gate is {got[0] / 2} bpw and up {got[1] / 2}")
                if k2s is None:
                    k2s = (got[0], got[2])
                elif k2s != (got[0], got[2]):
                    raise S.ConvertError(f"layer {l}: expert {e} is {got[0] / 2} / {got[2] / 2} bpw (gate-up / down), "
                                         f"expert 0 {k2s[0] / 2} / {k2s[1] / 2}; one bitrate per layer is required")
                layer.append(ex)
            blob = 2 * (h * ff * k2s[0] // 16) + h * ff * k2s[1] // 16 + self.vec_bytes
            self.entries.append(layer)
            self.k2.append(k2s)
            self.layout.append((l, TYPE_BASE + k2s[0], TYPE_BASE + k2s[1], offset, blob))
            offset += blob * self.n_expert
        self.total = offset

    def layout_text(self) -> bytes:
        head = ("# experts v5: layer gu_type d_type offset blob_bytes (n_expert %d, total %d); EXL3 mul1 experts, "
                "type 160 + 2 K, a blob ends in fp16 suh_g svh_g suh_u svh_u suh_d svh_d\n" % (self.n_expert, self.total))
        return (head + "".join("%d %d %d %d %d\n" % x for x in self.layout)).encode("utf-8")

    def jobs(self, group: int):
        for l in range(self.n_layers):
            for s in range(0, self.n_expert, group):
                yield self.entries[l][s:s + group]

    def chunks(self, pool, group: int = 32, depth: int = 32):
        for b in S.bounded_imap(pool, _expert_job, self.jobs(group), depth):
            if isinstance(b, str):
                raise S.ConvertError(b)
            yield b

    def bitrates(self) -> dict:
        """bits per weight -> layers, for the pack's metadata."""
        out = {}
        for l, (g, d) in enumerate(self.k2):
            out.setdefault(f"{g / 2:g}/{d / 2:g}", []).append(l)
        return out


# ------------------------------------------------------------------ every other language-model tensor
class DenseView:
    """An EXL3 checkpoint as checkpoint.CheckpointDense reads one: a linear layer stored as trellis / suh / svh is
    the tensor `<stem>.weight` of type EXL3 and shape [outputs, inputs], a token embedding stored as trellis / signs
    one of type EXL3ROWS; `parts[name]` holds what its values are read from."""

    def __init__(self, ck: Exl3Checkpoint):
        self.dir, self.config, self.text, self.shards = ck.dir, ck.config, ck.text, ck.shards
        self.t, self.order, self.parts = {}, [], {}
        by_file = {}
        for k, v in ck.t.items():
            by_file.setdefault(v[0], []).append((v[1], k))
        for p in ck.shards:
            for _, k in sorted(by_file.get(p, [])):
                if C.SKIP.match(k):
                    continue
                stem, _, part = k.rpartition(".")
                if part in ("suh", "svh", "mul1", "signs"):
                    continue
                if part == "trellis":
                    tr = ck.t[k]
                    if tr[2] != "I16":
                        raise S.ConvertError(f"{k} is {tr[2]}")
                    name = stem + ".weight"
                    if stem + ".signs" in ck.t:
                        sg = ck.t[stem + ".signs"]
                        if sg[2] != "F16" or len(sg[3]) != 2 or sg[3][1] != X.RING or len(tr[3]) != 2:
                            raise S.ConvertError(f"{stem}: an embedding of trellis {tr[3]} and signs {sg[2]} {sg[3]}")
                        try:
                            X.ring_bits(tr[3][1], sg[3][0])
                        except X.Exl3Error as err:
                            raise S.ConvertError(f"{k}: {err}") from None
                        self.t[name] = (tr[0], tr[1], "EXL3ROWS", [tr[3][0], sg[3][0] * X.RING], tr[4])
                        self.parts[name] = (tr, sg)
                    else:
                        for q in ("suh", "svh", "mul1"):
                            if f"{stem}.{q}" not in ck.t:
                                raise S.ConvertError(f"the checkpoint has no {stem}.{q}" + (
                                    " (only the mul1 codebook is read)" if q == "mul1" else ""))
                        su, sv = ck.t[stem + ".suh"], ck.t[stem + ".svh"]
                        try:
                            X.k2_of_words(tr[3][2] if len(tr[3]) == 3 else 0)
                        except X.Exl3Error as err:
                            raise S.ConvertError(f"{k}: {err}") from None
                        k_in, n_out = tr[3][0] * 16, tr[3][1] * 16
                        if su[2] != "F16" or sv[2] != "F16" or su[3] != [k_in] or sv[3] != [n_out] or k_in % 128 or n_out % 128:
                            raise S.ConvertError(f"{stem}: trellis {tr[3]} with suh {su[2]} {su[3]} and svh {sv[2]} {sv[3]}")
                        self.t[name] = (tr[0], tr[1], "EXL3", [n_out, k_in], tr[4])
                        self.parts[name] = (tr, su, sv)
                    self.order.append(name)
                else:
                    self.t[k] = ck.t[k]
                    self.order.append(k)

    def values(self, name) -> np.ndarray:
        p, off, dt, shape, n = self.t[name]
        if dt == "EXL3":
            return _exl3_rows(self.parts[name], 0, shape[0])
        if dt == "EXL3ROWS":
            raise S.ConvertError(f"{name}: an embedding table is read in chunks")
        raw = np.fromfile(p, dtype=np.uint8, count=n, offset=off)
        if dt == "BF16":
            return S._bf16_to_f32(raw.view("<u2")).reshape(shape)
        if dt == "F16":
            return raw.view("<f2").astype(np.float32).reshape(shape)
        if dt == "F32":
            return raw.view("<f4").reshape(shape)
        raise S.ConvertError(f"{name} is {dt}")


def _exl3_rows(parts, out0: int, n_out: int) -> np.ndarray:
    """Outputs [out0, out0 + n_out) (whole 128-output blocks) of an EXL3 matrix as float32 [n_out, inputs]."""
    (tp, to, _, tshape, _), (up, uo, _, ushape, _), (vp, vo, _, _, _) = parts
    tr = np.memmap(tp, dtype="<u2", mode="r", offset=to, shape=tuple(tshape))[:, out0 // 16:(out0 + n_out) // 16]
    suh = np.fromfile(up, dtype="<f2", count=ushape[0], offset=uo)
    svh = np.fromfile(vp, dtype="<f2", count=n_out, offset=vo + 2 * out0)
    w = X.weight(np.ascontiguousarray(tr), suh, svh)
    return np.ascontiguousarray(w.T, dtype=np.float32)


def _dense_job(args):
    """Rows of one tensor as Q6_K or Q8_0: a row range of an EXL3 matrix or embedding table, or a whole EXL3 matrix
    with its rows or columns in the GGUF order."""
    kind, parts, row0, rows, index, ty = args
    if kind == "rows":
        x = _exl3_rows(parts, row0, rows)
    elif kind == "table":
        (tp, to, _, tshape, _), (sp, so, _, sshape, _) = parts
        tr = np.memmap(tp, dtype="<u2", mode="r", offset=to, shape=tuple(tshape))[row0:row0 + rows]
        x = X.embedding_rows(tr, np.fromfile(sp, dtype="<f2", count=sshape[0] * sshape[1], offset=so).reshape(sshape))
    else:
        x = _exl3_rows(parts, 0, rows)
        x = x[index] if kind == "all_rows" else x[:, index]
    if not np.isfinite(x).all():
        return "a non-finite value"
    x = np.ascontiguousarray(x).reshape(-1)
    return S.q6_k(x) if ty == "Q6_K" else S.q8_0(x)


class Exl3Dense(C.CheckpointDense):
    """checkpoint.CheckpointDense over an EXL3 checkpoint: the same GGUF names, order and forms.  A projection is its
    EXL3 weight W = diag(suh) H inner H diag(svh) computed in double and stored once, as Q6_K where its rows are whole
    256-value blocks (unless `q6k` is false) and Q8_0 elsewhere; a tensor the engine reads as BF16 is rounded to it
    (`rounded` counts the values that were not BF16 already)."""

    SOURCES = ("BF16", "F16", "F32", "EXL3", "EXL3ROWS")

    def __init__(self, ck: Exl3Checkpoint, q6k: bool = True):
        self.rounded = {}
        super().__init__(DenseView(ck), q6k=q6k)

    def bytes(self, name) -> np.ndarray:
        t = self.tensors[name]
        if t.type_name != "BF16":
            return super().bytes(name)
        x = self.values(name)
        if not np.isfinite(x).all():
            raise S.ConvertError(f"{name} has a non-finite value")
        u = x.view(np.uint32)
        inexact = int(np.count_nonzero(u & 0xFFFF))
        if inexact:
            self.rounded[name] = inexact
        out = ((u.astype(np.uint64) + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")   # nearest, ties to even
        return np.frombuffer(out.tobytes(), dtype=np.uint8)

    def jobs(self, name):
        t = self.tensors[name]
        dt = self.ck.t[t.src][2]
        if dt == "BF16":
            yield from super().jobs(name)
            return
        parts = self.ck.parts[t.src]
        ne0, ne1 = t.ne
        if dt == "EXL3ROWS":
            for r in range(0, ne1, 4096):
                yield ("table", parts, r, min(4096, ne1 - r), None, t.type_name)
        elif t.how == "id":
            step = max(128, C.VALUES_PER_JOB // ne0 // 128 * 128)
            for r in range(0, ne1, step):
                yield ("rows", parts, r, min(step, ne1 - r), None, t.type_name)
        elif t.how in ("vrows", "zrows"):
            yield ("all_rows", parts, 0, ne1, C._index(t.how, ne1, 2 * S.GDN_K_HEADS * S.GDN_HEAD), t.type_name)
        elif t.how == "vcols":
            yield ("all_cols", parts, 0, ne1, C._index(t.how, ne0, 0), t.type_name)
        else:
            raise S.ConvertError(f"{name}: no {t.type_name} path for the transform {t.how}")

    def chunks(self, name, pool, depth: int = 64):
        t = self.tensors[name]
        if self.ck.t[t.src][2] == "BF16":
            yield from super().chunks(name, pool, depth)
            return
        for b in S.bounded_imap(pool, _dense_job, self.jobs(name), depth):
            if isinstance(b, str):
                raise S.ConvertError(f"{name}: {b}")
            yield b

    def source_types(self) -> dict:
        """stored type <- source type -> count, for the pack's metadata."""
        out = {}
        for t in self.tensors.values():
            dt = self.ck.t[t.src][2]
            if dt == "EXL3":
                dt = "EXL3 %g bpw" % (X.k2_of_words(self.ck.parts[t.src][0][3][2]) / 2)
            elif dt == "EXL3ROWS":
                tr, sg = self.ck.parts[t.src]
                dt = "EXL3 rows %d bpw" % X.ring_bits(tr[3][1], sg[3][0])
            key = f"{t.type_name} <- {dt}"
            out[key] = out.get(key, 0) + 1
        return out
