"""Inkling 2 from an EXL3 checkpoint (ExLlamaV3's format, as the SAGE REXL3 quants of thinkingmachines/Inkling-2-Mini are),
for tools/convert/convert.py --sections-from BASE --exl3 DIR when BASE is an Inkling 2 pack (tools/convert/arch/inkling2x.py),
which gives the GGUF metadata, the tokenizer and every tensor's GGUF name and shape.

  experts          per expert [gate trellis | up trellis | down trellis | suh_g svh_g suh_u svh_u suh_d svh_d] as
                   exl3_checkpoint.py writes them, each expert at its own bitrate: a mixed-rate quant gives each expert
                   its own gate/up and down rates (gate and up of one expert must agree)
  experts.layout   v8: `layer expert gu_type d_type offset blob_bytes`, the types 160 + 2 K
  gguf             every other language-model tensor under BASE's names and shapes, in the forms of llama.cpp's
                   conversion/inkling2.py: an EXL3 matrix's weight computed in double and stored once as Q8_0 (the head's
                   rows past the vocabulary dropped), the projections the engine multiplies but the checkpoint keeps in
                   BF16 (the sinks, the convolutions' input taps) as Q8_0, the token embedding as BF16, the rest as F32:
                   the convolutions' taps squeezed (a full layer's static taps reversed), the relative-bias tables
                   transposed, the loop's cross-pass queries stacked per core layer (q, k, v, ffn)

Every matrix must use the mul1 codebook; the vision tower and the drafter (mtp.safetensors) are not read. Needs numpy.
"""
from __future__ import annotations

import re

import numpy as np

import exl3 as X
import exl3_checkpoint as E
import model_sections as S

P = "model.llm."
TYPE_BASE = 160
MUL1 = 0x83DCD12D
ROWS = 128                 # rows of one rebuild job: whole Hadamard blocks of outputs
Q8_0, F32, BF16 = 8, 0, 30
# GGUF name (after `blk.N.`) -> the checkpoint's (after `model.llm.layers.N.`)
EXL3_MATS = {"attn_q.weight": "attn.wq_du", "attn_k.weight": "attn.wk_dv", "attn_v.weight": "attn.wv_dv",
             "attn_output.weight": "attn.wo_ud", "attn_rel.weight": "attn.wr_du", "attn_gate.weight": "attn.gate_attn"}
Q8_MATS = {"attn_sink_proj.weight": "attn.attn_sink_proj.weight", "attn_k_conv_proj.weight": "attn.k_sconv.weight_proj",
           "attn_v_conv_proj.weight": "attn.v_sconv.weight_proj"}
PLAIN = {"attn_q_clip.weight": "attn.q_l2_clip.scale", "attn_rel_clip.weight": "attn.rel_l2_clip.scale",
         "attn_norm.weight": "attn_norm.weight", "attn_q_inp_norm.weight": "q_input_norm.weight",
         "attn_k_inp_norm.weight": "k_input_norm.weight", "attn_v_inp_norm.weight": "v_input_norm.weight",
         "ffn_norm.weight": "mlp_norm.weight", "ffn_gate_inp.weight": "mlp.gate.weight", "exp_probs_b.bias": "mlp.gate.bias"}
SQUEEZED = {"attn_post_conv.weight": "attn_sconv.weight", "ffn_post_conv.weight": "mlp_sconv.weight"}
CONTROLLERS = {"q": "attn", "k": "k", "v": "v", "ffn": "mlp"}
GLOBAL_PLAIN = {"token_embd_norm.weight": "embed_norm.weight", "output_norm.weight": "norm.weight",
                "output_res_query.weight": "attn_res.output_query"}
LAYER_TENSOR = re.compile(r"blk\.(\d+)\.(.+)")


def _values(ck: E.Exl3Checkpoint, name: str) -> np.ndarray:
    """A BF16, F16 or F32 tensor of the checkpoint as float32 in its own shape."""
    p, off, dt, shape, n = ck.t[name]
    raw = np.fromfile(p, dtype=np.uint8, count=n, offset=off)
    if dt == "BF16":
        return S._bf16_to_f32(raw.view("<u2")).reshape(shape)
    if dt == "F16":
        return raw.view("<f2").astype(np.float32).reshape(shape)
    if dt == "F32":
        return raw.view("<f4").reshape(shape).copy()
    raise S.ConvertError(f"{name} is {dt}")


def _exl3_q8_job(args):
    """Rows [row0, row0 + rows) of an EXL3 matrix, as Q8_0: whole 128-row blocks rebuilt, the rows past `n_out` dropped."""
    parts, row0, rows, n_out = args
    end = min(row0 + rows, n_out)
    x = E._exl3_rows(parts, row0, rows)[: end - row0]
    if not np.isfinite(x).all():
        return f"a non-finite value in rows {row0} .."
    return S.q8_0(np.ascontiguousarray(x).reshape(-1))


class Tensor:
    """A pack tensor: its GGUF name, type and shape (GGUF order), and how it is made."""

    def __init__(self, name, type_id, ne, kind, src):
        self.name, self.type_id, self.ne, self.kind, self.src = name, type_id, [int(d) for d in ne], kind, src

    @property
    def type_name(self):
        return {Q8_0: "Q8_0", F32: "F32", BF16: "BF16"}[self.type_id]

    def nbytes(self) -> int:
        n = int(np.prod(self.ne))
        return n // 32 * 34 if self.type_id == Q8_0 else n * (4 if self.type_id == F32 else 2)


class Inkling2Exl3:
    """The tensors and experts of an Inkling 2 pack read from an EXL3 checkpoint, against BASE's tensor table."""

    def __init__(self, ck: E.Exl3Checkpoint, base_tensors: dict, geometry: dict):
        self.ck = ck
        g = geometry
        self.n_layers, self.n_expert = int(g["block_count"]), int(g["expert_count"])
        self.h, self.ff = int(g["embedding_length"]), int(g["expert_feed_forward_length"])
        self.loop_start, self.loop_end = int(g["loop.start"]), int(g["loop.end"])
        self.tensors = [self._tensor(n, bt) for n, bt in base_tensors.items()]
        self._experts()

    # ---------------------------------------------------------------------------------------------- the tensors
    def _need(self, name):
        if name not in self.ck.t:
            raise S.ConvertError(f"the checkpoint has no {name}")
        return name

    def _exl3_parts(self, stem):
        p, off, dt, shape, size = self.ck.t[self._need(stem + ".mul1")]
        if dt != "I32" or shape or size != 4 or int(np.fromfile(p, dtype="<u4", count=1, offset=off)[0]) != MUL1:
            raise S.ConvertError(f"{stem}: only mul1 ({MUL1:#x}) is read")
        tr, suh, svh = (self.ck.t[self._need(f"{stem}.{x}")] for x in ("trellis", "suh", "svh"))
        dims = tr[3]
        if tr[2] != "I16" or len(dims) != 3 or any(v <= 0 or v % 8 for v in dims[:2]) or tr[4] != 2 * int(np.prod(dims)):
            raise S.ConvertError(f"{stem}.trellis is {tr[2]} {dims}, expected I16 tiles of whole Hadamard blocks")
        try:
            X.k2_of_words(dims[2])
        except X.Exl3Error as err:
            raise S.ConvertError(f"{stem}.trellis: {err}") from None
        for name, vec, width in (("suh", suh, dims[0] * 16), ("svh", svh, dims[1] * 16)):
            if vec[2:] != ("F16", [width], 2 * width):
                raise S.ConvertError(f"{stem}.{name} is {vec[2]} {vec[3]}, expected F16 [{width}]")
        return tr, suh, svh

    def _tensor(self, name, bt) -> Tensor:
        ne = bt["shape"]
        m = LAYER_TENSOR.fullmatch(name)
        if m is None:
            if name == "token_embd.weight":
                return Tensor(name, BF16, ne, "bf16", self._need(P + "embed.weight"))
            if name == "output.weight":
                return Tensor(name, Q8_0, ne, "exl3", self._exl3_parts(P + "unembed"))
            if name in GLOBAL_PLAIN:
                return Tensor(name, F32, ne, "plain", self._need(P + GLOBAL_PLAIN[name]))
            raise S.ConvertError(f"BASE's {name} has no counterpart in the checkpoint")
        l, rest = int(m.group(1)), m.group(2)
        lp = f"{P}layers.{l}."
        if rest in EXL3_MATS:
            return Tensor(name, Q8_0, ne, "exl3", self._exl3_parts(lp + EXL3_MATS[rest]))
        if rest in Q8_MATS:
            return Tensor(name, Q8_0, ne, "q8", self._need(lp + Q8_MATS[rest]))
        if rest in PLAIN:
            return Tensor(name, F32, ne, "plain", self._need(lp + PLAIN[rest]))
        if rest in SQUEEZED:
            return Tensor(name, F32, ne, "squeeze", self._need(lp + SQUEEZED[rest]))
        if rest in ("attn_k_conv.weight", "attn_v_conv.weight"):
            c = "k_sconv" if rest.startswith("attn_k") else "v_sconv"
            if lp + f"attn.{c}.static_w" in self.ck.t:   # a full layer: static taps, stored oldest first
                return Tensor(name, F32, ne, "reversed_t", lp + f"attn.{c}.static_w")
            return Tensor(name, F32, ne, "squeeze", self._need(lp + f"attn.{c}.weight"))
        if rest == "attn_rel_bias.weight":
            return Tensor(name, F32, ne, "transpose", self._need(lp + "attn.rel_logits_proj.proj"))
        if rest == "attn_res_query.weight":   # queries.i feeds block i + 1
            return Tensor(name, F32, ne, "plain", self._need(f"{P}attn_res.attn_queries.{l - 1}"))
        if rest == "ffn_res_query.weight":
            return Tensor(name, F32, ne, "plain", self._need(f"{P}attn_res.mlp_queries.{l}"))
        if rest == "loop_msg_query.weight":
            return Tensor(name, F32, ne, "messages", l - self.loop_start)
        m = re.fullmatch(r"loop_(q|k|v|ffn)_ctrl_(down|gain|forget)\.weight", rest)
        if m:
            return Tensor(name, F32, ne, "plain", self._need(f"{lp}controllers.{CONTROLLERS[m.group(1)]}.{m.group(2)}_weight"))
        raise S.ConvertError(f"BASE's {name} has no counterpart in the checkpoint")

    def values(self, t: Tensor) -> np.ndarray:
        """A non-EXL3 tensor's values in the GGUF order, as float32 of shape reversed(ne)."""
        if t.kind == "messages":
            q = _values(self.ck, self._need(P + "middle_cycle_cross_pass_queries"))
            kv = _values(self.ck, self._need(P + "middle_cycle_cross_pass_kv_queries"))
            i = t.src
            x = np.stack([q[2 * i], kv[2 * i], kv[2 * i + 1], q[2 * i + 1]])
        else:
            x = _values(self.ck, t.src)
            if t.kind == "squeeze":
                x = x.reshape(x.shape[0], -1)
            elif t.kind == "reversed_t":
                x = np.ascontiguousarray(x[::-1].T)
            elif t.kind == "transpose":
                x = np.ascontiguousarray(x.T)
        want = tuple(reversed([d for d in t.ne]))
        if x.size != int(np.prod(want)):
            raise S.ConvertError(f"{t.name}: the checkpoint's {list(x.shape)} does not fill BASE's {t.ne}")
        x = x.reshape(want)
        if not np.isfinite(x).all():
            raise S.ConvertError(f"{t.name} has a non-finite value")
        return x

    def bytes(self, t: Tensor) -> bytes:
        """A tensor made here in one piece (all but the EXL3 matrices)."""
        if t.type_id == BF16:   # the token embedding, its bytes as they are
            p, off, dt, shape, n = self.ck.t[t.src]
            if dt != "BF16" or list(reversed(shape)) != t.ne:
                raise S.ConvertError(f"{t.src} is {dt} {shape}, expected BF16 of BASE's {t.ne}")
            return np.fromfile(p, dtype=np.uint8, count=n, offset=off).tobytes()
        x = self.values(t)
        if t.type_id == Q8_0:
            return S.q8_0(x.reshape(-1))
        return x.astype("<f4").tobytes()

    def jobs(self, t: Tensor):
        n_out = t.ne[1]
        rows_in = t.src[0][3][1] * 16
        expected_rows = ((n_out + ROWS - 1) // ROWS * ROWS) if t.name == "output.weight" else n_out
        if t.src[0][3][0] * 16 != t.ne[0] or rows_in != expected_rows:
            raise S.ConvertError(f"{t.name}: the EXL3 matrix is {t.src[0][3][0] * 16} x {rows_in}, BASE's {t.ne}")
        for r in range(0, n_out, ROWS):
            yield (t.src, r, ROWS, n_out)

    def chunks(self, t: Tensor, pool, depth: int = 32):
        for b in S.bounded_imap(pool, _exl3_q8_job, self.jobs(t), depth):
            if isinstance(b, str):
                raise S.ConvertError(f"{t.name}: {b}")
            yield b

    # ---------------------------------------------------------------------------------------------- the experts
    def _experts(self):
        h, ff = self.h, self.ff
        if h % 128 or ff % 128:
            raise S.ConvertError(f"expert widths {h} x {ff} are not whole 128-value Hadamard blocks")
        dims = {"gate_proj": (h, ff), "up_proj": (h, ff), "down_proj": (ff, h)}   # inputs, outputs
        self.entries, self.layout, self.rates = [], [], {}
        offset = 0
        for l in range(self.n_layers):
            for e in range(self.n_expert):
                ex, k2 = [], []
                for r in E.ROLES:
                    p = f"{P}layers.{l}.mlp.experts.{e}.{r}."
                    k, n = dims[r]
                    tr, suh, svh = self._exl3_parts(p[:-1])
                    if tr[3][:2] != [k // 16, n // 16]:
                        raise S.ConvertError(f"{p}trellis is {tr[3]}, expected [{k // 16}, {n // 16}, 16 K]")
                    k2.append(X.k2_of_words(tr[3][2]))
                    ex.append(((tr[0], tr[1], tr[3]), (suh[0], suh[1], k), (svh[0], svh[1], n)))
                if k2[0] != k2[1]:
                    raise S.ConvertError(f"layer {l} expert {e}: gate is {k2[0] / 2} bpw and up {k2[1] / 2}")
                blob = 2 * (h * ff * k2[0] // 16) + h * ff * k2[2] // 16 + 6 * (h + ff)
                self.entries.append([ex])
                self.layout.append((l, e, TYPE_BASE + k2[0], TYPE_BASE + k2[2], offset, blob))
                key = f"{k2[0] / 2:g}/{k2[2] / 2:g}"
                self.rates[key] = self.rates.get(key, 0) + 1
                offset += blob
        self.total = offset

    def layout_text(self) -> bytes:
        head = ("# experts v8: layer expert gu_type d_type offset blob_bytes (n_expert %d, total %d); EXL3 mul1 experts, "
                "type 160 + 2 K, a blob ends in fp16 suh_g svh_g suh_u svh_u suh_d svh_d\n" % (self.n_expert, self.total))
        return (head + "".join("%d %d %d %d %d %d\n" % x for x in self.layout)).encode("utf-8")

    def expert_chunks(self, pool, depth: int = 64):
        for b in S.bounded_imap(pool, E._expert_job, self.entries, depth):
            if isinstance(b, str):
                raise S.ConvertError(b)
            yield b
