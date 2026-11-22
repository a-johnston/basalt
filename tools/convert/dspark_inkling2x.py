"""Inkling 2's DSpark drafter (the checkpoint's mtp.safetensors with the mtp_config of its config.json) as the dspark/gguf
section of an Inkling 2 pack, for convert.py --dspark mtp.safetensors --into PACK.basalt --undo-dir DIR.

The drafter (vLLM's vllm/models/inkling2/nvidia/dspark.py, Apache-2.0) is three dense layers over a block of the anchor
and seven mask tokens, with context keys and values from the target's features, and a recurrent Markov head that biases
the target head's logits. The section holds, each tensor at a multiple of TENSOR_ALIGN, with GGUF names and shapes:

  Q8_0  dspark.blk.N.{attn_q, attn_k, attn_v, attn_output, attn_k_ctx, attn_v_ctx, ffn_gate, ffn_up, ffn_down}.weight
  F32   dspark.blk.N.{feature_norm, attn_norm, ffn_norm, attn_q_norm, attn_k_norm, attn_k_ctx_norm}.weight,
        dspark.{output_norm, mask_embd, fuse, markov_feature_norm, markov_rnn, markov_rnn_bias, markov_fold_bias}
  BF16  dspark.markov_token_embd [n_vocab, rank] (markov_w1), dspark.markov_fold [n_vocab, rank]

markov_fold = head @ hidden_out.weight and markov_fold_bias = head @ hidden_out.bias, in double, with the pack's own head
(its first n_vocab rows): the Markov head's latent then gives logits directly, as vLLM folds it at load. meta.dspark
records the config, the tensors (offsets within the section) and the source.

Needs numpy.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

import model_sections as S
from model_file import TENSOR_ALIGN, ModelFile

ARCH = "inkling2-dspark"
Q8_0, F32, BF16 = 8, 0, 30
TYPE_NAMES = {Q8_0: "Q8_0", F32: "F32", BF16: "BF16"}
# GGUF name (after dspark.blk.N.) -> the checkpoint's (after layers.N.)
LAYER_Q8 = {"attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
            "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
            "attn_k_ctx.weight": "self_attn.k_proj_ctx.weight", "attn_v_ctx.weight": "self_attn.v_proj_ctx.weight",
            "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight", "ffn_down.weight": "mlp.down_proj.weight"}
LAYER_F32 = {"feature_norm.weight": "dflare_norm.weight", "attn_norm.weight": "input_layernorm.weight",
             "ffn_norm.weight": "post_attention_layernorm.weight", "attn_q_norm.weight": "self_attn.q_norm.weight",
             "attn_k_norm.weight": "self_attn.k_norm.weight", "attn_k_ctx_norm.weight": "self_attn.k_norm_ctx.weight"}
TOP_F32 = {"dspark.output_norm": "norm.weight", "dspark.mask_embd": "mask_embed", "dspark.fuse": "fuse_weight",
           "dspark.markov_feature_norm": "markov_head.dflare_norm.weight", "dspark.markov_rnn": "markov_head.rnn_proj.weight",
           "dspark.markov_rnn_bias": "markov_head.rnn_proj.bias"}


class DsparkError(Exception):
    pass


class Safetensors:
    """One .safetensors file's tensors: name -> (dtype, shape, offset in the file, bytes)."""

    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
        head.pop("__metadata__", None)
        self.t = {k: (v["dtype"], v["shape"], 8 + n + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0])
                  for k, v in head.items()}

    def values(self, name: str) -> np.ndarray:
        """float32 in the tensor's own shape."""
        if name not in self.t:
            raise DsparkError(f"{self.path.name} has no {name}")
        dt, shape, off, n = self.t[name]
        raw = np.fromfile(self.path, dtype=np.uint8, count=n, offset=off)
        if dt == "BF16":
            x = S._bf16_to_f32(raw.view("<u2"))
        elif dt == "F32":
            x = raw.view("<f4").copy()
        elif dt == "F16":
            x = raw.view("<f2").astype(np.float32)
        else:
            raise DsparkError(f"{name} is {dt}")
        return x.reshape(shape)

    def raw_bf16(self, name: str) -> bytes:
        dt, _, off, n = self.t[name]
        if dt != "BF16":
            raise DsparkError(f"{name} is {dt}, expected BF16")
        return np.fromfile(self.path, dtype=np.uint8, count=n, offset=off).tobytes()


def check_config(c: dict, geo: dict, n_vocab: int) -> None:
    """Stops on a drafter this engine does not run."""
    want = {"hidden_size": int(geo["embedding_length"]), "head_dim": 128, "markov_head_type": "rnn_hidden", "markov_rank": 256}
    for k, v in want.items():
        if c.get(k) != v:
            raise DsparkError(f"mtp_config {k} = {c.get(k)!r}, this engine runs {v!r}")
    if c.get("q_bias") or c.get("o_bias"):
        raise DsparkError("mtp_config has attention biases, which this engine does not run")
    if c.get("num_attention_heads", 0) % max(c.get("num_key_value_heads", 1), 1):
        raise DsparkError("mtp_config's query heads are not a multiple of its K/V heads")
    visits = int(geo["block_count"]) + (int(geo["loop.count"]) - 1) * (int(geo["loop.end"]) - int(geo["loop.start"]))
    ids = list(c.get("target_layer_ids", []))
    if len(ids) < 2 or ids != sorted(set(ids)) or ids[-1] != visits:
        raise DsparkError(f"target_layer_ids {ids}: sorted layer visits ending in the visit count {visits} (the final-norm output)")
    if not 2 <= int(c.get("block_size", 0)) <= 16:
        raise DsparkError(f"block_size {c.get('block_size')}: 2 to 16 rows a block")
    if c.get("local_layer_ids") and int(c.get("sliding_window_size", 0)) <= 0:
        raise DsparkError("mtp_config has local layers but no sliding_window_size")


def head_values(pack: ModelFile, meta: dict, n_vocab: int) -> np.ndarray:
    """The first n_vocab rows of the pack's output.weight as float64 [n_vocab, n_embd]."""
    t = meta["tensors"]["output.weight"]
    sec = "gguf" if "section" not in t else t["section"]
    off, _ = pack.section(sec)
    raw = np.fromfile(pack.path, dtype=np.uint8, count=t["bytes"], offset=off + t["offset"])
    ne0, ne1 = t["shape"][0], t["shape"][1]
    if ne1 < n_vocab:
        raise DsparkError(f"the pack's head has {ne1} rows for {n_vocab} tokens")
    rb = len(raw) // ne1
    x = S._dequantize(raw[: rb * n_vocab].copy(), t["type"]).reshape(n_vocab, ne0)
    return x.astype(np.float64)


class Tensor:
    def __init__(self, name, type_id, ne, data: bytes):
        self.name, self.type_id, self.ne, self.data = name, type_id, [int(d) for d in ne], data


def build(mtp: Path, pack_path: Path) -> tuple[bytes, dict]:
    """The section's bytes and its meta.dspark record."""
    cfg_path = mtp.with_name("config.json")
    if not cfg_path.is_file():
        raise DsparkError(f"no config.json beside {mtp} (its mtp_config describes the drafter)")
    c = json.loads(cfg_path.read_text()).get("mtp_config")
    if not c:
        raise DsparkError(f"{cfg_path} has no mtp_config")
    pack = ModelFile(str(pack_path))
    meta = json.loads(pack.read("meta"))
    if meta.get("arch") != "inkling2x":
        raise DsparkError(f"{pack_path} is no inkling2x pack")
    geo = meta["geometry"]
    n_vocab = len(meta["gguf_kv"]["tokenizer.ggml.tokens"])
    check_config(c, geo, n_vocab)
    st = Safetensors(mtp)
    tensors: list[Tensor] = []
    E, n_layers = int(c["hidden_size"]), int(c["num_hidden_layers"])
    for i in range(n_layers):
        for gg, ck in LAYER_Q8.items():
            w = st.values(f"layers.{i}.{ck}")
            tensors.append(Tensor(f"dspark.blk.{i}.{gg}", Q8_0, [w.shape[1], w.shape[0]], S.q8_0(np.ascontiguousarray(w).reshape(-1))))
        for gg, ck in LAYER_F32.items():
            w = st.values(f"layers.{i}.{ck}")
            tensors.append(Tensor(f"dspark.blk.{i}.{gg}", F32, [w.size], w.astype("<f4").tobytes()))
    for gg, ck in TOP_F32.items():
        w = st.values(ck)
        tensors.append(Tensor(gg, F32, list(reversed(w.shape)), np.ascontiguousarray(w).astype("<f4").tobytes()))
    w1 = st.t["markov_head.markov_w1.weight"]
    rank = int(c["markov_rank"])
    if w1[1][1] != rank or w1[1][0] < n_vocab:
        raise DsparkError(f"markov_w1 is {w1[1]}, expected [>= {n_vocab}, {rank}]")
    tensors.append(Tensor("dspark.markov_token_embd", BF16, [rank, n_vocab], st.raw_bf16("markov_head.markov_w1.weight")[: n_vocab * rank * 2]))
    # the Markov head's latent through hidden_out and the head, folded: fold = head @ W, fold_bias = head @ b (double)
    ho_w = st.values("markov_head.hidden_out.weight").astype(np.float64)   # [E, rank]
    ho_b = st.values("markov_head.hidden_out.bias").astype(np.float64)     # [E]
    if ho_w.shape != (E, rank):
        raise DsparkError(f"hidden_out is {list(ho_w.shape)}, expected [{E}, {rank}]")
    head = head_values(pack, meta, n_vocab)
    fold = head @ ho_w
    fold_b = head @ ho_b
    if not (np.isfinite(fold).all() and np.isfinite(fold_b).all()):
        raise DsparkError("the folded Markov head has a non-finite value")
    fold_bf16 = S.ggufq.quantize(fold.astype(np.float32), S.ggufq.GGMLQuantizationType.BF16).tobytes()   # round to nearest even
    tensors.append(Tensor("dspark.markov_fold", BF16, [rank, n_vocab], fold_bf16))
    tensors.append(Tensor("dspark.markov_fold_bias", F32, [n_vocab], fold_b.astype("<f4").tobytes()))
    # the section: every tensor at a multiple of TENSOR_ALIGN
    blob, table = bytearray(), {}
    for t in tensors:
        pad = (-len(blob)) % TENSOR_ALIGN
        blob += b"\0" * pad
        table[t.name] = {"type": TYPE_NAMES[t.type_id], "type_id": t.type_id, "shape": t.ne, "offset": len(blob), "bytes": len(t.data)}
        blob += t.data
    record = {"section": "dspark/gguf", "architecture": ARCH, "bytes": len(blob), "config": c, "tensors": table,
              "markov_fold": "the pack's head (first n_vocab rows) times hidden_out, in double"}
    return bytes(blob), record
