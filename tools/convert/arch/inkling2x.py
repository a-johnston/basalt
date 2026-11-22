"""Inkling 2 Mini (architecture `inkling2x`; GGUF architecture `inkling2_loop`), a pack from a llama.cpp GGUF.

Every tensor keeps the GGUF's type: the float tensors (norms, convolution taps, relative-position tables, depth-mixing
queries, loop controllers) in `dense`, the quantized projections in `gguf`, and the routed experts in `experts`, from
llama.cpp's fused gate_up (the gate's rows first) and down. The looped core layers are stored once: the loop (its first
and last layer, the passes and the controller rank) is in gguf_kv and meta.geometry. The vision projector comes as
`--mmproj` (llama.cpp's inkling2v projector GGUF, the section `vision/mmproj`); the MTP drafter is not in the GGUF.
"""
from __future__ import annotations

from pathlib import Path

from . import Arch, gguf_stem, register

# the release is Apache-2.0, whose section 4(a) asks for the license text in every copy
LICENSE_TEXT = (Path(__file__).resolve().parent.parent / "apache-2.0.txt").read_text(encoding="utf-8")
P = "inkling2_loop."
# what the engine reads from gguf_kv: refused when missing
REQUIRED = (
    "block_count", "embedding_length", "expert_count", "expert_used_count", "expert_feed_forward_length",
    "attention.head_count", "attention.head_count_kv", "attention.key_length", "attention.sliding_window",
    "attention.sliding_window_pattern", "attention.relative_length", "attention.relative_extent",
    "shortconv.l_cache", "attn_res.block_size", "loop.start", "loop.end", "loop.count", "loop.gate_rank",
    "vocab_size", "vocab_size_unpadded",
)


class Inkling2(Arch):
    name = "inkling2x"
    gguf_architecture = "inkling2_loop"
    side_sections = False
    mmproj_only = True
    projector_type = "inkling2v"
    needs_hf_dir = False

    def license(self, kv):
        lic = {"name": "Apache-2.0", "text": LICENSE_TEXT}
        if kv.get("general.quantized_by"):
            lic["quantized_by"] = kv["general.quantized_by"]
        return lic

    def default_name(self, first_gguf, kv, sections_from=False):
        stem = gguf_stem(first_gguf)
        base = "Inkling-2-Mini"
        tail = stem[len(base):].strip("-_. ") if stem.lower().startswith(base.lower()) else ""
        return f"{base} {tail}" if tail else base

    def check(self, kv, own):
        from model_sections import ConvertError
        missing = [k for k in REQUIRED if P + k not in kv]
        if missing:
            raise ConvertError("the GGUF lacks " + ", ".join(P + k for k in missing))
        n = int(kv[P + "block_count"])
        block = int(kv[P + "attn_res.block_size"])
        start, end, passes = int(kv[P + "loop.start"]), int(kv[P + "loop.end"]), int(kv[P + "loop.count"])
        if not 0 < start < end <= n or start % block or end % block or passes < 1:
            raise ConvertError(f"the loop [{start}, {end}) x {passes} does not fit {n} layers in blocks of {block}")
        if len(kv[P + "attention.sliding_window_pattern"]) != n:
            raise ConvertError(f"the sliding-window pattern has {len(kv[P + 'attention.sliding_window_pattern'])} layers, not {n}")
        if int(kv[P + "vocab_size_unpadded"]) != len(kv.get("tokenizer.ggml.tokens") or []):
            raise ConvertError("the vocabulary differs from vocab_size_unpadded")

    def meta_extra(self, kv, tensors, own):
        return {
            "arch": self.name,
            # the engine and the server pick the backend by general.architecture: inkling2x, not the GGUF's inkling2_loop
            # (the keys keep its prefix)
            "gguf_kv": {**kv, "general.architecture": self.name},
            "geometry": {k[len(P):]: v for k, v in kv.items() if k.startswith(P)},
            "vision": None,
            "mtp": None,
        }


register(Inkling2(), "inkling2_loop")
