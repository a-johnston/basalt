// src/models/inkling2x/meta.hpp - what the file's metadata says about the model: free of the runtime, so a CPU test can
// check it without linking the model.
#pragma once

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/model/adapter.hpp"

#include <string>

namespace basalt::models::inkling2x {

/// The architecture name the GGUF metadata (and so a pack's `gguf_kv`) carries.
inline constexpr const char* kGgufArch = "inkling2_loop";
/// The id of the model.
inline constexpr const char* kArch = "inkling2x";

inline const MetaValue* key(const GgufFile& g, const char* suffix) {
    return g.get(std::string(kGgufArch) + "." + suffix);   // the keys keep the GGUF's prefix in a pack too
}

inline uint64_t key_u(const GgufFile& g, const char* suffix) {
    const MetaValue* v = key(g, suffix);
    return v ? v->u : 0;
}

/// The looped core: layers [start, end) run `passes` times, each pass after the first mixing in the previous one's
/// blocks through controllers of rank `gate_rank`.
struct Loop {
    int start = 0, end = 0, passes = 0, gate_rank = 0;
    bool varying = false;   ///< trained for a pass count chosen at the start
};

inline Loop loop_of(const GgufFile& g) {
    Loop l;
    l.start = (int) key_u(g, "loop.start");
    l.end = (int) key_u(g, "loop.end");
    l.passes = (int) key_u(g, "loop.count");
    l.gate_rank = (int) key_u(g, "loop.gate_rank");
    l.varying = key_u(g, "loop.varying") != 0;
    return l;
}

inline std::string validate_meta(const GgufFile& g) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != kArch)
        return "architecture is '" + arch->s + "', this adapter requires '" + kArch + "'";
    for (const char* k : {"block_count", "embedding_length", "expert_count", "expert_used_count", "expert_feed_forward_length",
                          "attention.key_length", "attention.sliding_window", "attention.relative_length",
                          "attention.relative_extent", "shortconv.l_cache", "attn_res.block_size", "loop.end", "loop.count",
                          "loop.gate_rank", "vocab_size", "vocab_size_unpadded"})
        if (key_u(g, k) == 0) return std::string("missing or zero ") + kGgufArch + "." + k;
    const int n = (int) key_u(g, "block_count");
    const int block = (int) key_u(g, "attn_res.block_size");
    const Loop l = loop_of(g);
    if (l.start <= 0 || l.start >= l.end || l.end > n || l.start % block || l.end % block || l.passes < (l.varying ? 1 : 2))
        return "the loop [" + std::to_string(l.start) + ", " + std::to_string(l.end) + ") x " + std::to_string(l.passes) +
               " does not fit " + std::to_string(n) + " layers in blocks of " + std::to_string(block);
    const MetaValue* swa = key(g, "attention.sliding_window_pattern");
    if (!swa || (int) swa->count != n) return std::string("the sliding-window pattern is not one entry per layer (") + kGgufArch + ".attention.sliding_window_pattern)";
    const MetaValue* tokens = g.get("tokenizer.ggml.tokens");
    if (!tokens || tokens->count != key_u(g, "vocab_size_unpadded")) return "the vocabulary is not vocab_size_unpadded tokens";
    return {};
}

inline model::ModelDesc describe_meta(const GgufFile& g) {
    model::ModelDesc d;
    d.arch = kArch;
    d.n_layers = (int64_t) key_u(g, "block_count");     // the layers stored; the loop runs some of them again
    d.n_embd = (int64_t) key_u(g, "embedding_length");
    d.n_expert = (int64_t) key_u(g, "expert_count");
    d.n_expert_used = (int64_t) key_u(g, "expert_used_count");
    d.n_ff_exp = (int64_t) key_u(g, "expert_feed_forward_length");
    d.context_length = (int64_t) key_u(g, "context_length");
    if (const MetaValue* t = g.get("tokenizer.ggml.tokens")) d.n_vocab = (int64_t) t->count;
    return d;
}

}  // namespace basalt::models::inkling2x
