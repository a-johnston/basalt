// src/models/inkling2x/pack.hpp - an Inkling 2 .basalt pack as the loader sees it: geometry, the tensor table and the experts
// layout, all from the pack's meta section (tools/convert/arch/inkling2x.py writes it).
#pragma once

#include "basalt/core/model_file.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace basalt::models::inkling2x {

struct Geometry {
    int n_layers = 0, n_embd = 0, n_vocab = 0, vocab_padded = 0;
    int n_head = 0, n_kv = 0, head_dim = 0;
    int n_expert = 0, n_used = 0, n_ff_exp = 0;
    int window = 0, d_rel = 0, rel_extent = 0, n_conv = 0, block = 0;
    float rms_eps = 1e-6f, qk_clip = 0, rel_clip = 0, log_alpha = 0, log_floor = 1;
    float expert_scale = 1, situ_beta = 0, situ_up = 0, logit_cap = 0;
    int loop_start = 0, loop_end = 0, passes = 1, gate_rank = 0;
    int64_t context_length = 0;
    std::vector<bool> swa;   ///< per stored layer: a sliding-window layer

    int core() const { return loop_end - loop_start; }
    /// The layer slots a token runs: the stored layers, then the core once more per pass after the first.
    int slots() const { return n_layers + (passes - 1) * core(); }
    /// The stored layer of slot `s`.
    int layer_of(int s) const { return s < n_layers ? s : loop_start + (s - n_layers) % core(); }
};

struct TensorRef {
    std::string name;
    int type = 0;
    std::vector<uint64_t> shape;   ///< GGUF order, innermost first
    uint64_t offset = 0;           ///< absolute file offset
    uint64_t bytes = 0;
};

/// One routed expert's blob: [gate | up | down] (and its vectors, for EXL3) at `offset` inside the experts section.
struct ExpertBlob {
    int gu_type = 0, d_type = 0;
    uint64_t offset = 0, bytes = 0;
};

struct ExpertLayer {
    std::vector<ExpertBlob> blobs;   ///< per expert: one type pair for a whole layer, or each its own (a mixed-rate quant)
};

struct Pack {
    core::ModelFile file;
    Geometry geo;
    std::map<std::string, TensorRef> tensors;
    std::vector<ExpertLayer> experts;   ///< per stored layer
    core::FileRange experts_range;

    const TensorRef* find(const std::string& n) const {
        auto it = tensors.find(n);
        return it == tensors.end() ? nullptr : &it->second;
    }
};

/// Opens the pack and fills `p`. False with `err` when it is not an Inkling 2 pack, is malformed or has a layout this
/// engine does not run.
bool open_pack(const std::string& path, Pack& p, std::string& err);

}  // namespace basalt::models::inkling2x
