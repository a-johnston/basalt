// src/models/inkling2x/pack.cpp - reads the Inkling 2 pack's meta section.
#include "models/inkling2x/pack.hpp"

#include <algorithm>
#include <sstream>

namespace basalt::models::inkling2x {
namespace {

using core::Json;

double num_or(const Json* j, double d) { return j && j->is_number() ? j->num() : d; }

std::vector<bool> bools(const Json* j) {
    std::vector<bool> v;
    if (j && j->is_array())
        for (const Json& x : j->items) v.push_back(x.is_number() ? x.num() != 0 : x.b);
    return v;
}

// the first entry of a per-layer array (the layers agree; checked by the caller)
std::vector<double> numbers(const Json* j) {
    std::vector<double> v;
    if (j && j->is_array())
        for (const Json& x : j->items) v.push_back(x.num());
    else if (j && j->is_number())
        v.push_back(j->num());
    return v;
}

}  // namespace

bool open_pack(const std::string& path, Pack& p, std::string& err) {
    if (!p.file.open(path, err) || !p.file.load_meta(err)) return false;
    const Json& meta = p.file.meta();
    const Json* arch = meta.get("arch");
    if (!arch || !arch->is_string() || arch->s != "inkling2x") {
        err = path + ": not an inkling2x pack";
        return false;
    }
    const Json* gj = meta.get("geometry");
    const Json* tj = meta.get("tensors");
    if (!gj || !gj->is_object() || !tj || !tj->is_object()) {
        err = path + ": meta has no geometry / tensors";
        return false;
    }
    Geometry& g = p.geo;
    auto u = [&](const char* k, double d = 0) { return (int) num_or(gj->get(k), d); };
    auto f = [&](const char* k, double d = 0) { return (float) num_or(gj->get(k), d); };
    g.n_layers = u("block_count");
    g.n_embd = u("embedding_length");
    g.head_dim = u("attention.key_length");
    g.n_expert = u("expert_count");
    g.n_used = u("expert_used_count");
    g.n_ff_exp = u("expert_feed_forward_length");
    g.window = u("attention.sliding_window");
    g.d_rel = u("attention.relative_length");
    g.rel_extent = u("attention.relative_extent");
    g.n_conv = u("shortconv.l_cache");
    g.block = u("attn_res.block_size");
    g.vocab_padded = u("vocab_size");
    g.rms_eps = f("attention.layer_norm_rms_epsilon", 1e-6);
    g.qk_clip = f("attention.qk_clip");
    g.rel_clip = f("attention.relative_clip");
    g.log_alpha = f("attention.log_scale_alpha");
    g.log_floor = f("attention.log_scale_floor", 1);
    g.expert_scale = f("expert_weights_scale", 1);
    g.situ_beta = f("activation.situ_beta");
    g.situ_up = f("activation.situ_linear_beta");
    g.logit_cap = f("final_logit_softcapping");
    g.loop_start = u("loop.start");
    g.loop_end = u("loop.end");
    g.passes = u("loop.count", 1);
    g.gate_rank = u("loop.gate_rank");
    g.context_length = (int64_t) num_or(gj->get("context_length"), 0);
    g.swa = bools(gj->get("attention.sliding_window_pattern"));
    const std::vector<double> heads = numbers(gj->get("attention.head_count")), kv = numbers(gj->get("attention.head_count_kv"));
    if (const Json* kvj = meta.get("gguf_kv"))
        if (const Json* t = kvj->get("tokenizer.ggml.tokens")) g.n_vocab = (int) t->count;
    if (g.n_layers <= 0 || g.n_embd <= 0 || g.n_expert <= 0 || heads.empty() || kv.empty() || (int) g.swa.size() != g.n_layers ||
        g.n_vocab <= 0 || g.vocab_padded < g.n_vocab) {
        err = path + ": meta.geometry is incomplete";
        return false;
    }
    for (double h : heads) if ((int) h != (int) heads[0]) { err = path + ": the layers' head counts differ"; return false; }
    for (double h : kv) if ((int) h != (int) kv[0]) { err = path + ": the layers' K/V head counts differ"; return false; }
    g.n_head = (int) heads[0];
    g.n_kv = (int) kv[0];
    // the layouts this engine runs
    if (g.head_dim != 128 || g.d_rel != 16 || g.n_conv != 4 || g.n_kv <= 0 || g.n_head % g.n_kv || g.n_embd % 256 || g.n_ff_exp % 32 ||
        g.rel_extent <= 0 || g.window <= 0 || g.block <= 0 || g.loop_start <= 0 || g.loop_start >= g.loop_end ||
        g.loop_end > g.n_layers || g.loop_start % g.block || g.loop_end % g.block || g.passes < 1 || g.gate_rank <= 0 ||
        g.n_used > 16 || g.n_expert > 512 || g.n_expert % 32 || g.situ_beta <= 0) {
        err = path + ": an Inkling 2 layout this engine does not run (heads of 128, relative states of 16, convolutions of 4 taps, "
                     "the loop on the block grid, situ experts)";
        return false;
    }
    for (const auto& [name, t] : tj->members) {
        TensorRef r;
        r.name = name;
        r.type = (int) num_or(t.get("type_id"), 0);
        if (const Json* sh = t.get("shape"))
            for (const Json& d : sh->items) r.shape.push_back(d.u64());
        r.bytes = (uint64_t) num_or(t.get("bytes"), 0);
        std::string section = "gguf";
        if (const Json* s = t.get("section"); s && s->is_string()) section = s->s;
        const core::ModelFile::Section* sec = p.file.find(section);
        if (!sec) {
            err = path + ": tensor " + name + " names a missing section " + section;
            return false;
        }
        r.offset = sec->offset + t.get("offset")->u64();
        p.tensors.emplace(name, std::move(r));
    }
    // the drafter, when the pack carries it
    if (const Json* dj = meta.get("dspark"); dj && dj->is_object()) {
        const Json* arch_j = dj->get("architecture");
        const Json* cj = dj->get("config");
        const Json* tj2 = dj->get("tensors");
        const core::ModelFile::Section* sec = p.file.find("dspark/gguf");
        if (!arch_j || !arch_j->is_string() || arch_j->s != "inkling2-dspark" || !cj || !tj2 || !sec) {
            err = path + ": meta.dspark is not an Inkling 2 DSpark drafter (or its section is missing)";
            return false;
        }
        DsparkConfig& c = p.dspark;
        auto cu = [&](const char* k) { return (int) num_or(cj->get(k), 0); };
        c.n_layers = cu("num_hidden_layers");
        c.n_head = cu("num_attention_heads");
        c.n_kv = cu("num_key_value_heads");
        c.head_dim = cu("head_dim");
        c.n_ff = cu("intermediate_size");
        c.window = cu("sliding_window_size");
        c.block = cu("block_size");
        c.rank = cu("markov_rank");
        c.rope_theta = (float) num_or(cj->get("rope_theta"), 1e6);
        c.rms_eps = (float) num_or(cj->get("rms_norm_eps"), 1e-6);
        c.local.assign((size_t) std::max(c.n_layers, 0), false);
        for (double v : numbers(cj->get("local_layer_ids")))
            if ((int) v >= 0 && (int) v < c.n_layers) c.local[(size_t) v] = true;
        for (double v : numbers(cj->get("target_layer_ids"))) c.target_visits.push_back((int) v);
        const bool any_local = std::find(c.local.begin(), c.local.end(), true) != c.local.end();
        if (c.n_layers <= 0 || c.head_dim != 128 || c.n_kv <= 0 || c.n_head % c.n_kv || c.block < 2 || c.block > 16 || c.rank != 256 ||
            c.target_visits.size() < 2 || (int) c.target_visits.size() > 4 || c.target_visits.back() != g.slots() || (any_local && c.window <= 0)) {
            err = path + ": a DSpark drafter this engine does not run (heads of 128, blocks of 2 to 16, a Markov rank of 256, two to four "
                         "features ending in the output, a window for its local layers)";
            return false;
        }
        for (const auto& [name, t] : tj2->members) {
            TensorRef r;
            r.name = name;
            r.type = (int) num_or(t.get("type_id"), 0);
            if (const Json* sh = t.get("shape"))
                for (const Json& d : sh->items) r.shape.push_back(d.u64());
            r.bytes = (uint64_t) num_or(t.get("bytes"), 0);
            r.offset = sec->offset + t.get("offset")->u64();
            p.dspark_tensors.emplace(name, std::move(r));
        }
        p.has_dspark = true;
    }
    p.experts_range = p.file.range("experts");
    std::vector<uint8_t> text;
    if (!p.file.read("experts.layout", text, err)) return false;
    // v5: `layer gu_type d_type offset blob_bytes`, the layer's experts one after another; v8: `layer expert gu_type d_type
    // offset blob_bytes`, each expert its own
    std::istringstream in(std::string(text.begin(), text.end()));
    std::string line;
    std::getline(in, line);
    const bool per_expert = line.rfind("# experts v8:", 0) == 0;
    if (!per_expert && line.rfind("# experts v5:", 0) != 0) {
        err = "experts.layout: '" + line + "' is not a layout this engine reads (v5 or v8)";
        return false;
    }
    p.experts.assign((size_t) g.n_layers, ExpertLayer{});
    for (ExpertLayer& el : p.experts) el.blobs.assign((size_t) g.n_expert, ExpertBlob{});
    std::vector<int> seen((size_t) g.n_layers, 0);
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        int l, e = 0, gu, d;
        unsigned long long off, by;
        std::string extra;
        if (!(ls >> l) || (per_expert && !(ls >> e)) || !(ls >> gu >> d >> off >> by) || ls >> extra || l < 0 ||
            l >= g.n_layers || e < 0 || e >= g.n_expert) {
            err = "experts.layout: bad line '" + line + "'";
            return false;
        }
        const int count = per_expert ? 1 : g.n_expert;
        if (!by || off > p.experts_range.bytes || by > (p.experts_range.bytes - off) / count) {
            err = "experts.layout: empty or out-of-range expert blob";
            return false;
        }
        auto& blobs = p.experts[(size_t) l].blobs;
        if (blobs[(size_t) e].bytes) {
            err = "experts.layout: repeated row '" + line + "'";
            return false;
        }
        for (int x = 0; x < count; ++x) blobs[(size_t) (e + x)] = {gu, d, off + (uint64_t) x * by, by};
        seen[(size_t) l] += count;
    }
    for (int l = 0; l < g.n_layers; ++l)
        if (seen[(size_t) l] != g.n_expert) {
            err = "experts.layout lists " + std::to_string(seen[(size_t) l]) + " experts of layer " + std::to_string(l) + ", the model has " +
                  std::to_string(g.n_expert);
            return false;
        }
    return true;
}

}  // namespace basalt::models::inkling2x
