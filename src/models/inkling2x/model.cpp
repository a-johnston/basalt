// src/models/inkling2x/model.cpp - loads an Inkling 2 pack: every tensor to the main card but the token embedding, the routed
// experts of every layer in one device buffer.
#include "models/inkling2x/model.hpp"

#include "basalt/core/loader.hpp"
#include "basalt/kernels/cpu/exl3_cpu.hpp"
#include "basalt/log.hpp"
#include "basalt/models/inkling2x/dspark_kernels.hpp"
#include "basalt/platform/file.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace basalt::models::inkling2x {
namespace {
namespace ik = kernels;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
}

Inkling2Model::~Inkling2Model() {
    for (auto* m : {&dev_, &dsp_})
        for (auto& [name, t] : *m)
            if (t.ptr) t.host ? cudaFreeHost(t.ptr) : cudaFree(t.ptr);
    if (screen_) cudaFree(screen_);
    if (experts_) cudaFree(experts_);
    if (d_ptr_) cudaFree(d_ptr_);
    if (d_rates_) cudaFree(d_rates_);
    if (fd_ >= 0) platform::close_fd(fd_);
}

const DevTensor* Inkling2Model::tensor(const std::string& name) const {
    auto it = dev_.find(name);
    return it == dev_.end() ? nullptr : &it->second;
}

const DevTensor* Inkling2Model::dtensor(const std::string& name) const {
    auto it = dsp_.find(name);
    return it == dsp_.end() ? nullptr : &it->second;
}

model::Result Inkling2Model::plan(const model::LoadRequest& req, core::LoadPlan& reads, std::vector<std::unique_ptr<model::ModelSession>>&) {
    if (req.paths.empty()) return {model::Status::Error, "inkling2x: no model path"};
    std::string err;
    const std::string& path = req.paths.front();
    if (!open_pack(path, pack_, err)) return {model::Status::Error, "inkling2x: " + err};
    cudaGetDevice(&device_);
    const Geometry& g = pack_.geo;
    desc_.arch = "inkling2x";
    desc_.n_layers = g.n_layers;
    desc_.n_embd = g.n_embd;
    desc_.n_vocab = g.n_vocab;
    desc_.n_expert = g.n_expert;
    desc_.n_expert_used = g.n_used;
    desc_.n_ff_exp = g.n_ff_exp;
    desc_.context_length = g.context_length;
    namespace bk = ::basalt::kernels;
    layouts_.resize(g.n_layers);
    mixed_.assign(g.n_layers, false);
    std::vector<bk::Exl3Rates> rates((size_t) g.n_layers * g.n_expert);
    for (int l = 0; l < g.n_layers; ++l) {
        const ExpertLayer& el = pack_.experts[(size_t) l];
        for (int e = 0; e < g.n_expert; ++e) {
            const ExpertBlob& b = el.blobs[(size_t) e];
            if (!bk::native_expert_supported(b.gu_type, b.d_type, g.n_embd, g.n_ff_exp))
                return {model::Status::Error, "inkling2x: layer " + std::to_string(l) + "'s expert types have no kernels"};
            auto lay = bk::native_expert_layout(b.gu_type, b.d_type, g.n_embd, g.n_ff_exp);
            if (e == 0) {
                lay.situ_beta = g.situ_beta;
                lay.situ_up = g.situ_up;
                layouts_[l] = lay;
            } else if (b.gu_type != layouts_[l].gu_type || b.d_type != layouts_[l].d_type) {
                if (!bk::cpu::exl3_supported(b.gu_type) || !bk::cpu::exl3_supported(layouts_[l].gu_type))
                    return {model::Status::Error, "inkling2x: mixed expert rates require EXL3"};
                mixed_[l] = true;
            }
            if (b.bytes != lay.bytes)
                return {model::Status::Error, "inkling2x: layer " + std::to_string(l) + " expert " + std::to_string(e) + "'s blob size does not match its layout"};
            rates[(size_t) l * g.n_expert + e] = {(uint8_t) bk::cpu::exl3_k2(b.gu_type), (uint8_t) bk::cpu::exl3_k2(b.d_type)};
        }
        desc_.expert_gate_up_type.push_back(el.blobs[0].gu_type);
        desc_.expert_down_type.push_back(el.blobs[0].d_type);
    }
    diag_dump_ = req.diag_dump;
    spec_ = req.spec;
    fd_ = platform::open_fd_utf8(path);
    if (fd_ < 0) return {model::Status::Error, "inkling2x: cannot open " + path};
    const int file = reads.file_fd(fd_);
    uint64_t dense = 0;
    uint64_t embed = 0;
    for (const auto& [name, t] : pack_.tensors) {
        DevTensor d;
        d.type = t.type;
        d.bytes = t.bytes;
        d.shape = t.shape;
        core::LoadItem it;
        it.file = file;
        it.off = t.offset;
        it.bytes = t.bytes;
        // the token embedding in mapped host memory: a decode token reads one row (4.6 KB of BF16), a prompt chunk one a token,
        // and the card keeps the 0.4-0.9 GiB it would take
        d.host = name == "token_embd.weight";
        if (d.host) {
            if (cudaHostAlloc(&d.ptr, d.bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                cudaGetLastError();
                return {model::Status::Error, "inkling2x: no page-locked host memory for the token embedding"};
            }
            it.host = (uint8_t*) d.ptr;
            embed += d.bytes;
        } else {
            if (cudaMalloc(&d.ptr, d.bytes) != cudaSuccess) {
                cudaGetLastError();
                return {model::Status::Error, "inkling2x: out of device memory loading " + name};
            }
            it.dev = d.ptr;
            it.device = device_;
            dense += d.bytes;
        }
        reads.add(it);
        dev_.emplace(name, std::move(d));
    }
    // the drafter: on the card but its Markov token table, which a step reads one row of (mapped host memory)
    uint64_t draft_dev = 0, draft_host = 0;
    dsp_on_ = pack_.has_dspark && spec_ >= 2;
    if (dsp_on_)
        for (const auto& [name, t] : pack_.dspark_tensors) {
            DevTensor d;
            d.type = t.type;
            d.bytes = t.bytes;
            d.shape = t.shape;
            core::LoadItem it;
            it.file = file;
            it.off = t.offset;
            it.bytes = t.bytes;
            d.host = name == "dspark.markov_token_embd";
            if (d.host ? cudaHostAlloc(&d.ptr, d.bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess
                       : cudaMalloc(&d.ptr, d.bytes) != cudaSuccess) {
                cudaGetLastError();
                return {model::Status::Error, "inkling2x: out of memory loading the drafter's " + name};
            }
            if (d.host) {
                it.host = (uint8_t*) d.ptr;
                draft_host += d.bytes;
            } else {
                it.dev = d.ptr;
                it.device = device_;
                draft_dev += d.bytes;
            }
            reads.add(it);
            dsp_.emplace(name, std::move(d));
        }
    const uint64_t eb = pack_.experts_range.bytes;
    if (cudaMalloc(&experts_, eb) != cudaSuccess) {
        cudaGetLastError();
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        return {model::Status::Error, "inkling2x: the routed experts take " + std::to_string((int) (eb / (1024 * 1024))) +
                                          " MiB and the card has " + std::to_string((int) (free_b / (1024 * 1024))) +
                                          " MiB free after the other tensors: this engine holds every expert on the card"};
    }
    core::LoadItem it;
    it.file = file;
    it.off = pack_.experts_range.offset;
    it.bytes = eb;
    it.dev = experts_;
    it.device = device_;
    reads.add(it);
    // every expert's device address and rate pair, per layer
    std::vector<unsigned long long> ptr((size_t) g.n_layers * g.n_expert);
    for (int l = 0; l < g.n_layers; ++l)
        for (int e = 0; e < g.n_expert; ++e)
            ptr[(size_t) l * g.n_expert + e] = (unsigned long long) (uintptr_t) (experts_ + pack_.experts[(size_t) l].blobs[(size_t) e].offset);
    const size_t rate_bytes = rates.size() * sizeof(rates[0]);
    const bool mixed = std::find(mixed_.begin(), mixed_.end(), true) != mixed_.end();
    if (cudaMalloc(&d_ptr_, ptr.size() * sizeof(ptr[0])) != cudaSuccess ||
        cudaMemcpy(d_ptr_, ptr.data(), ptr.size() * sizeof(ptr[0]), cudaMemcpyHostToDevice) != cudaSuccess ||
        (mixed && (cudaMalloc(&d_rates_, rate_bytes) != cudaSuccess || cudaMemcpy(d_rates_, rates.data(), rate_bytes, cudaMemcpyHostToDevice) != cudaSuccess))) {
        cudaGetLastError();
        return {model::Status::Error, "inkling2x: out of device memory for the expert tables"};
    }
    ctx_ = req.max_context > 0 ? req.max_context : kDefaultContext;
    if (g.context_length > 0) ctx_ = std::min<int64_t>(ctx_, g.context_length);
    basalt::log::info("basalt inkling2x", "%zu tensors %.2f GiB and the routed experts %.2f GiB on the GPU, the token embedding %.2f GiB in "
                      "host memory; context %lld", dev_.size(), (double) dense / kGiB, (double) eb / kGiB, (double) embed / kGiB, (long long) ctx_);
    if (dsp_on_)
        basalt::log::info("basalt inkling2x", "the DSpark drafter: %zu tensors %.2f GiB on the GPU, its Markov token table %.2f GiB in host memory; "
                          "blocks of %d", dsp_.size(), (double) draft_dev / kGiB, (double) draft_host / kGiB, pack_.dspark.block);
    else if (pack_.has_dspark)
        basalt::log::info("basalt inkling2x", "the pack's DSpark drafter is off (--spec %d)", spec_);
    return {};
}

model::Result Inkling2Model::attach(const model::LoadRequest& req, std::vector<std::unique_ptr<model::ModelSession>>& sessions) {
    if (dsp_on_) {   // the feature mixes' weights: softmax over the features, per row
        const DevTensor* f = dtensor("dspark.fuse");
        const int nf = (int) pack_.dspark.target_visits.size(), rows = pack_.dspark.n_layers + 1;
        if (!f || f->bytes != (uint64_t) rows * nf * 4) return {model::Status::Error, "inkling2x: the drafter's dspark.fuse is not [features, layers + 1]"};
        std::vector<float> w((size_t) rows * nf);
        if (cudaMemcpy(w.data(), f->ptr, f->bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return {model::Status::Error, "inkling2x: reading dspark.fuse"};
        fuse_alpha_.assign(w.size(), 0.0f);
        for (int r = 0; r < rows; ++r) {
            float mx = -INFINITY, sum = 0.0f;
            for (int j = 0; j < nf; ++j) mx = std::max(mx, w[(size_t) r * nf + j]);
            for (int j = 0; j < nf; ++j) sum += std::exp(w[(size_t) r * nf + j] - mx);
            for (int j = 0; j < nf; ++j) fuse_alpha_[(size_t) r * nf + j] = std::exp(w[(size_t) r * nf + j] - mx) / sum;
        }
        // the head's 2-bit copy, packed from the head on the card: the draft's base logits read it instead of the head
        const DevTensor* head = tensor("output.weight");
        const int64_t vocab = pack_.geo.n_vocab;
        if (!head || head->type != 8 || head->shape.size() != 2 || head->shape[0] != 2304 || head->shape[1] < (uint64_t) vocab)
            return {model::Status::Error, "inkling2x: the drafter screens a Q8_0 head of rows of 2304, which this pack does not have"};
        const size_t sb = ik::head_screen_bytes(vocab);
        if (cudaMalloc(&screen_, sb) != cudaSuccess) {
            cudaGetLastError();
            screen_ = nullptr;
            return {model::Status::Error, "inkling2x: out of device memory for the drafter's head screen"};
        }
        ik::head_screen_pack(head->ptr, vocab, screen_, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) return {model::Status::Error, "inkling2x: packing the drafter's head screen failed"};
        basalt::log::info("basalt inkling2x", "the drafter's head screen: %.2f GiB", (double) sb / kGiB);
    }
    return make_sessions(*this, std::max(req.sessions, 1), sessions);
}

}  // namespace basalt::models::inkling2x
