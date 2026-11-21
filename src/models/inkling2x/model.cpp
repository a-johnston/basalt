// src/models/inkling2x/model.cpp - loads an Inkling 2 pack: every tensor to the main card, the routed experts of every layer
// in one device buffer.
#include "models/inkling2x/model.hpp"

#include "basalt/core/loader.hpp"
#include "basalt/kernels/cpu/exl3_cpu.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace basalt::models::inkling2x {
namespace {
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
}

Inkling2Model::~Inkling2Model() {
    for (auto& [name, t] : dev_)
        if (t.ptr) cudaFree(t.ptr);
    if (experts_) cudaFree(experts_);
    if (d_ptr_) cudaFree(d_ptr_);
    if (d_rates_) cudaFree(d_rates_);
    if (fd_ >= 0) platform::close_fd(fd_);
}

const DevTensor* Inkling2Model::tensor(const std::string& name) const {
    auto it = dev_.find(name);
    return it == dev_.end() ? nullptr : &it->second;
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
    for (const auto& [name, t] : pack_.tensors) {
        DevTensor d;
        d.type = t.type;
        d.bytes = t.bytes;
        d.shape = t.shape;
        if (cudaMalloc(&d.ptr, d.bytes) != cudaSuccess) {
            cudaGetLastError();
            return {model::Status::Error, "inkling2x: out of device memory loading " + name};
        }
        core::LoadItem it;
        it.file = file;
        it.off = t.offset;
        it.bytes = t.bytes;
        it.dev = d.ptr;
        it.device = device_;
        reads.add(it);
        dense += d.bytes;
        dev_.emplace(name, std::move(d));
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
    basalt::log::info("basalt inkling2x", "%zu tensors %.2f GiB and the routed experts %.2f GiB on the GPU; context %lld", dev_.size(),
                      (double) dense / kGiB, (double) eb / kGiB, (long long) ctx_);
    return {};
}

model::Result Inkling2Model::attach(const model::LoadRequest& req, std::vector<std::unique_ptr<model::ModelSession>>& sessions) {
    std::string err;
    for (int i = 0; i < std::max(req.sessions, 1); ++i) {
        auto s = make_session(*this, err);
        if (!s) return {model::Status::Error, "inkling2x: " + err};
        sessions.push_back(std::move(s));
    }
    return {};
}

}  // namespace basalt::models::inkling2x
