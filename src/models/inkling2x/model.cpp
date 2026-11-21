// src/models/inkling2x/model.cpp - loads an Inkling 2 pack: every tensor to the main card, the routed experts of every layer
// in one device buffer.
#include "models/inkling2x/model.hpp"

#include "basalt/core/loader.hpp"
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
    if (d_cls_) cudaFree(d_cls_);
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
    // each layer's experts by rate pair: a layout per pair (one for a uniform layer), each expert its pair's index
    classes_.assign((size_t) g.n_layers, {});
    std::vector<int32_t> cls((size_t) g.n_layers * g.n_expert);
    for (int l = 0; l < g.n_layers; ++l) {
        const ExpertLayer& el = pack_.experts[(size_t) l];
        std::vector<std::pair<int, int>> pairs;
        for (int e = 0; e < g.n_expert; ++e) {
            const ExpertBlob& b = el.blobs[(size_t) e];
            const auto it = std::find(pairs.begin(), pairs.end(), std::make_pair(b.gu_type, b.d_type));
            int c = (int) (it - pairs.begin());
            if (it == pairs.end()) {
                if (!::basalt::kernels::native_expert_supported(b.gu_type, b.d_type, g.n_embd, g.n_ff_exp))
                    return {model::Status::Error, "inkling2x: layer " + std::to_string(l) + "'s experts (gate/up type " + std::to_string(b.gu_type) +
                                                      ", down " + std::to_string(b.d_type) + ") have no kernels"};
                ::basalt::kernels::NativeExpertLayout lay = ::basalt::kernels::native_expert_layout(b.gu_type, b.d_type, g.n_embd, g.n_ff_exp);
                lay.situ_beta = g.situ_beta;
                lay.situ_up = g.situ_up;
                classes_[(size_t) l].push_back(lay);
                pairs.emplace_back(b.gu_type, b.d_type);
            }
            if (b.bytes != classes_[(size_t) l][(size_t) c].bytes)
                return {model::Status::Error, "inkling2x: layer " + std::to_string(l) + " expert " + std::to_string(e) + "'s blob size does not match its layout"};
            cls[(size_t) l * g.n_expert + e] = c;
        }
        max_classes_ = std::max(max_classes_, (int) pairs.size());
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
    if (cudaMalloc((void**) &d_ptr_, ptr.size() * sizeof(ptr[0])) != cudaSuccess || cudaMalloc((void**) &d_cls_, cls.size() * 4) != cudaSuccess ||
        cudaMemcpy(d_ptr_, ptr.data(), ptr.size() * sizeof(ptr[0]), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_cls_, cls.data(), cls.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
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
