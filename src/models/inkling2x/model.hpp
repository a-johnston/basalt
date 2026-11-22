// src/models/inkling2x/model.hpp - a loaded Inkling 2 pack: every tensor on the main card but the token embedding, which
// stays in mapped host memory (a token reads one row of it), and the routed experts of every layer in one device buffer.
#pragma once

#include "basalt/kernels/exl3_kernels.hpp"
#include "basalt/model/adapter.hpp"
#include "models/inkling2x/pack.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::inkling2x {

struct DevTensor {
    void* ptr = nullptr;   ///< an address the card reads: device memory, or mapped host memory when `host`
    bool host = false;
    int type = 0;
    uint64_t bytes = 0;
    std::vector<uint64_t> shape;
};

constexpr int64_t kDefaultContext = 8192;

class Inkling2Model final : public model::LoadedModel {
public:
    Inkling2Model() = default;
    ~Inkling2Model() override;
    const model::ModelDesc& desc() const override { return desc_; }
    int64_t context() const override { return ctx_; }
    model::Result plan(const model::LoadRequest& req, core::LoadPlan& reads,
                       std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    model::Result attach(const model::LoadRequest& req, std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;

    const Geometry& geo() const { return pack_.geo; }
    /// Device tensor `blk.<layer>.<suffix>` or a global name; null when absent.
    const DevTensor* tensor(const std::string& name) const;
    const DevTensor* tensor(int layer, const char* suffix) const { return tensor("blk." + std::to_string(layer) + "." + suffix); }
    /// Per-expert blob addresses and, for a mixed-rate layer only, the rates (both on device).
    const unsigned long long* expert_ptrs(int l) const { return d_ptr_ + (size_t) l * pack_.geo.n_expert; }
    const ::basalt::kernels::Exl3Rates* expert_rates(int l) const { return mixed_[l] ? d_rates_ + (size_t) l * pack_.geo.n_expert : nullptr; }
    const ::basalt::kernels::NativeExpertLayout& expert_layout(int l) const { return layouts_[l]; }
    int device() const { return device_; }
    int spec() const { return spec_; }
    /// The DSpark drafter, when the pack carries it and the start drafts (spec >= 2); null otherwise.
    const DsparkConfig* dspark() const { return dsp_on_ ? &pack_.dspark : nullptr; }
    /// A drafter tensor (`dspark.*`); null when absent.
    const DevTensor* dtensor(const std::string& name) const;
    const DevTensor* dtensor(int layer, const char* suffix) const { return dtensor("dspark.blk." + std::to_string(layer) + "." + suffix); }
    /// softmax(fuse) per row: [n_layers + 1][features], row n_layers the Markov head's (host).
    const std::vector<float>& fuse_alpha() const { return fuse_alpha_; }
    /// The 2-bit copy of the head the drafter screens its base logits with (dspark_kernels.hpp's head_screen_*); null
    /// without the drafter.
    const void* head_screen() const { return screen_; }
    const std::string& diag_dump() const { return diag_dump_; }

private:
    Pack pack_;
    model::ModelDesc desc_;
    std::map<std::string, DevTensor> dev_, dsp_;
    bool dsp_on_ = false;
    std::vector<float> fuse_alpha_;
    void* screen_ = nullptr;
    uint8_t* experts_ = nullptr;
    std::vector<::basalt::kernels::NativeExpertLayout> layouts_;
    std::vector<bool> mixed_;
    unsigned long long* d_ptr_ = nullptr;
    ::basalt::kernels::Exl3Rates* d_rates_ = nullptr;
    int device_ = 0, spec_ = 0, fd_ = -1;
    int64_t ctx_ = kDefaultContext;
    std::string diag_dump_;
};

/// `n` sessions over `model`, which share the forward's scratch (they run one after another).
model::Result make_sessions(Inkling2Model& model, int n, std::vector<std::unique_ptr<model::ModelSession>>& sessions);

}  // namespace basalt::models::inkling2x
