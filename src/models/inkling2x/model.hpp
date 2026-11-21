// src/models/inkling2x/model.hpp - a loaded Inkling 2 pack: every tensor on the main card, the routed experts of every
// layer in one device buffer.
#pragma once

#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/model/adapter.hpp"
#include "models/inkling2x/pack.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::inkling2x {

struct DevTensor {
    void* ptr = nullptr;
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
    /// Stored layer `l`'s experts: each one's device address and rate pair ([n_expert] each, device), and the layout of each
    /// pair (one for a layer of one type pair).
    const unsigned long long* expert_ptrs(int l) const { return d_ptr_ + (size_t) l * pack_.geo.n_expert; }
    const int32_t* expert_classes(int l) const { return d_cls_ + (size_t) l * pack_.geo.n_expert; }
    const std::vector<::basalt::kernels::NativeExpertLayout>& expert_layouts(int l) const { return classes_[(size_t) l]; }
    /// The most rate pairs a layer has.
    int max_classes() const { return max_classes_; }
    int device() const { return device_; }
    int spec() const { return spec_; }
    const std::string& diag_dump() const { return diag_dump_; }

private:
    Pack pack_;
    model::ModelDesc desc_;
    std::map<std::string, DevTensor> dev_;
    uint8_t* experts_ = nullptr;
    std::vector<std::vector<::basalt::kernels::NativeExpertLayout>> classes_;
    unsigned long long* d_ptr_ = nullptr;
    int32_t* d_cls_ = nullptr;
    int max_classes_ = 1;
    int device_ = 0, spec_ = 0, fd_ = -1;
    int64_t ctx_ = kDefaultContext;
    std::string diag_dump_;
};

/// A session over `model`; null with `err` set.
std::unique_ptr<model::ModelSession> make_session(Inkling2Model& model, std::string& err);

}  // namespace basalt::models::inkling2x
