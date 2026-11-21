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
    /// Stored layer `l`'s experts: the first blob and the layout native_expert_grouped runs them by.
    const uint8_t* experts(int l) const { return experts_ + pack_.experts[(size_t) l].offset; }
    const ::basalt::kernels::NativeExpertLayout& expert_layout(int l) const { return layouts_[(size_t) l]; }
    int device() const { return device_; }
    int spec() const { return spec_; }
    const std::string& diag_dump() const { return diag_dump_; }

private:
    Pack pack_;
    model::ModelDesc desc_;
    std::map<std::string, DevTensor> dev_;
    uint8_t* experts_ = nullptr;
    std::vector<::basalt::kernels::NativeExpertLayout> layouts_;
    int device_ = 0, spec_ = 0, fd_ = -1;
    int64_t ctx_ = kDefaultContext;
    std::string diag_dump_;
};

/// A session over `model`; null with `err` set.
std::unique_ptr<model::ModelSession> make_session(Inkling2Model& model, std::string& err);

}  // namespace basalt::models::inkling2x
