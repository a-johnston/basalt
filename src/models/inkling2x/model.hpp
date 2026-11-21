// src/models/inkling2x/model.hpp - the loaded Inkling 2 model.
#pragma once

#include "basalt/model/adapter.hpp"

namespace basalt::models::inkling2x {

class Inkling2Model final : public model::LoadedModel {
public:
    const model::ModelDesc& desc() const override { return desc_; }
    model::Result plan(const model::LoadRequest& req, core::LoadPlan& reads,
                       std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    model::Result attach(const model::LoadRequest& req, std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;

private:
    model::ModelDesc desc_;
};

}  // namespace basalt::models::inkling2x
