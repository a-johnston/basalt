// src/models/inkling2x/model.cpp - the loaded Inkling 2 model: the pack is read and checked; its sessions come with the
// engine's forward.
#include "models/inkling2x/model.hpp"

namespace basalt::models::inkling2x {

model::Result Inkling2Model::plan(const model::LoadRequest&, core::LoadPlan&, std::vector<std::unique_ptr<model::ModelSession>>&) {
    return {model::Status::NotImplemented, "inkling2x: the engine cannot run this model yet (its forward is not built)"};
}

model::Result Inkling2Model::attach(const model::LoadRequest&, std::vector<std::unique_ptr<model::ModelSession>>&) {
    return {model::Status::NotImplemented, "inkling2x: the engine cannot run this model yet (its forward is not built)"};
}

}  // namespace basalt::models::inkling2x
