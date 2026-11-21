#pragma once
#include "basalt/model/adapter.hpp"

namespace basalt::models::inkling2x {
std::unique_ptr<model::ModelAdapter> make_adapter();
void pretok_o200k(const char* text, size_t size, std::vector<size_t>& ends);
}
