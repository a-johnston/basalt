// src/models/inkling2x/kernels/q8_1.cuh - device code shared by Inkling 2's kernel files (no kernels): the q8_1 block the
// quantized matvecs read, written as quantize_q8_1_rows writes it.
#pragma once

#include "basalt/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>

#include <cstdint>

namespace basalt::models::inkling2x::kernels {

// iq_kernels' block_q8_1: fp16 scale and sum, then 32 int8 values
struct Q81 {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81) == 36, "a q8_1 block is 36 bytes");

// quantize_q8_1_kernel's arithmetic for value i of a row of blocks; the warp's 32 lanes hold one block's 32 values
__device__ __forceinline__ void q8_1_value(const float xi, Q81* __restrict__ y, const long long i) {
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = ::basalt::kernels::q8_1_finite(amax / 127.0f);
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = ::basalt::kernels::q8_1_quant(xi, d, amax);
    if (iqs == 0) y[ib].ds = ::basalt::kernels::q8_1_ds(d, sum);
}

}  // namespace basalt::models::inkling2x::kernels
