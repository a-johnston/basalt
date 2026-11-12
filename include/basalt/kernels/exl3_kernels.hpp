// include/basalt/kernels/exl3_kernels.hpp - EXL3 routed experts on the GPU.  The format and the blob layout are in
// include/basalt/kernels/cpu/exl3_cpu.hpp.
//
// The decode kernels compute  y = (H (inner^T (H (suh * x)))) * svh  per matrix on the tensor cores: the weights are
// the mul1 codebook's fp16 values (the fma ExLlamaV3 computes, exact), and each transformed activation is scaled by a
// power of two per 128-value block and split into two fp16 values, hi + lo, so it keeps about 22 bits.  The products
// are summed in fp32, each block's hi and lo sums apart, then scaled back.  An entry's result does not depend on the
// other entries of the call.
//
// The prompt path's dequantizer writes the matrices in the original basis, W = diag(suh) H inner H diag(svh),
// rounded once to fp16, so nothing downstream of it transforms an activation.
#pragma once

#include "basalt/kernels/iq_kernels.hpp"

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

/// Types 160 + 2 K (cpu::exl3_supported) at widths that are whole 128-value blocks.
bool exl3_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept;
/// The layout of an EXL3 expert blob.
NativeExpertLayout exl3_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
/// Bytes of scratch `exl3_expert_grouped` needs for `cap_entries` entries.
size_t exl3_expert_scratch_bytes(int64_t cap_entries, int64_t n_embd, int64_t n_ff);
/// Bytes of one token's activation as the grouped kernels read it: its n_embd floats.
inline size_t exl3_act_bytes(int64_t n_embd) { return (size_t) n_embd * sizeof(float); }
/// y = x for `n` floats: a token's activation kept for the grouped call.
void exl3_copy_rows(const float* x, float* y, int64_t n, void* stream);

/// native_expert_grouped's contract with `x` the tokens' float rows (n_embd each): group g's blob at grp_ptr[g],
/// its entries [grp_start[g], grp_start[g+1]) read row ent_tok[e] of x and write row ent_dst[e] of `out`.
void exl3_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                         const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                         int64_t cap_entries, const float* x, void* scratch, float* out, void* stream,
                         int64_t grid_groups = 0);

/// One expert blob's gate and up in the original basis as fp16, interleaved as the prompt path reads them: row 2r =
/// gate row r, row 2r+1 = up row r, n_embd values each.
void exl3_dequant_gu_f16(const NativeExpertLayout& L, const void* blob, uint16_t* dst, void* stream);
/// The blob's down matrix in the original basis as fp16: n_embd rows of n_ff.
void exl3_dequant_down_f16(const NativeExpertLayout& L, const void* blob, uint16_t* dst, void* stream);

}  // namespace basalt::kernels
