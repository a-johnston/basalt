// include/basalt/kernels/cpu/exl3_cpu.hpp - EXL3 routed experts (ExLlamaV3's trellis format, mul1 codebook) on the
// CPU: the scalar reference decoder and the decode-and-dot kernels, on AVX-512, AVX2 or plain C++.
//
// THE FORMAT (tools/convert/exl3.py has it in numpy and is checked against ExLlamaV3).  A matrix of n_in inputs
// and n_out outputs is  W = diag(suh) H inner H diag(svh):  suh / svh fp16 vectors, H the normalized 128-wide
// Sylvester-Hadamard matrix per block of 128, inner a trellis.  So
//
//     y = (H (inner^T (H (suh * x)))) * svh
//
// and each matrix has its own input transform: an activation is transformed per expert and per matrix, inside the
// expert job, never once per token.
//
// inner is stored as 16 x 16 tiles (16 inputs x 16 outputs).  A tile is a tail-biting bit stream of 256 steps read
// as little-endian uint32 words, most significant bit first; step t's state is the 16 bits that end at its last
// bit.  A step is K bits, or at K = KA + 0.5 alternately KA (even t) and KA + 1 (odd t) bits.  The mul1 codebook
// maps a state s to  fp16(fma(1024 + S, 0x1eee, 0xc931)),  S = bytesum(s * 0x83DCD12D mod 2^32): affine in S.
// Step t is input kExl3Perm[t] / 16 and output kExl3Perm[t] % 16 of the tile.
//
// The engine's type id of a matrix is kExl3TypeBase + 2 K.  In an expert blob the tiles are
// output-major: tile (kt, nt) at nt * n_in/16 + kt, so 16 output rows are n_in * K / 8 contiguous bytes each
// ("row bytes").  The blob is [gate | up | down | suh_g svh_g suh_u svh_u suh_d svh_d].
//
// THE ARITHMETIC, identical on every ISA so every path gives the same bits (and so does any token-group size):
//   xr      = H (suh * x), in float (hadamard_rotate)
//   per block b of 128 inputs:  d[b] = max|xr| / 32767,  q = round-to-nearest-even(xr * (32767 / max|xr|)) as int16
//   I[b][r] = sum over the block's inputs k of (S[k][r] - 510) * q[k]                 exact in int32
//   acc[r]  = fma((float) I[b][r], d[b], acc[r])                                      blocks in order
//   inner_r = fma(acc[r], k_inv, off),  off = (1534 k_inv + k_bias) * sum_b d[b] * sum_k q[k]
//   y       = H (inner) * svh, in float
// The weight is the codebook's affine form before its rounding to fp16 (within 2^-10 of the fp16 value).
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels::cpu {

struct NativeFmt;

inline constexpr int kExl3TypeBase = 160;
inline constexpr int kExl3Had = 128;
inline constexpr int kExl3MaxN = 4096;       ///< widest input
inline constexpr int kExl3MaxTokens = 8;     ///< largest token group per call
inline constexpr uint32_t kExl3Mul1 = 0x83DCD12Du;
inline constexpr float kExl3KInv = 0.00676727294921875f;        ///< fp16 0x1eee
inline constexpr float kExl3KBias = -10.3828125f;               ///< fp16 0xc931
inline constexpr float kExl3Offset = -0.0018157958984375f;      ///< 1534 k_inv + k_bias, exact

/// Types kExl3TypeBase + 2 K for K = 1 .. 8 and 1.5 / 2.5 / 3.5.
inline bool exl3_supported(int type) noexcept {
    const int k2 = type - kExl3TypeBase;
    return k2 >= 2 && k2 <= 16 && (k2 % 2 == 0 || k2 <= 7);
}
/// The bitrate in half-bit units (2 K); 0 for any other type.
inline int exl3_k2(int type) noexcept { return exl3_supported(type) ? type - kExl3TypeBase : 0; }
/// Bytes of a matrix's trellis.
inline size_t exl3_matrix_bytes(int type, int64_t n_in, int64_t n_out) noexcept {
    return (size_t) (n_in * n_out * exl3_k2(type) / 16);
}
/// Bytes of an expert blob's six fp16 vectors.
inline size_t exl3_vec_bytes(int64_t n_embd, int64_t n_ff) noexcept { return (size_t) (6 * (n_embd + n_ff)); }

enum class Exl3Isa : int { scalar = 0, avx2 = 1, avx512 = 2 };
Exl3Isa exl3_isa();
bool exl3_isa_ok(Exl3Isa isa);
const char* exl3_isa_name(Exl3Isa isa);

/// Step -> tile element (input * 16 + output).
extern const uint8_t kExl3Perm[256];
/// The bit just past step t's last bit.
int exl3_step_end(int k2, int t) noexcept;
/// The 256 states of one tile (8 * k2 uint16).
void exl3_tile_states(const uint8_t* tile, int k2, uint16_t st[256]);
/// bytesum(state * 0x83DCD12D), 0 .. 1020.
inline int exl3_bytesum(uint32_t state) noexcept {
    const uint32_t x = state * kExl3Mul1;
    return (int) ((x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24));
}
/// The fp16 weight of a state, as ExLlamaV3 computes it.
float exl3_mul1_value(uint32_t state) noexcept;
/// The reference: an output-major trellis decoded to its inner matrix, out[r * n_in + k] (row r = output).
void exl3_dequant_inner(int type, const uint8_t* w, int n_in, int n_out, float* out);

/// One activation ready for a matrix: H (suh * x) as int16 with a scale per 128-value block.
struct Exl3Act {
    /// per 16-input tile, the 16 values in the order the kernels' weight pairs meet them (each value twice)
    alignas(64) int16_t qp[kExl3MaxN / 16][32];
    alignas(64) float d[kExl3MaxN / kExl3Had];
    alignas(64) int32_t qs[kExl3MaxN / kExl3Had];   ///< per block, the sum of its int16 values
    float off;
};
/// qp slot -> the tile input it holds.
extern const uint8_t kExl3Slot[32];
void exl3_prep(Exl3Isa isa, const float* x, const uint8_t* suh_f16, int n, Exl3Act& a);
/// out[t][r] = inner_r . a[t] for rows [r0, r1), both multiples of 16; `w` is the matrix's trellis.
void exl3_rows(Exl3Isa isa, int type, const uint8_t* w, int n_in, const Exl3Act* const* a, int nt, float* const* out,
               int r0, int r1);
/// In place over rows [r0, r1), both multiples of 128: v = H (v) * svh.
void exl3_finish(Exl3Isa isa, float* v, const uint8_t* svh_f16, int r0, int r1);

/// Gate, up and SwiGLU of one expert blob: ff[t][r] = silu(g) * u over the 128-row blocks that start in [r0, r1);
/// x[t] is the token's n_embd floats.
void exl3_gu_rows(Exl3Isa isa, const NativeFmt& f, const uint8_t* blob, const float* const* x, int nt,
                  float* const* ff, int r0, int r1);
/// out[t][r] = (down h[t])_r over the 128-row blocks that start in [r0, r1); h[t] is the token's n_ff floats.
void exl3_down_rows(Exl3Isa isa, const NativeFmt& f, const uint8_t* blob, const float* const* h, int nt,
                    float* const* out, int r0, int r1);

namespace exl3_detail {
/// How to read the 16 states of steps 16 c .. 16 c + 15 from a tile copy that starts with the tile's last word:
/// state i = ((word[base + hi[i]] >> shr[i]) | (word[base + hi[i] - 1] << (32 - shr[i]))) & 0xffff.
struct Chunk {
    alignas(64) uint32_t hi[16];
    alignas(64) uint32_t shr[16];
    /// The same 16 states two to a 64-bit lane: dword 2 q gets word[base + pair[2 q]], dword 2 q + 1 the word before
    /// it, and bytes 0, 1 (4, 5) of the lane's control pick the even (odd) step's state out of that pair of words.
    alignas(64) uint32_t pair[16];
    alignas(64) uint8_t ctrl[64];
    int base;
};
struct Tables {
    Chunk c[16];
    int words;   ///< uint32 per tile
};
const Tables& tables(int k2);
void rows_scalar(int k2, const uint8_t* w, int n_in, const Exl3Act* const* a, int nt, float* const* out, int r0, int r1);
void rows_avx2(int k2, const uint8_t* w, int n_in, const Exl3Act* const* a, int nt, float* const* out, int r0, int r1);
void rows_avx512(int k2, const uint8_t* w, int n_in, const Exl3Act* const* a, int nt, float* const* out, int r0, int r1);
// the same values as the plain C++ in exl3_cpu.cpp
void prep_avx512(const float* x, const uint8_t* suh_f16, int n, Exl3Act& a);
void finish_avx512(float* v, const uint8_t* svh_f16, int r0, int r1);
void prep_avx2(const float* x, const uint8_t* suh_f16, int n, Exl3Act& a);
void finish_avx2(float* v, const uint8_t* svh_f16, int r0, int r1);
}  // namespace exl3_detail

}  // namespace basalt::kernels::cpu
