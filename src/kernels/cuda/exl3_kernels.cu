// src/kernels/cuda/exl3_kernels.cu - see include/basalt/kernels/exl3_kernels.hpp.
//
// The trellis layout, the mul1 codebook and the tile order are ExLlamaV3's (exllamav3_ext/quant, commit
// bd1f519172406a9560bd5c90e61000691bb69a07; MIT, Copyright (c) 2025 Turboderp, see THIRD_PARTY_LICENSES).  The
// kernels are Basalt's own.
#include "basalt/kernels/exl3_kernels.hpp"
#include "basalt/kernels/cpu/exl3_cpu.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/log.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>

namespace basalt::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e)); std::exit(1); }
}

constexpr int kHad = cpu::kExl3Had;

// the bit just past step t's last bit (cpu::exl3_step_end)
__host__ __device__ constexpr int step_end(int k2, int t) {
    return k2 % 2 == 0 ? (t + 1) * (k2 / 2) : ((t + 1) / 2) * k2 + (t % 2 == 0 ? k2 / 2 : 0);
}

// The codebook value of a state: fp16(fma(1024 + S, k_inv, k_bias)), S the byte sum of state * 0x83DCD12D; 0x6400
// is 1024 as fp16 and its mantissa step there is 1.
__device__ __forceinline__ float mul1(uint32_t state) {
    const uint32_t s = __dp4a(state * cpu::kExl3Mul1, 0x01010101u, 0x6400u);
    return __half2float(__hfma(__ushort_as_half((unsigned short) s), __ushort_as_half((unsigned short) 0x1eee),
                               __ushort_as_half((unsigned short) 0xc931)));
}

// The 64 stream bits that end at bit `e` of a tail-biting tile of 4 K2 words, most significant bit first.
template<int K2>
__device__ __forceinline__ uint64_t window(const uint32_t* __restrict__ tile, int e) {
    constexpr int W = 4 * K2;
    const int hi = (e - 1) >> 5, s = ((hi + 1) << 5) - e;
    const int m1 = hi == 0 ? W - 1 : hi - 1, m2 = m1 == 0 ? W - 1 : m1 - 1;
    const uint64_t w = ((uint64_t) __ldg(tile + m1) << 32) | __ldg(tile + hi);
    return (w >> s) | (((uint64_t) __ldg(tile + m2) << 32) << (32 - s));
}

// The same window's last 32 bits.
template<int K2>
__device__ __forceinline__ uint32_t window32(const uint32_t* __restrict__ tile, int e) {
    constexpr int W = 4 * K2;
    const int hi = (e - 1) >> 5, s = ((hi + 1) << 5) - e;
    return __funnelshift_r(__ldg(tile + hi), __ldg(tile + (hi == 0 ? W - 1 : hi - 1)), s);
}

// The states of steps 8 lane .. 8 lane + 7.  Step 8 lane is even, so the steps' ends relative to the first are the
// same for every lane.  Eight states span 16 bits more than seven steps: one 32-bit window up to 2 bits per weight,
// two up to 5, two 64-bit windows above.
template<int K2>
__device__ __forceinline__ void states8(const uint32_t* __restrict__ tile, int lane, uint32_t (&st)[8]) {
    const int e0 = 4 * lane * K2;   // the bit before step 8 lane
    constexpr int kEnd[8] = {step_end(K2, 0), step_end(K2, 1), step_end(K2, 2), step_end(K2, 3),
                             step_end(K2, 4), step_end(K2, 5), step_end(K2, 6), step_end(K2, 7)};
    if constexpr (kEnd[7] - kEnd[0] + 16 <= 32) {
        const uint32_t v = window32<K2>(tile, e0 + kEnd[7]);
#pragma unroll
        for (int i = 0; i < 8; ++i) st[i] = (v >> (kEnd[7] - kEnd[i])) & 0xffffu;
    } else if constexpr (kEnd[3] - kEnd[0] + 16 <= 32 && kEnd[7] - kEnd[4] + 16 <= 32) {
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const uint32_t v = window32<K2>(tile, e0 + kEnd[4 * g + 3]);
#pragma unroll
            for (int i = 0; i < 4; ++i) st[4 * g + i] = (v >> (kEnd[4 * g + 3] - kEnd[4 * g + i])) & 0xffffu;
        }
    } else {
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const uint64_t v = window<K2>(tile, e0 + kEnd[4 * g + 3]);
#pragma unroll
            for (int i = 0; i < 4; ++i) st[4 * g + i] = (uint32_t) (v >> (kEnd[4 * g + 3] - kEnd[4 * g + i])) & 0xffffu;
        }
    }
}

// states8's reads, split in two so a tile's words can be loaded ahead of the arithmetic: the tile words that
// hold lane `lane`'s eight states, and the shifts that take them out (the same windows as states8).
template<int K2>
struct LaneWin {
    static constexpr int W = 4 * K2;
    static constexpr int kind = step_end(K2, 7) - step_end(K2, 0) + 16 <= 32 ? 1
                              : (step_end(K2, 3) - step_end(K2, 0) + 16 <= 32 && step_end(K2, 7) - step_end(K2, 4) + 16 <= 32) ? 2 : 3;
    static constexpr int NW = kind == 1 ? 2 : kind == 2 ? 4 : 6;   // words a lane reads per tile
    int idx[NW];
    int sh[2];
    __device__ __forceinline__ explicit LaneWin(int lane) {
        const int e0 = 4 * lane * K2;
#pragma unroll
        for (int g = 0; g < (kind == 1 ? 1 : 2); ++g) {
            const int e = e0 + (kind == 1 ? step_end(K2, 7) : step_end(K2, 4 * g + 3));
            const int hi = (e - 1) >> 5, m1 = hi == 0 ? W - 1 : hi - 1;
            sh[g] = ((hi + 1) << 5) - e;
            if constexpr (kind == 3) {
                idx[3 * g] = hi;
                idx[3 * g + 1] = m1;
                idx[3 * g + 2] = m1 == 0 ? W - 1 : m1 - 1;
            } else {
                idx[2 * g] = hi;
                idx[2 * g + 1] = m1;
            }
        }
    }
    // the 16 bits at offset o of v, zero-extended: a byte permute after a shift by o % 8 (states that share the
    // residue share the shift)
    static __device__ __forceinline__ uint32_t field16(uint32_t v, int o) {
        return __byte_perm(v >> (o & 7), 0u, (o >> 3) == 0 ? 0x4410 : (o >> 3) == 1 ? 0x4421 : 0x4432);
    }
    __device__ __forceinline__ void states(const uint32_t (&w)[NW], uint32_t (&st)[8]) const {
        if constexpr (kind == 1) {
            const uint32_t v = __funnelshift_r(w[0], w[1], sh[0]);
#pragma unroll
            for (int i = 0; i < 8; ++i) st[i] = field16(v, step_end(K2, 7) - step_end(K2, i));
        } else if constexpr (kind == 2) {
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                const uint32_t v = __funnelshift_r(w[2 * g], w[2 * g + 1], sh[g]);
#pragma unroll
                for (int i = 0; i < 4; ++i) st[4 * g + i] = field16(v, step_end(K2, 4 * g + 3) - step_end(K2, 4 * g + i));
            }
        } else {
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                const uint64_t x = ((uint64_t) w[3 * g + 1] << 32) | w[3 * g];
                const uint64_t v = (x >> sh[g]) | (((uint64_t) w[3 * g + 2] << 32) << (32 - sh[g]));
#pragma unroll
                for (int i = 0; i < 4; ++i) st[4 * g + i] = (uint32_t) (v >> (step_end(K2, 4 * g + 3) - step_end(K2, 4 * g + i))) & 0xffffu;
            }
        }
    }
};

// In place over a block of 128 threads: the normalized Sylvester-Hadamard transform of the block's values, the
// butterflies of cpu::hadamard_rotate.
__device__ __forceinline__ float fwht128(float v, float* sm) {
    const int i = threadIdx.x;
#pragma unroll
    for (int len = 1; len < kHad; len <<= 1) {
        sm[i] = v;
        __syncthreads();
        const float o = sm[i ^ len];
        v = (i & len) ? o - v : v + o;
        __syncthreads();
    }
    return v * 0.08838834764831845f;
}

__global__ void exl3_copy_kernel(const float* __restrict__ x, float* __restrict__ y, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}

// The group whose entries [grp_start[g], grp_start[g + 1]) hold e, or -1.
__device__ __forceinline__ int group_of(const int32_t* __restrict__ grp_start, int ng, int e) {
    if (ng <= 0 || e < grp_start[0] || e >= grp_start[ng]) return -1;
    int lo = 0, hi = ng - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (grp_start[mid] <= e) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

// An activation as the tensor cores read it: per 128-value block a power-of-two scale s (the block's largest
// magnitude to [2^13, 2^14)), and v s = hi + lo as two fp16 values, so hi + lo carries about 22 bits of v.  Each
// 16-value tile is stored in mma B-fragment order: position 4 t + j holds input 2 t + {0, 1, 8, 9}[j].
struct ActRef {
    __half* hi;
    __half* lo;
    float* inv;   // 1 / s per block
};

__device__ __forceinline__ int frag_pos(int i) {   // the stored position of input i of a block
    const int r = i & 15, t = (r & 7) >> 1, j = (r & 1) + ((r >> 3) << 1);
    return (i & ~15) + 4 * t + j;
}

// Over a block of 128 threads: thread i's value v of block `blk` of row `row` (n values a row) into the split form.
__device__ __forceinline__ void split128(float v, float* smax, const ActRef& a, size_t row, int n, int blk) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float m = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    __syncthreads();   // smax may still be read by an earlier call
    if (lane == 0) smax[warp] = m;
    __syncthreads();
    const float amax = fmaxf(fmaxf(smax[0], smax[1]), fmaxf(smax[2], smax[3]));
    // amax = f 2^ex, f in [0.5, 1); the scales are powers of two built from the exponent bits (amax is normal or 0)
    const int ex = amax > 0.f ? (int) ((__float_as_uint(amax) >> 23) & 0xff) - 126 : 0;
    const float s = amax > 0.f ? __uint_as_float((uint32_t) (127 + 14 - ex) << 23) : 1.f;
    const float x = v * s;
    const __half h = __float2half_rn(x);
    const size_t p = row * (size_t) n + (size_t) blk * kHad + frag_pos(threadIdx.x);
    a.hi[p] = h;
    a.lo[p] = __float2half_rn(x - __half2float(h));
    if (threadIdx.x == 0) a.inv[row * (size_t) (n / kHad) + blk] = amax > 0.f ? __uint_as_float((uint32_t) (127 + ex - 14) << 23) : 0.f;
}

// Each of the three trellises needs unit bytes per half bit; the scale vectors follow them.
struct BlobVectors {
    const Exl3Rates* rates;
    size_t unit, uniform_offset;
    __device__ size_t offset(int g) const {
        if (!rates) return uniform_offset;
        const Exl3Rates r = rates[g];
        return unit * (2 * r.gu + r.down);
    }
};

// x[m][e] = H (suh_m * x[tok[e]]) for the gate (m = 0) and the up (m = 1) matrix of entry e's expert, split.
// Grid (2 n_embd / 128, cap), 128 threads.
__global__ void __launch_bounds__(kHad) exl3_in_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                       const int32_t* __restrict__ grp_start,
                                                       const int32_t* __restrict__ n_groups,
                                                       const int32_t* __restrict__ ent_tok, const float* __restrict__ x,
                                                       ActRef xa, BlobVectors vecs, int n_embd, int n_ff, long long cap) {
    __shared__ float sm[kHad];
    __shared__ float smax[4];
    pdl_launch_dependents();
    const int e = blockIdx.y, g = group_of(grp_start, *n_groups, e);
    if (g < 0) return;
    const int nb = n_embd / kHad, m = blockIdx.x / nb, b = blockIdx.x % nb, i = b * kHad + threadIdx.x;
    const __half* suh = (const __half*) (grp_ptr[g] + vecs.offset(g)) + (m ? n_embd + n_ff : 0);
    const float v = fwht128(x[(size_t) ent_tok[e] * n_embd + i] * __half2float(suh[i]), sm);
    split128(v, smax, xa, (size_t) m * cap + e, n_embd, b);
}

// h[e] = H (suh_d * silu(g) * u), split,  g = H (gate inner) * svh_g,  u = H (up inner) * svh_u.  With beta > 0 the
// gated unit is situ instead (NativeExpertLayout::situ_beta): beta tanh(g / beta) sigmoid(g) up_beta tanh(u / up_beta), in
// native_expert_grouped's order of operations.  Grid (n_ff / 128, cap).
__global__ void __launch_bounds__(kHad) exl3_mid_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                        const int32_t* __restrict__ grp_start,
                                                        const int32_t* __restrict__ n_groups, const float* __restrict__ gu,
                                                        ActRef ha, BlobVectors vecs, int n_embd, int n_ff, long long cap,
                                                        float beta, float up_beta) {
    __shared__ float sm[kHad];
    __shared__ float smax[4];
    pdl_launch_dependents();
    pdl_wait();
    const int e = blockIdx.y, g = group_of(grp_start, *n_groups, e);
    if (g < 0) return;
    const int i = blockIdx.x * kHad + threadIdx.x;
    const __half* vec = (const __half*) (grp_ptr[g] + vecs.offset(g));
    const float gv = fwht128(gu[(size_t) e * n_ff + i], sm) * __half2float(vec[n_embd + i]);
    const float uv = fwht128(gu[((size_t) cap + e) * n_ff + i], sm) * __half2float(vec[2 * n_embd + n_ff + i]);
    float h;
    if (beta > 0.f) {
        const float a = __fmul_rn(beta, tanhf(__fmul_rn(gv, 1.f / beta)));
        const float u = up_beta > 0.f ? __fmul_rn(up_beta, tanhf(__fmul_rn(uv, 1.f / up_beta))) : uv;
        h = __fmul_rn(__fmul_rn(a, 1.f / (1.f + expf(-gv))), u);
    } else {
        h = (gv / (1.f + expf(-gv))) * uv;
    }
    split128(fwht128(h * __half2float(vec[2 * (n_embd + n_ff) + i]), sm), smax, ha, (size_t) e, n_ff, blockIdx.x);
}

// out[dst[e]] = H (down inner) * svh_d.  Grid (n_embd / 128, cap).
__global__ void __launch_bounds__(kHad) exl3_out_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                        const int32_t* __restrict__ grp_start,
                                                        const int32_t* __restrict__ n_groups,
                                                        const int32_t* __restrict__ ent_dst, const float* __restrict__ dn,
                                                        float* __restrict__ out, BlobVectors vecs, int n_embd, int n_ff) {
    __shared__ float sm[kHad];
    pdl_wait();
    const int e = blockIdx.y, g = group_of(grp_start, *n_groups, e);
    if (g < 0) return;
    const int i = blockIdx.x * kHad + threadIdx.x;
    const __half* svh = (const __half*) (grp_ptr[g] + vecs.offset(g)) + 2 * (n_embd + n_ff) + n_ff;
    out[(size_t) ent_dst[e] * n_embd + i] = fwht128(dn[(size_t) e * n_embd + i], sm) * __half2float(svh[i]);
}

struct RowsArgs {
    const unsigned long long* grp_ptr;
    const int32_t* grp_start;
    const int32_t* n_groups;
    ActRef x;            // [n_mat][cap][n_in] split: the entries' transformed inputs
    float* out;          // [n_mat][cap][n_out]: their inner products
    size_t mat_off;      // the first matrix in the blob
    size_t mat_stride;   // bytes from one matrix to the next
    long long cap;
    int n_mat, n_in, n_out;
    const Exl3Rates* rates;
};

// d += A B on the tensor cores: A 16 x 16 fp16 (four registers of the A fragment), B 16 x 8 fp16, d 16 x 8 fp32.
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// One tile's A fragment from lane l's eight states: lane l's eight steps are outputs l / 4 (steps 0..3) and l / 4 + 8 (steps 4..7) against
// inputs 2 (l % 4) + {0, 1, 8, 9}, the fragment's rows g, g + 8 and columns 2 t, 2 t + 1, 2 t + 8, 2 t + 9.  A
// weight is the mul1 codebook's fp16 value, two to an hfma2.
__device__ __forceinline__ void tile_frag(const uint32_t (&st)[8], uint32_t (&a)[4]) {
    uint32_t s[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) s[j] = __dp4a(st[j] * cpu::kExl3Mul1, 0x01010101u, 0x6400u);
    const __half2 k_inv = __halves2half2(__ushort_as_half((unsigned short) 0x1eee), __ushort_as_half((unsigned short) 0x1eee));
    const __half2 k_bias = __halves2half2(__ushort_as_half((unsigned short) 0xc931), __ushort_as_half((unsigned short) 0xc931));
    constexpr int kStep[4] = {0, 4, 2, 6};   // the first of the two steps in A register r
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const uint32_t p = __byte_perm(s[kStep[r]], s[kStep[r] + 1], 0x5410);
        const __half2 w = __hfma2(*(const __half2*) &p, k_inv, k_bias);
        a[r] = *(const uint32_t*) &w;
    }
}

// One warp, blocks [b0, b1) of 128 inputs of one band of 16 outputs, entries row0 .. row0 + nc - 1 (nc <= 8) of
// one expert: a block's tile words and activations are loaded first, then the tensor cores sum its hi and lo parts,
// and each block's hi and lo sums are scaled into running sums, added at the end.  Up to four entries (NARROW) share
// one mma, columns 0..3 their hi parts and 4..7 their lo parts; five to eight take an mma for the hi parts and one
// for the lo parts, with the same sums in the same order, so an entry's result does not depend on the others.  A
// column past the entries reads the last entry's row and is never written.  The warp's sums go to
// part[entry][output].
template<int K2, bool NARROW>
__device__ __forceinline__ void band_rows(const uint32_t* __restrict__ tiles, int b0, int b1, const ActRef& x,
                                          size_t row0, int nc, int n_in, float (*part)[16], int lane) {
    constexpr int W = 4 * K2;
    using LW = LaneWin<K2>;
    const LW lw(lane);
    const int gid = lane >> 2, tig = lane & 3, nb = n_in / kHad;
    // the entry of B column gid, and of C columns 2 tig, 2 tig + 1
    const int eb = NARROW ? (gid & 3) : gid, ec0 = NARROW ? (2 * tig) & 3 : 2 * tig, ec1 = ec0 + 1;
    const size_t xrow = (row0 + min(eb, nc - 1)) * (size_t) n_in;
    const uint2* __restrict__ xh = (const uint2*) ((NARROW && gid >= 4 ? x.lo : x.hi) + xrow) + tig;
    const uint2* __restrict__ xl = (const uint2*) (x.lo + xrow) + tig;
    const float* __restrict__ inv0 = x.inv + (row0 + min(ec0, nc - 1)) * (size_t) nb;
    const float* __restrict__ inv1 = x.inv + (row0 + min(ec1, nc - 1)) * (size_t) nb;
    float rh[4] = {0.f, 0.f, 0.f, 0.f}, rl[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
    for (int b = b0; b < b1; ++b) {
        const uint32_t* __restrict__ tb = tiles + (size_t) (8 * b) * W;
        uint32_t w[8][LW::NW];
        uint2 h[8], l[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < LW::NW; ++j) w[i][j] = __ldg(tb + i * W + lw.idx[j]);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            h[i] = __ldg(xh + 32 * b + 4 * i);
            if constexpr (!NARROW) l[i] = __ldg(xl + 32 * b + 4 * i);
        }
        const float s0 = __ldg(inv0 + b), s1 = __ldg(inv1 + b);
        float dh[4] = {0.f, 0.f, 0.f, 0.f}, dl[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t st[8], a[4];
            lw.states(w[i], st);
            tile_frag(st, a);
            mma16816(dh, a, h[i].x, h[i].y);
            if constexpr (!NARROW) mma16816(dl, a, l[i].x, l[i].y);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            rh[j] = fmaf(dh[j], j & 1 ? s1 : s0, rh[j]);
            if constexpr (!NARROW) rl[j] = fmaf(dl[j], j & 1 ? s1 : s0, rl[j]);
        }
    }
    float run[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)   // NARROW: columns 4..7 (lanes with tig 2, 3) are the lo parts of columns 0..3
        run[j] = rh[j] + (NARROW ? __shfl_xor_sync(0xffffffffu, rh[j], 2) : rl[j]);
    if (NARROW && tig >= 2) return;
    if (ec0 < nc) { part[ec0][gid] = run[0]; part[ec0][gid + 8] = run[2]; }
    if (ec1 < nc) { part[ec1][gid] = run[1]; part[ec1][gid + 8] = run[3]; }
}

// Up to 8 entries of one expert on one band: each of the block's NWARP warps takes a run of the band's 128-input
// blocks, and their sums are added in warp order.
template<int K2, int NWARP>
__device__ __forceinline__ void band_entries(const uint32_t* __restrict__ tiles, const ActRef& x, size_t row0, int nc,
                                             int n_in, float* __restrict__ out, int n_out, int band,
                                             float (*part)[8][16]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nb = n_in / kHad;
    if (nc <= 4) band_rows<K2, true>(tiles, warp * nb / NWARP, (warp + 1) * nb / NWARP, x, row0, nc, n_in, part[warp], lane);
    else band_rows<K2, false>(tiles, warp * nb / NWARP, (warp + 1) * nb / NWARP, x, row0, nc, n_in, part[warp], lane);
    __syncthreads();
    if (threadIdx.x < 16 * nc) {
        const int c = threadIdx.x >> 4, o = threadIdx.x & 15;
        float v = part[0][c][o];
#pragma unroll
        for (int w = 1; w < NWARP; ++w) v += part[w][c][o];
        out[(size_t) c * n_out + 16 * band + o] = v;
    }
    __syncthreads();
}

#define EXL3_RATES(X) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(10) X(12) X(14) X(16)

// K2 = 0: one launch for all rates, with a group-uniform branch around the same band computation.
template<int K2>
__device__ const uint32_t* band_tiles(const RowsArgs& a, int g, int m, int band, int& rate) {
    size_t off;
    if constexpr (K2 == 0) {
        const Exl3Rates r = a.rates[g];
        const size_t gu_bytes = (size_t) a.n_in * a.n_out / 16 * r.gu;
        off = (a.n_mat == 2 ? m : 2) * gu_bytes;
        rate = a.n_mat == 2 ? r.gu : r.down;
    } else {
        off = a.mat_off + m * a.mat_stride;
        rate = K2;
    }
    return (const uint32_t*) (a.grp_ptr[g] + off) + (size_t) band * (a.n_in / 16) * (4 * rate);
}

template<int K2, int NWARP>
__device__ void group_band(const uint32_t* tiles, const ActRef& x, int e0, int e1, int n_in, float* out, int n_out,
                           int band, float (*part)[8][16]) {
    for (int e = e0; e < e1; e += 8)
        band_entries<K2, NWARP>(tiles, x, (size_t) e, min(8, e1 - e), n_in, out + (size_t) e * n_out, n_out, band, part);
}

// Grid (bands of every matrix, groups side by side), 32 NWARP threads: a block per band; a block row strides over
// the groups past the grid's.
template<int K2, int NWARP>
__device__ __forceinline__ void rows_body(const RowsArgs& a) {
    __shared__ float part[NWARP][8][16];
    const int bands = a.n_out / 16, m = blockIdx.x / bands, band = blockIdx.x % bands, kt_n = a.n_in / 16;
    const ActRef xm{a.x.hi + (size_t) m * a.cap * a.n_in, a.x.lo + (size_t) m * a.cap * a.n_in,
                    a.x.inv + (size_t) m * a.cap * (a.n_in / kHad)};
    float* om = a.out + (size_t) m * a.cap * a.n_out;
    const int ng = *a.n_groups;
    pdl_launch_dependents();
    if ((int) blockIdx.y < ng) {   // the plan and the weights were written before this call: the first band into L2
        int rate;
        const char* t0 = (const char*) band_tiles<K2>(a, blockIdx.y, m, band, rate);
        for (int o = 128 * threadIdx.x; o < kt_n * rate * 16; o += 128 * blockDim.x)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(t0 + o));
    }
    pdl_wait();
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        int rate;
        const uint32_t* tiles = band_tiles<K2>(a, g, m, band, rate);
        const int e0 = a.grp_start[g], e1 = a.grp_start[g + 1];
        if constexpr (K2 != 0) group_band<K2, NWARP>(tiles, xm, e0, e1, a.n_in, om, a.n_out, band, part);
        else switch (rate) {
#define X(K) case K: group_band<K, NWARP>(tiles, xm, e0, e1, a.n_in, om, a.n_out, band, part); break;
            EXL3_RATES(X)
#undef X
        }
    }
}

template<int K2, int NWARP>
__global__ void __launch_bounds__(32 * NWARP) exl3_rows_kernel(RowsArgs a) { rows_body<K2, NWARP>(a); }

// The all-rates kernel, held to four blocks an SM: its switch over every rate otherwise takes 250 registers (one block
// an SM), where a single rate takes 64 to 80. At four it takes 64 and spills 388 bytes, and is still faster than three
// (80 registers, 48 bytes spilled).
template<int NWARP>
__global__ void __launch_bounds__(32 * NWARP, 4) exl3_rows_all_kernel(RowsArgs a) { rows_body<0, NWARP>(a); }

// One 128 x 128 block (inputs kb, outputs nb) of a matrix in the original basis, a half of 64 outputs at a time in
// shared memory, the Hadamard butterflies along the inputs and then along the outputs in the stage order of
// cpu::hadamard_rotate.  A half takes four passes: its 32 tiles decoded with the inputs' stage of 1; the inputs'
// stages 2 to 16; the inputs' stages 32 and 64 with the outputs' stages 1 to 4; the outputs' stages 8 to 32, whose
// results stay in registers.  The outputs' stage of 64 pairs the two halves, then fp16(suh * svh * value / 128) goes
// to dst row (r * row_mul + matrix).  Grid (n_in / 128, n_out / 128, matrices), kDqT threads.  A row of the half is
// 129 floats apart, so a warp's lanes, which take neighbouring rows or neighbouring inputs, fall in different banks.
constexpr int kDqRow = kHad + 1;
constexpr int kDqHalf = kHad / 2;   // outputs a half
constexpr int kDqT = 512;           // threads a block: one group of 16 or two of 8 each

struct DqArgs {
    const uint32_t* w[2];   // the matrices' trellises
    const __half* suh[2];
    const __half* svh[2];
    uint16_t* dst;
    int n_in, row_mul;
};

// the butterflies of stages 1 .. n / 2 over v[0], v[step], .. (n values)
template<int N>
__device__ __forceinline__ void dq_bfly(float* v, int step = 1) {
#pragma unroll
    for (int h = 1; h < N; h <<= 1)
#pragma unroll
        for (int m = 0; m < N; ++m)
            if (!(m & h)) {
                const float a = v[m * step], c = v[(m + h) * step];
                v[m * step] = a + c;
                v[(m + h) * step] = a - c;
            }
}

template<int K2>
__global__ void __launch_bounds__(kDqT) exl3_dequant_kernel(DqArgs q) {
    constexpr int W = 4 * K2;
    const uint32_t* __restrict__ w = q.w[blockIdx.z];
    const __half* __restrict__ suh = q.suh[blockIdx.z];
    const __half* __restrict__ svh = q.svh[blockIdx.z];
    uint16_t* __restrict__ dst = q.dst;
    const int n_in = q.n_in, row_mul = q.row_mul, row_add = blockIdx.z;
    __shared__ float sm[kDqHalf * kDqRow];   // [output of the half][input], rows kDqRow apart
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int kb = blockIdx.x, nb = blockIdx.y, kt_n = n_in / 16;
    float keep[16];
#pragma unroll 1
    for (int half = 0; half < 2; ++half) {
        {   // the half's 32 tiles over the warps; a lane's inputs r0, r0 + 1 and r0 + 8, r0 + 9 are stage 1's pairs
            const int r0 = (lane & 3) * 2;
#pragma unroll 1
            for (int j = warp; j < 32; j += kDqT / 32) {
                const int kt = j & 7, n8 = j >> 3;
                uint32_t st[8];
                states8<K2>(w + ((size_t) (8 * nb + 4 * half + n8) * kt_n + 8 * kb + kt) * W, lane, st);
                float* o = sm + (16 * n8 + (lane >> 2)) * kDqRow + 16 * kt + r0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float a = mul1(st[2 * i]), c = mul1(st[2 * i + 1]);
                    float* e = o + (i >> 1) * 8 * kDqRow + (i & 1) * 8;
                    e[0] = a + c;
                    e[1] = a - c;
                }
            }
        }
        __syncthreads();
        {   // the inputs' stages 2 to 16: a row's 8 groups of 16 values 2 apart
            const int g = t >> 6;
            float* e = sm + (t & 63) * kDqRow + (g & 1) + ((g >> 1) << 5);
            float v[16];
#pragma unroll
            for (int m = 0; m < 16; ++m) v[m] = e[2 * m];
            dq_bfly<16>(v);
#pragma unroll
            for (int m = 0; m < 16; ++m) e[2 * m] = v[m];
        }
        __syncthreads();
        if (t < kDqT / 2) {   // 8 outputs x 4 inputs 32 apart: the inputs' stages 32 and 64, then the outputs' 1 to 4
            float* e = sm + (t >> 5) * 8 * kDqRow + (t & 31);
            float v[32];   // [input][output]
#pragma unroll
            for (int m = 0; m < 32; ++m) v[m] = e[(m & 7) * kDqRow + (m >> 3) * 32];
#pragma unroll
            for (int r = 0; r < 8; ++r) dq_bfly<4>(v + r, 8);
#pragma unroll
            for (int c = 0; c < 4; ++c) dq_bfly<8>(v + 8 * c);
#pragma unroll
            for (int m = 0; m < 32; ++m) e[(m & 7) * kDqRow + (m >> 3) * 32] = v[m];
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 2; ++i) {   // the outputs' stages 8 to 32: an input's 8 groups of 8 outputs 8 apart
            const int k = t & 127, j = (t >> 7) + 4 * i;
            float v[8];
#pragma unroll
            for (int m = 0; m < 8; ++m) v[m] = sm[(8 * m + j) * kDqRow + k];
            dq_bfly<8>(v);
            if (half == 0) {
#pragma unroll
                for (int m = 0; m < 8; ++m) keep[8 * i + m] = v[m];
            } else {   // the outputs' stage of 64, the scales and the rounding
                const int c = kb * kHad + k;
                const float sc = __half2float(suh[c]);
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const int r = nb * kHad + 8 * m + j;
                    const float a = keep[8 * i + m], b = v[m];
                    const float v0 = (a + b) * 0.0078125f * sc * __half2float(svh[r]);
                    const float v1 = (a - b) * 0.0078125f * sc * __half2float(svh[r + kDqHalf]);
                    dst[(size_t) (r * row_mul + row_add) * n_in + c] = __half_as_ushort(__float2half(v0));
                    dst[(size_t) ((r + kDqHalf) * row_mul + row_add) * n_in + c] = __half_as_ushort(__float2half(v1));
                }
            }
        }
        __syncthreads();
    }
}

// a band's warps: five where the input's 128-value blocks split evenly over them, else eight
template<int NWARP>
void launch_rows_w(int k2, dim3 grid, cudaStream_t s, const RowsArgs& a) {
    switch (k2) {
        case 0: launch_pdl(exl3_rows_all_kernel<NWARP>, grid, dim3(32 * NWARP), 0, s, a); break;
#define X(K2) case K2: launch_pdl(exl3_rows_kernel<K2, NWARP>, grid, dim3(32 * NWARP), 0, s, a); break;
        EXL3_RATES(X)
#undef X
        default: basalt::log::error("basalt", "exl3_expert_grouped: bitrate %d/2", k2); std::exit(1);
    }
}

void launch_rows(int k2, dim3 grid, cudaStream_t s, const RowsArgs& a) {
    const int nb = a.n_in / kHad;
    if (nb % 5 == 0) launch_rows_w<5>(k2, grid, s, a);
    else launch_rows_w<8>(k2, grid, s, a);
}

size_t al256(size_t n) { return (n + 255) & ~(size_t) 255; }

// matrices of one type and shape in the original basis: matrix m's row r is dst row (r * row_mul + m), n_in values
void dequant_matrices(int type, const DqArgs& q, int n_mat, int n_out, cudaStream_t s) {
    const dim3 grid((unsigned) (q.n_in / kHad), (unsigned) (n_out / kHad), (unsigned) n_mat);
    switch (cpu::exl3_k2(type)) {
#define X(K2) case K2: exl3_dequant_kernel<K2><<<grid, kDqT, 0, s>>>(q); break;
        EXL3_RATES(X)
#undef X
        default: basalt::log::error("basalt", "exl3 dequantizer: type %d", type); std::exit(1);
    }
    check("exl3 dequantizer");
}

}  // namespace

bool exl3_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    return cpu::exl3_supported(gu_type) && cpu::exl3_supported(d_type) && n_embd > 0 && n_ff > 0 && n_embd % kHad == 0 &&
           n_ff % kHad == 0;
}

NativeExpertLayout exl3_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.up_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = cpu::exl3_matrix_bytes(gu_type, n_embd, 1);
    L.up_row = L.gu_row;
    L.d_row = cpu::exl3_matrix_bytes(d_type, n_ff, 1);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row + cpu::exl3_vec_bytes(n_embd, n_ff);
    return L;
}

// [2 cap][n_embd] of split gate and up inputs, [2 cap][n_ff] of their inner products, [cap][n_ff] of split down
// inputs, [cap][n_embd] of its inner products
size_t exl3_expert_scratch_bytes(int64_t cap, int64_t n_embd, int64_t n_ff) {
    const size_t c = (size_t) cap, f = sizeof(float), h = sizeof(__half);
    auto split = [&](size_t rows, int64_t n) { return 2 * al256(rows * (size_t) n * h) + al256(rows * (size_t) (n / kHad) * f); };
    return split(2 * c, n_embd) + al256(2 * c * (size_t) n_ff * f) + split(c, n_ff) + al256(c * (size_t) n_embd * f);
}

void exl3_copy_rows(const float* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    exl3_copy_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(x, y, (long long) n);
    check("exl3_copy_rows");
}

void exl3_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                         const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                         int64_t cap_entries, const float* x, void* scratch, float* out, void* stream,
                         int64_t grid_groups, const Exl3Rates* rates) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (!exl3_expert_supported(L.gu_type, L.d_type, L.n_embd, L.n_ff)) {
        basalt::log::error("basalt", "exl3_expert_grouped: types %d / %d at %lld x %lld", L.gu_type, L.d_type,
                           (long long) L.n_embd, (long long) L.n_ff);
        std::exit(1);
    }
    cudaStream_t s = (cudaStream_t) stream;
    const int ne = (int) L.n_embd, nf = (int) L.n_ff;
    const size_t c = (size_t) cap_entries, f = sizeof(float);
    uint8_t* p = (uint8_t*) scratch;
    auto split = [&](size_t rows, int n) {
        ActRef r;
        r.hi = (__half*) p;
        p += al256(rows * (size_t) n * sizeof(__half));
        r.lo = (__half*) p;
        p += al256(rows * (size_t) n * sizeof(__half));
        r.inv = (float*) p;
        p += al256(rows * (size_t) (n / kHad) * f);
        return r;
    };
    const ActRef xa = split(2 * c, ne);
    float* gu = (float*) p;
    p += al256(2 * c * (size_t) nf * f);
    const ActRef ha = split(c, nf);
    float* dn = (float*) p;
    const BlobVectors vecs{rates, (size_t) ne * nf / 16, L.bytes - cpu::exl3_vec_bytes(ne, nf)};
    const unsigned gy = (unsigned) ((grid_groups <= 0 || grid_groups > cap_groups) ? cap_groups : grid_groups);
    const unsigned ce = (unsigned) cap_entries;

    exl3_in_kernel<<<dim3((unsigned) (2 * ne / kHad), ce), kHad, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, x, xa, vecs,
                                                                       ne, nf, (long long) cap_entries);
    PdlScope pdl(true);   // the four after the first start early (the plan and the weights predate the call)
    RowsArgs a{grp_ptr, grp_start, n_groups, xa, gu, 0, L.up_off, (long long) cap_entries, 2, ne, nf, rates};
    launch_rows(rates ? 0 : cpu::exl3_k2(L.gu_type), dim3((unsigned) (2 * nf / 16), gy), s, a);
    launch_pdl(exl3_mid_kernel, dim3((unsigned) (nf / kHad), ce), dim3(kHad), 0, s, grp_ptr, grp_start, n_groups,
               (const float*) gu, ha, vecs, ne, nf, (long long) cap_entries, L.situ_beta, L.situ_up);
    RowsArgs d{grp_ptr, grp_start, n_groups, ha, dn, L.down_off, (size_t) 0, (long long) cap_entries, 1, nf, ne, rates};
    launch_rows(rates ? 0 : cpu::exl3_k2(L.d_type), dim3((unsigned) (ne / 16), gy), s, d);
    launch_pdl(exl3_out_kernel, dim3((unsigned) (ne / kHad), ce), dim3(kHad), 0, s, grp_ptr, grp_start, n_groups, ent_dst,
               (const float*) dn, out, vecs, ne, nf);
    check("exl3_expert_grouped");
}

void exl3_dequant_gu_f16(const NativeExpertLayout& L, const void* blob, uint16_t* dst, void* stream) {
    const int ne = (int) L.n_embd, nf = (int) L.n_ff;
    const uint8_t* b = (const uint8_t*) blob;
    const __half* vec = (const __half*) (b + L.bytes - cpu::exl3_vec_bytes(ne, nf));
    const DqArgs q{{(const uint32_t*) b, (const uint32_t*) (b + L.up_off)}, {vec, vec + ne + nf},
                   {vec + ne, vec + 2 * ne + nf}, dst, ne, 2};
    dequant_matrices(L.gu_type, q, 2, nf, (cudaStream_t) stream);
}

void exl3_dequant_down_f16(const NativeExpertLayout& L, const void* blob, uint16_t* dst, void* stream) {
    const int ne = (int) L.n_embd, nf = (int) L.n_ff;
    const uint8_t* b = (const uint8_t*) blob;
    const __half* vec = (const __half*) (b + L.bytes - cpu::exl3_vec_bytes(ne, nf)) + 2 * (ne + nf);
    const DqArgs q{{(const uint32_t*) (b + L.down_off), nullptr}, {vec, nullptr}, {vec + nf, nullptr}, dst, nf, 1};
    dequant_matrices(L.d_type, q, 1, ne, (cudaStream_t) stream);
}

}  // namespace basalt::kernels
