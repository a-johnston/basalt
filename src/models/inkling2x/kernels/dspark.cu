// src/models/inkling2x/kernels/dspark.cu - see include/basalt/models/inkling2x/dspark_kernels.hpp.
#include "basalt/models/inkling2x/dspark_kernels.hpp"
#include "basalt/kernels/head_proxy.cuh"
#include "common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace basalt::models::inkling2x::kernels {
namespace {

namespace px = ::basalt::kernels::px;

constexpr int kHeadDim = 128;
constexpr int kRowThreads = 512;   // a row kernel's block (a q8_1 block is a warp's 32 values)

__device__ __forceinline__ float ds_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// the block's sum, the warps' sums added in order; every thread gets it
__device__ __forceinline__ float ds_block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = ds_warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += red[i];
    return t;
}

__device__ __forceinline__ float bf16_to_f(uint16_t u) { return __uint_as_float((uint32_t) u << 16); }
__device__ __forceinline__ uint16_t f_to_bf16(float x) { return __bfloat16_as_ushort(__float2bfloat16_rn(x)); }

// ------------------------------------------------------------------------------------------------- features
__global__ void features_store_kernel(const float* __restrict__ x, Members m, FeatureDst dst, int n_feat, int slot, int E) {
    const int r = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= E) return;
    const int i = member_of(m, r);
    dst.p[i][((size_t) (r - m.start[i]) * n_feat + slot) * E + c] = f_to_bf16(x[(size_t) r * E + c]);
}

// Grid (rows, mixes).  Pass one sums the squares, pass two writes; a thread keeps its values (the same columns).
__global__ void __launch_bounds__(kRowThreads) feature_mix_kernel(const uint16_t* __restrict__ feat, int n_feat, Mixes mx, int rows,
                                                                  int E, float eps, float* __restrict__ out, Q81* __restrict__ xq) {
    __shared__ float red[32];
    const int r = blockIdx.x, j = blockIdx.y;
    const uint16_t* f = feat + (size_t) r * n_feat * E;
    auto mixed = [&](int c) {
        float v = 0.0f;
        for (int t = 0; t < n_feat; ++t) v += mx.alpha[j][t] * bf16_to_f(f[(size_t) t * E + c]);
        return v;
    };
    float ss = 0.0f;
    for (int c = threadIdx.x; c < E; c += kRowThreads) {
        const float v = mixed(c);
        ss = fmaf(v, v, ss);
    }
    const float inv = rsqrtf(ds_block_sum(ss, red) / (float) E + eps);
    float* o = out + ((size_t) j * rows + r) * E;
    Q81* q = xq ? xq + ((size_t) j * rows + r) * (E / 32) : nullptr;
    const float* w = mx.w[j];
    for (int c = threadIdx.x; c < E; c += kRowThreads) {
        const float v = mixed(c) * inv * w[c];
        o[c] = v;
        if (q) q8_1_value(v, q, c);
    }
}

// --------------------------------------------------------------------------------- query and key heads
// Grid (rows, Q heads (with q) + 2 K/V heads), 128 threads: a head's RMS norm with its weight, then neox RoPE at the row's
// position (cos and sin of fp32 position x fp32 inverse frequency, as vLLM's float32 cache); V as it is.
__global__ void __launch_bounds__(kHeadDim) qk_norm_rope_kernel(float* __restrict__ q, const float* __restrict__ q_w,
                                                                const float* __restrict__ k, const float* __restrict__ k_w,
                                                                const float* __restrict__ v, QkArgs a, const int64_t* __restrict__ pos,
                                                                __half* __restrict__ kc, __half* __restrict__ vc, long long cells) {
    __shared__ float red[32], xs[kHeadDim];
    const int r = blockIdx.x, d = threadIdx.x, nq = q ? a.n_head : 0;
    int seg = blockIdx.y;
    const long long p = pos[r];
    const size_t cell = (size_t) (p % cells);
    if (seg >= nq + a.n_kv) {   // V
        const int h = seg - nq - a.n_kv;
        vc[(cell * a.n_kv + h) * kHeadDim + d] = __float2half_rn(v[((size_t) r * a.n_kv + h) * kHeadDim + d]);
        return;
    }
    const bool is_q = seg < nq;
    const int h = is_q ? seg : seg - nq;
    const float* src = is_q ? q + ((size_t) r * a.n_head + h) * kHeadDim : k + ((size_t) r * a.n_kv + h) * kHeadDim;
    const float* w = is_q ? q_w : k_w;
    const float x = src[d];
    const float ms = ds_block_sum(x * x, red) / (float) kHeadDim;
    const float y = x * rsqrtf(ms + a.eps) * w[d];
    xs[d] = y;
    __syncthreads();
    const int i = d & 63;
    const float inv_freq = 1.0f / powf(a.theta, (float) (2 * i) / (float) kHeadDim);
    const float ang = (float) p * inv_freq;
    const float cs = cosf(ang), sn = sinf(ang);
    const float out = d < 64 ? y * cs - xs[d + 64] * sn : y * cs + xs[d - 64] * sn;
    if (is_q) q[((size_t) r * a.n_head + h) * kHeadDim + d] = out;
    else kc[(cell * a.n_kv + h) * kHeadDim + d] = __float2half_rn(out);
}

// ------------------------------------------------------------------------------------------ residual norms
__global__ void __launch_bounds__(kRowThreads) add_norm_quant_kernel(float* __restrict__ res, const float* __restrict__ add,
                                                                     const float* __restrict__ w, int E, float eps, float* __restrict__ out,
                                                                     Q81* __restrict__ xq) {
    __shared__ float red[32];
    const int r = blockIdx.x;
    float* x = res + (size_t) r * E;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < E; c += kRowThreads) {
        float v = x[c];
        if (add) x[c] = v = v + add[(size_t) r * E + c];
        ss = fmaf(v, v, ss);
    }
    const float inv = rsqrtf(ds_block_sum(ss, red) / (float) E + eps);
    Q81* q = xq ? xq + (size_t) r * (E / 32) : nullptr;
    for (int c = threadIdx.x; c < E; c += kRowThreads) {
        const float v = x[c] * inv * w[c];
        if (out) out[(size_t) r * E + c] = v;
        if (q) q8_1_value(v, q, c);
    }
}

__global__ void swiglu_quant_kernel(const float* __restrict__ gate, const float* __restrict__ up, int n, Q81* __restrict__ xq) {
    const int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;   // n a multiple of 32: whole warps
    const float g = gate[(size_t) r * n + c];
    const float h = (g / (1.0f + expf(-g))) * up[(size_t) r * n + c];
    q8_1_value(h, xq + (size_t) r * (n / 32), c);
}

__global__ void fill_mask_kernel(float* __restrict__ x, const float* __restrict__ mask, int E) {
    const int r = blockIdx.y + 1, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < E) x[(size_t) r * E + c] = mask[c];
}

// ------------------------------------------------------------------------------------------------- Markov head
// Grid 3 rank (an output each), 256 threads: f's part once, then each step's h.
constexpr int kMaxSteps = 16;
__global__ void __launch_bounds__(256) markov_context_kernel(const float* __restrict__ W, const float* __restrict__ b,
                                                             const float* __restrict__ h, int ld_h, const float* __restrict__ f, int n,
                                                             int rank, int E, float* __restrict__ out) {
    __shared__ float red[32];
    const int o = blockIdx.x, R3 = 3 * rank;
    const float* wr = W + (size_t) o * (2 * rank + 2 * E) + 2 * rank;
    float accf = 0.0f, acc[kMaxSteps];
#pragma unroll
    for (int k = 0; k < kMaxSteps; ++k) acc[k] = 0.0f;
    for (int c = threadIdx.x; c < E; c += blockDim.x) {
        const float wh = wr[c];
        accf = fmaf(wr[E + c], f[c], accf);
#pragma unroll
        for (int k = 0; k < kMaxSteps; ++k)
            if (k < n) acc[k] = fmaf(wh, h[(size_t) k * ld_h + c], acc[k]);
    }
    const float tf = ds_block_sum(accf, red);
    for (int k = 0; k < n; ++k) {
        const float t = ds_block_sum(acc[k], red);
        if (threadIdx.x == 0) out[(size_t) k * R3 + o] = t + tf + b[o];
    }
}

// raw[o] = ctx[o] + W[o][:rank] . state + W[o][rank:2 rank] . emb: a warp an output, eight a block
__global__ void __launch_bounds__(256) markov_raw_kernel(const float* __restrict__ ctx, const float* __restrict__ W,
                                                         const float* __restrict__ state, const float* __restrict__ emb, int rank, int E,
                                                         float* __restrict__ raw) {
    const int lane = threadIdx.x & 31, o = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (o >= 3 * rank) return;
    const float* wr = W + (size_t) o * (2 * rank + 2 * E);
    float acc = 0.0f;
    for (int i = lane; i < rank; i += 32) acc = fmaf(wr[i], state[i], fmaf(wr[rank + i], emb[i], acc));
    acc = ds_warp_sum(acc);
    if (lane == 0) raw[o] = ctx[o] + acc;
}

__global__ void markov_gate_kernel(const float* __restrict__ raw, int rank, float* __restrict__ state, float* __restrict__ latent) {
    const int o = threadIdx.x;
    if (o >= rank) return;
    const float g = 1.0f / (1.0f + expf(-raw[o]));
    const float cand = tanhf(raw[rank + o]);
    state[o] = g * state[o] + (1.0f - g) * cand;
    latent[o] = tanhf(raw[2 * rank + o]);
}

// emb = w1[*token] as fp32 (the table may be mapped host memory: one row crosses the bus, once)
__global__ void markov_embed_kernel(const uint16_t* __restrict__ w1, const int32_t* __restrict__ token, int rank, float* __restrict__ emb) {
    const int i = threadIdx.x;
    if (i < rank) emb[i] = bf16_to_f(w1[(size_t) *token * rank + i]);
}

// (m, s, i) += (m2, s2, i2): the larger max (the lower index of equals) and the sums of exp rescaled to it
__device__ __forceinline__ void pick_join(float& m, float& s, int& i, float m2, float s2, int i2) {
    if (m2 == -INFINITY) return;
    if (m == -INFINITY) {
        m = m2;
        s = s2;
        i = i2;
    } else if (m2 > m || (m2 == m && i2 < i)) {
        s = s * expf(m - m2) + s2;
        m = m2;
        i = i2;
    } else {
        s += s2 * expf(m2 - m);
    }
}

// ------------------------------------------------------------------------------------------- the head's screen
// basalt/kernels/head_proxy.cuh's 2-bit copy of the Q8_0 head (rows of 2304), as Hy4-flash's drafter screens its head.
constexpr int kScreenIn = 2304, kScreenNb = kScreenIn / 32, kScreenBlocks = 512;
struct Q80B {
    __half d;
    int8_t qs[32];
};
static_assert(sizeof(Q80B) == 34, "a Q8_0 block is 34 bytes");

// a thread per 32-weight group: the group's Q8_0 values fitted into the copy (q80_screen's packing)
__global__ void head_screen_pack_kernel(const Q80B* __restrict__ head, int64_t n_rows, uint4* __restrict__ wq, uint16_t* __restrict__ sc) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows * kScreenNb) return;
    const int64_t row = i / kScreenNb;
    const int g = (int) (i % kScreenNb);
    const Q80B* b = head + (size_t) row * kScreenNb + g;
    const float d = __half2float(b->d);
    float w[32];
    for (int l = 0; l < 32; ++l) w[l] = d * (float) b->qs[l];
    px::pack_group<kScreenIn, true>(w, row, g, wq, sc);
}

// A block's tiles of 32 rows in turn (head_proxy's score stage: four warps split a row, lane = row) against the n_act
// q8_1 rows, staged once: out[a * ld + row] = the copy's dot of the row with activation a.
__global__ void __launch_bounds__(128) head_screen_score_kernel(const uint4* __restrict__ wq, const uint32_t* __restrict__ sc,
                                                                const px::X81* __restrict__ x, int n_act, int n_rows,
                                                                float* __restrict__ out, int ld) {
    constexpr int kNw4 = px::nw4(kScreenIn), NW = kNw4 / 4;
    __shared__ __align__(16) int xi[kScreenMaxRows][kScreenIn / 4];
    __shared__ float xs[kScreenMaxRows][kScreenNb], xdd[kScreenMaxRows][kScreenNb];
    __shared__ float red[kScreenMaxRows][4][32];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int i = tid; i < n_act * (kScreenIn / 4); i += 128) {
        const int a = i / (kScreenIn / 4), j = i % (kScreenIn / 4);
        xi[a][j] = reinterpret_cast<const int*>(x[(size_t) a * kScreenNb + j / 8].qs)[j % 8];
    }
    for (int i = tid; i < n_act * kScreenNb; i += 128) {
        const int a = i / kScreenNb, b = i % kScreenNb;
        const px::X81& blk = x[(size_t) a * kScreenNb + b];
        const int* q = reinterpret_cast<const int*>(blk.qs);
        int sum = 0;
#pragma unroll
        for (int t = 0; t < 8; ++t) sum = DP4A(0x01010101, q[t], sum);
        const float d8 = __low2float(blk.ds);
        xdd[a][b] = d8;
        xs[a][b] = d8 * (float) sum;
    }
    __syncthreads();
    const int n_tiles = (int) px::tiles(n_rows);
    for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
        const size_t base = ((size_t) tile * kNw4 + warp * NW) * 32 + lane;
        uint4 w[NW];
        uint32_t s[NW];
#pragma unroll
        for (int j = 0; j < NW; ++j) {
            w[j] = __ldcs(wq + base + (size_t) j * 32);
            s[j] = __ldcs(sc + base + (size_t) j * 32);
        }
        for (int a = 0; a < n_act; ++a) {
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < NW; ++j) {
                const int w4 = warp * NW + j;
                const uint32_t ww[4] = {w[j].x, w[j].y, w[j].z, w[j].w};
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int g = 2 * w4 + h;
                    int P = 0;
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const uint32_t v = ww[2 * h + c];
                        const int4 x0 = *reinterpret_cast<const int4*>(&xi[a][g * 8 + c * 4]);
                        P = DP4A((int) (v & 0x03030303u), x0.x, P);
                        P = DP4A((int) ((v >> 2) & 0x03030303u), x0.y, P);
                        P = DP4A((int) ((v >> 4) & 0x03030303u), x0.z, P);
                        P = DP4A((int) ((v >> 6) & 0x03030303u), x0.w, P);
                    }
                    const float sg = __half2float(__ushort_as_half((unsigned short) (h ? (s[j] >> 16) : (s[j] & 0xffff))));
                    acc += sg * (xdd[a][g] * (float) P - 1.5f * xs[a][g]);
                }
            }
            red[a][warp][lane] = acc;
        }
        __syncthreads();
        const int row = tile * 32 + lane;
        for (int a = warp; a < n_act; a += 4)
            if (row < n_rows) out[(size_t) a * ld + row] = red[a][0][lane] + red[a][1][lane] + red[a][2][lane] + red[a][3][lane];
        __syncthreads();
    }
}

// ------------------------------------------------------------------------------------------------ the step's pick
// Grid of blocks over the vocabulary, 256 threads, rank 256: a warp takes kPickRows rows at a time (each lane one 16-byte
// load of each fold row's eight values, all in flight before the sums).  Row v's approximate total is t = base[v] (the
// screen's) + fold[v] . latent + fold_b[v]; the block keeps the running softmax of the capped totals (max, sum of exp)
// in part[3 block ..] and its px::kPer best totals (the lower row of equals) in cand[block px::kPer ..] = {t, row}.
constexpr int kPickThreads = 256, kPickBlocks = 768, kPickRows = 4, kRank = 256, kJoinThreads = 512;
constexpr int kPer = px::kPer;
__device__ __forceinline__ void keep_best(float (&tv)[kPer], int (&ti)[kPer], float t, int v) {
    if (!(t > tv[kPer - 1] || (t == tv[kPer - 1] && v < ti[kPer - 1]))) return;
    int k = kPer - 1;
    while (k > 0 && (t > tv[k - 1] || (t == tv[k - 1] && v < ti[k - 1]))) {
        tv[k] = tv[k - 1];
        ti[k] = ti[k - 1];
        --k;
    }
    tv[k] = t;
    ti[k] = v;
}

__global__ void __launch_bounds__(kPickThreads) markov_logits_kernel(const float* __restrict__ base, const uint16_t* __restrict__ fold,
                                                                     const float* __restrict__ fold_b, const float* __restrict__ latent,
                                                                     int n_vocab, float cap, float* __restrict__ part,
                                                                     float2* __restrict__ cand) {
    __shared__ float wm[kPickThreads / 32], ws[kPickThreads / 32];
    __shared__ float bv[kPickThreads / 32][kPer];
    __shared__ int bi[kPickThreads / 32][kPer];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float lat[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) lat[t] = latent[lane * 8 + t];
    float m = -INFINITY, s = 0.0f;
    float tv[kPer];
    int ti[kPer];
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        tv[k] = -INFINITY;
        ti[k] = 0x7fffffff;
    }
    const int stride = gridDim.x * (kPickThreads / 32) * kPickRows;
    for (int v0 = (blockIdx.x * (kPickThreads / 32) + warp) * kPickRows; v0 < n_vocab; v0 += stride) {
        uint4 w[kPickRows];
        float add[kPickRows];
#pragma unroll
        for (int u = 0; u < kPickRows; ++u) {
            const int v = v0 + u;
            w[u] = v < n_vocab ? __ldg((const uint4*) (fold + (size_t) v * kRank) + lane) : make_uint4(0, 0, 0, 0);
            add[u] = v < n_vocab ? __ldg(base + v) + __ldg(fold_b + v) : 0.0f;
        }
#pragma unroll
        for (int u = 0; u < kPickRows; ++u) {
            const int v = v0 + u;
            const uint32_t q[4] = {w[u].x, w[u].y, w[u].z, w[u].w};
            float d = 0.0f;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                d = fmaf(__uint_as_float(q[t] << 16), lat[2 * t], d);
                d = fmaf(__uint_as_float(q[t] & 0xffff0000u), lat[2 * t + 1], d);
            }
            d = ds_warp_sum(d);
            if (v >= n_vocab) break;
            const float t = add[u] + d, x = cap * tanhf(t / cap);
            if (x > m) {
                s = s * expf(m - x) + 1.0f;
                m = x;
            } else {
                s += expf(x - m);
            }
            keep_best(tv, ti, t, v);
        }
    }
    if (lane == 0) {
        wm[warp] = m;
        ws[warp] = s;
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            bv[warp][k] = tv[k];
            bi[warp][k] = ti[k];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float bm = -INFINITY, bs = 0.0f;
        int unused = 0x7fffffff;
        float cv[kPer];
        int ci[kPer];
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            cv[k] = -INFINITY;
            ci[k] = 0x7fffffff;
        }
        for (int w = 0; w < kPickThreads / 32; ++w) {
            pick_join(bm, bs, unused, wm[w], ws[w], 0);
            for (int k = 0; k < kPer; ++k) keep_best(cv, ci, bv[w][k], bi[w][k]);
        }
        part[2 * blockIdx.x] = bm;
        part[2 * blockIdx.x + 1] = bs;
        for (int k = 0; k < kPer; ++k) cand[blockIdx.x * kPer + k] = make_float2(cv[k], __int_as_float(ci[k] == 0x7fffffff ? 0 : ci[k]));
    }
}

// The step's draft (kJoinThreads threads): the rows within px::kDelta of the best approximate total (px::collect, up to
// px::kMaxCand) rescored exactly, a warp a row: the Q8_0 head row against the step's q8_1 row x, plus the bias through
// fold; the best exact capped logit is the draft (the lower row of equals), its probability against the blocks' softmax
// of the approximate ones.  The token also goes to *prev and its w1 row to emb (null: not).
__global__ void __launch_bounds__(kJoinThreads) markov_join_kernel(const float* __restrict__ part, const float2* __restrict__ cand,
                                                                   const Q80B* __restrict__ head, const px::X81* __restrict__ x,
                                                                   const uint16_t* __restrict__ fold, const float* __restrict__ fold_b,
                                                                   const float* __restrict__ latent, float cap,
                                                                   const uint16_t* __restrict__ w1, int32_t* __restrict__ token,
                                                                   float* __restrict__ prob, int32_t* __restrict__ prev,
                                                                   float* __restrict__ emb) {
    __shared__ px::Final f;
    __shared__ float lat[kRank], jm[kJoinThreads / 32], js[kJoinThreads / 32];
    __shared__ int pick;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid < kRank) lat[tid] = latent[tid];
    if (tid == 0) f.cnt = 0;
    {   // the approximate softmax: each thread its stride of parts, then the warps, then their sixteen, in a fixed order
        float m = -INFINITY, s = 0.0f;
        int unused = 0;
        for (int b = tid; b < kPickBlocks; b += kJoinThreads) pick_join(m, s, unused, part[2 * b], part[2 * b + 1], 0);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float m2 = __shfl_xor_sync(0xffffffffu, m, o), s2 = __shfl_xor_sync(0xffffffffu, s, o);
            pick_join(m, s, unused, m2, s2, 0);
        }
        if (lane == 0) {
            jm[warp] = m;
            js[warp] = s;
        }
    }
    __syncthreads();
    int none = 0;
    const int n = px::collect<false, false>(f, kPickBlocks, nullptr, cand, &none, nullptr);
    for (int c = warp; c < n; c += kJoinThreads / 32) {
        const int v = f.list[c];
        const Q80B* hr = head + (size_t) v * kScreenNb;
        float dot = 0.0f;
        for (int b = lane; b < kScreenNb; b += 32) {
            const uint16_t* q16 = reinterpret_cast<const uint16_t*>(hr[b].qs);
            const int* xq = reinterpret_cast<const int*>(x[b].qs);
            int sumi = 0;
#pragma unroll
            for (int t = 0; t < 8; ++t) sumi = DP4A((int) q16[2 * t] | ((int) q16[2 * t + 1] << 16), xq[t], sumi);
            dot += __half2float(hr[b].d) * __low2float(x[b].ds) * (float) sumi;
        }
        const uint4 fw = __ldg((const uint4*) (fold + (size_t) v * kRank) + lane);
        const uint32_t q[4] = {fw.x, fw.y, fw.z, fw.w};
        float d = 0.0f;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            d = fmaf(__uint_as_float(q[t] << 16), lat[lane * 8 + 2 * t], d);
            d = fmaf(__uint_as_float(q[t] & 0xffff0000u), lat[lane * 8 + 2 * t + 1], d);
        }
        dot = ds_warp_sum(dot);
        d = ds_warp_sum(d);
        if (lane == 0) f.ex[c] = cap * tanhf((dot + d + fold_b[v]) / cap);
    }
    __syncthreads();
    if (warp == 0) px::pick<false, false>(f, n, kPickBlocks, nullptr, nullptr, &pick, nullptr);
    __syncthreads();
    if (tid == 0) {
        float m = -INFINITY, s = 0.0f, best = -INFINITY;
        int unused = 0;
        for (int w = 0; w < kJoinThreads / 32; ++w) pick_join(m, s, unused, jm[w], js[w], 0);
        for (int c = 0; c < n; ++c)
            if (f.list[c] == pick) best = f.ex[c];
        *token = pick;
        *prob = n > 0 && s > 0.0f ? fminf(1.0f, expf(best - m) / s) : 1.0f;
        *prev = pick;
    }
    if (emb && tid < kRank) emb[tid] = bf16_to_f(w1[(size_t) pick * kRank + tid]);
}

}  // namespace

void features_store(const float* x, const Members& m, const FeatureDst& dst, int n_feat, int slot, int n_embd, void* stream) {
    if (m.rows() <= 0) return;
    features_store_kernel<<<dim3((unsigned) m.rows(), (unsigned) ((n_embd + 255) / 256)), 256, 0, (cudaStream_t) stream>>>(
        x, m, dst, n_feat, slot, n_embd);
}

void feature_mix(const uint16_t* feat, int rows, int n_feat, const Mixes& mixes, int n_embd, float eps, float* out, void* xq, void* stream) {
    if (rows <= 0 || mixes.n <= 0) return;
    feature_mix_kernel<<<dim3((unsigned) rows, (unsigned) mixes.n), kRowThreads, 0, (cudaStream_t) stream>>>(
        feat, n_feat, mixes, rows, n_embd, eps, out, (Q81*) xq);
}

void qk_norm_rope_store(float* q, const float* q_w, const float* k, const float* k_w, const float* v, const QkArgs& a, const int64_t* pos,
                        int rows, uint16_t* k_cache, uint16_t* v_cache, int64_t cells, void* stream) {
    if (rows <= 0) return;
    const unsigned segs = (unsigned) ((q ? a.n_head : 0) + 2 * a.n_kv);
    qk_norm_rope_kernel<<<dim3((unsigned) rows, segs), kHeadDim, 0, (cudaStream_t) stream>>>(q, q_w, k, k_w, v, a, pos, (__half*) k_cache,
                                                                                             (__half*) v_cache, (long long) cells);
}

void add_norm_quant(float* res, const float* add, const float* w, int rows, int n_embd, float eps, float* out, void* xq, void* stream) {
    if (rows <= 0) return;
    add_norm_quant_kernel<<<(unsigned) rows, kRowThreads, 0, (cudaStream_t) stream>>>(res, add, w, n_embd, eps, out, (Q81*) xq);
}

void swiglu_quant(const float* gate, const float* up, int rows, int n, void* xq, void* stream) {
    if (rows <= 0) return;
    swiglu_quant_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) rows), 256, 0, (cudaStream_t) stream>>>(gate, up, n, (Q81*) xq);
}

void fill_mask_rows(float* x, const float* mask, int rows, int n_embd, void* stream) {
    if (rows <= 1) return;
    fill_mask_kernel<<<dim3((unsigned) ((n_embd + 255) / 256), (unsigned) (rows - 1)), 256, 0, (cudaStream_t) stream>>>(x, mask, n_embd);
}

void markov_context(const float* W, const float* b, const float* h, int ld_h, const float* f, int n, int rank, int n_embd, float* out,
                    void* stream) {
    if (n <= 0 || n > kMaxSteps) return;
    markov_context_kernel<<<(unsigned) (3 * rank), 256, 0, (cudaStream_t) stream>>>(W, b, h, ld_h, f, n, rank, n_embd, out);
}

void markov_step(const float* ctx, const float* W, const float* emb, int rank, int n_embd, float* raw, float* state, float* latent,
                 void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    markov_raw_kernel<<<(unsigned) ((3 * rank + 7) / 8), 256, 0, st>>>(ctx, W, state, emb, rank, n_embd, raw);
    markov_gate_kernel<<<1, (unsigned) ((rank + 31) / 32 * 32), 0, st>>>(raw, rank, state, latent);
}

void markov_embed(const uint16_t* w1, const int32_t* token, int rank, float* emb, void* stream) {
    markov_embed_kernel<<<1, (unsigned) ((rank + 31) / 32 * 32), 0, (cudaStream_t) stream>>>(w1, token, rank, emb);
}

size_t head_screen_bytes(int64_t n_rows) { return px::bytes(n_rows, kScreenIn); }

void head_screen_pack(const void* head_q8_0, int64_t n_rows, void* screen, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    cudaMemsetAsync(screen, 0, px::bytes(n_rows, kScreenIn), st);   // the last tile's rows past n_rows score as zeros
    const int64_t total = n_rows * kScreenNb;
    head_screen_pack_kernel<<<(unsigned) ((total + 127) / 128), 128, 0, st>>>((const Q80B*) head_q8_0, n_rows, (uint4*) screen,
                                                                             px::scales(screen, n_rows, kScreenIn));
}

void head_screen_scores(const void* screen, const void* x_q8_1, int n_act, int64_t n_rows, float* out, int ld, void* stream) {
    if (n_act <= 0 || n_act > kScreenMaxRows) return;
    const int tiles = (int) px::tiles(n_rows);
    head_screen_score_kernel<<<(unsigned) std::min(tiles, kScreenBlocks), 128, 0, (cudaStream_t) stream>>>(
        (const uint4*) screen, (const uint32_t*) px::scales(const_cast<void*>(screen), n_rows, kScreenIn), (const px::X81*) x_q8_1, n_act,
        (int) n_rows, out, ld);
}

size_t markov_pick_scratch_floats(int) { return (size_t) 2 * kPickBlocks + (size_t) 2 * kPickBlocks * kPer; }

void markov_pick(const float* base, const void* head_q8_0, const void* x_q8_1, const uint16_t* fold, const float* fold_b, const float* latent,
                 int n_vocab, float cap, float* scratch, const uint16_t* w1, int32_t* token, float* prob, int32_t* prev, float* emb,
                 void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    float2* cand = (float2*) (scratch + 2 * kPickBlocks);
    markov_logits_kernel<<<kPickBlocks, kPickThreads, 0, st>>>(base, fold, fold_b, latent, n_vocab, cap, scratch, cand);
    markov_join_kernel<<<1, kJoinThreads, 0, st>>>(scratch, cand, (const Q80B*) head_q8_0, (const px::X81*) x_q8_1, fold, fold_b, latent,
                                                   cap, w1, token, prob, prev, emb);
}

}  // namespace basalt::models::inkling2x::kernels
