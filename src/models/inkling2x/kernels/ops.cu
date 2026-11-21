// src/models/inkling2x/kernels/ops.cu - Inkling 2's row kernels: the depth mix, the loop controllers, the short
// convolutions, the attention's input preparation, the routed experts' grouping and sum, and the capped logits.
#include "basalt/models/inkling2x/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace basalt::models::inkling2x::kernels {
namespace {

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// The sum over a block of up to 1024 threads; every thread gets it. `red` holds 32 floats.
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = lane < nw ? red[lane] : 0.0f;
    return warp_sum(t);
}

// ---------------------------------------------------------------------------------------------------------- depth mix
// A block per row, query and slice of the output. The scores need every value of the states, so each block sums all of them
// in one pass (the same threads over the same values whatever the slice, so a row's output does not depend on the slicing);
// the slices spread a short run of rows over more of the card.
constexpr int kMixThreads = 256;

__global__ void __launch_bounds__(kMixThreads) depth_mix_kernel(States s, const float* __restrict__ queries, int rows, int E, float eps,
                                                                int slice, float* __restrict__ out) {
    const int r = blockIdx.x, qi = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ float part[kMixThreads / 32][2 * kMaxStates], inv[kMaxStates], p[kMaxStates];
    const float* q = queries + (size_t) qi * E;
    float ss[kMaxStates], d[kMaxStates];
#pragma unroll
    for (int i = 0; i < kMaxStates; ++i) ss[i] = d[i] = 0.0f;
    for (int e = threadIdx.x; e < E; e += kMixThreads) {
        const float qe = q[e];
#pragma unroll
        for (int i = 0; i < kMaxStates; ++i)
            if (i < s.n) {
                const float v = s.p[i][(size_t) r * E + e];
                ss[i] += v * v;
                d[i] += v * qe;
            }
    }
#pragma unroll
    for (int i = 0; i < kMaxStates; ++i)
        if (i < s.n) {
            ss[i] = warp_sum(ss[i]);
            d[i] = warp_sum(d[i]);
            if (lane == 0) {
                part[warp][2 * i] = ss[i];
                part[warp][2 * i + 1] = d[i];
            }
        }
    __syncthreads();
    if (threadIdx.x < s.n) {
        const int i = threadIdx.x;
        float a = 0.0f, b = 0.0f;
        for (int w = 0; w < kMixThreads / 32; ++w) {
            a += part[w][2 * i];
            b += part[w][2 * i + 1];
        }
        const float iv = 1.0f / sqrtf(a / (float) E + eps);
        inv[i] = iv;
        p[i] = b * iv / sqrtf((float) E);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -INFINITY, sum = 0.0f;
        for (int i = 0; i < s.n; ++i) m = fmaxf(m, p[i]);
        for (int i = 0; i < s.n; ++i) sum += (p[i] = expf(p[i] - m));
        for (int i = 0; i < s.n; ++i) p[i] /= sum;
    }
    __syncthreads();
    float* o = out + ((size_t) qi * rows + r) * E;
    const int e1 = min(E, (int) (blockIdx.z + 1) * slice);
    for (int e = (int) blockIdx.z * slice + threadIdx.x; e < e1; e += kMixThreads) {
        float acc = 0.0f;
        for (int i = 0; i < s.n; ++i) acc += p[i] * (s.p[i][(size_t) r * E + e] * inv[i]);
        o[e] = acc;
    }
}

// ---------------------------------------------------------------------------------------------------- loop controllers
// g[r][o] = D[o] . [rms(cur); rms(msg)]: a block per output and row, the norms' sums taken in the same pass as the products
// (the norm is a factor of each half: g = (D_a . a) inv_a + (D_b . b) inv_b)
constexpr int kCtlThreads = 256;

__global__ void __launch_bounds__(kCtlThreads) control_down_kernel(const float* __restrict__ cur, const float* __restrict__ msg,
                                                                   const float* __restrict__ D, int E, int rank, float eps,
                                                                   float* __restrict__ g) {
    const int o = blockIdx.x, r = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ float part[kCtlThreads / 32][4];
    const float* a = cur + (size_t) r * E;
    const float* b = msg + (size_t) r * E;
    const float* w = D + (size_t) o * 2 * E;
    float sa = 0.0f, sb = 0.0f, da = 0.0f, db = 0.0f;
    for (int e = threadIdx.x; e < E; e += kCtlThreads) {
        const float av = a[e], bv = b[e];
        sa += av * av;
        sb += bv * bv;
        da += w[e] * av;
        db += w[E + e] * bv;
    }
    sa = warp_sum(sa);
    sb = warp_sum(sb);
    da = warp_sum(da);
    db = warp_sum(db);
    if (lane == 0) {
        part[warp][0] = sa;
        part[warp][1] = sb;
        part[warp][2] = da;
        part[warp][3] = db;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float t[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int w8 = 0; w8 < kCtlThreads / 32; ++w8)
            for (int j = 0; j < 4; ++j) t[j] += part[w8][j];
        g[(size_t) r * rank + o] = t[2] * (1.0f / sqrtf(t[0] / (float) E + eps)) + t[3] * (1.0f / sqrtf(t[1] / (float) E + eps));
    }
}

// out[r][e] = cur - tanh(F[e] . g) cur + tanh(G[e] . g) msg: a warp per value, its lanes over the rank (G and F rows contiguous)
__global__ void __launch_bounds__(kCtlThreads) control_out_kernel(const float* __restrict__ cur, const float* __restrict__ msg,
                                                                  const float* __restrict__ G, const float* __restrict__ F,
                                                                  const float* __restrict__ g, int E, int rank, float* __restrict__ out) {
    const int r = blockIdx.y, lane = threadIdx.x & 31, e = blockIdx.x * (kCtlThreads / 32) + (threadIdx.x >> 5);
    if (e >= E) return;
    const float* gr = g + (size_t) r * rank;
    float ga = 0.0f, fo = 0.0f;
    for (int i = lane; i < rank; i += 32) {
        const float gi = gr[i];
        ga += G[(size_t) e * rank + i] * gi;
        fo += F[(size_t) e * rank + i] * gi;
    }
    ga = warp_sum(ga);
    fo = warp_sum(fo);
    if (lane == 0) {
        const float c = cur[(size_t) r * E + e], m = msg[(size_t) r * E + e];
        out[(size_t) r * E + e] = (c - tanhf(fo) * c) + tanhf(ga) * m;
    }
}

// ----------------------------------------------------------------------------------------------- short convolutions
__global__ void set_position_kernel(int64_t* pos, long long value) { *pos = value; }

__global__ void short_conv_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ dyn, int hw,
                                  int n_dyn, ConvRing ring, int ch0, const int64_t* __restrict__ pos, int C, float scale,
                                  float* __restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (c >= C) return;
    const long long pos0 = *pos;
    float xv[4];   // x_(t-3) .. x_t
    for (int k = 0; k < 4; ++k) {
        const int rr = r - k;
        const long long p = pos0 + rr;
        xv[3 - k] = rr >= 0 ? x[(size_t) rr * C + c]
                  : p >= 0  ? ring.data[(size_t) (p % ring.slots) * ring.width + ch0 + c]
                            : 0.0f;
    }
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) acc += xv[i] * w[(size_t) c * 4 + i];
    if (dyn) {
        const float* tap = dyn + (size_t) r * n_dyn + (size_t) (c / hw) * 4;
        for (int lag = 0; lag < 4; ++lag) acc = acc + xv[3 - lag] * tap[lag];
    }
    out[(size_t) r * C + c] = (xv[3] + acc) * scale;
}

__global__ void conv_store_kernel(const float* __restrict__ x, ConvRing ring, int ch0, const int64_t* __restrict__ pos, int r0, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, r = r0 + blockIdx.y;
    if (c >= C) return;
    const long long pos0 = *pos;
    ring.data[(size_t) ((pos0 + r) % ring.slots) * ring.width + ch0 + c] = x[(size_t) r * C + c];
}

// ------------------------------------------------------------------------------------------ the attention's inputs
// blockIdx.y: n_head Q heads, n_kv K heads, n_kv V heads, then n_head R heads; 128 threads.
__global__ void qkvr_prep_kernel(float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ rr,
                                 const float* __restrict__ q_scale, const float* __restrict__ r_scale, PrepArgs a,
                                 const int64_t* __restrict__ pos, __half* __restrict__ kc, __half* __restrict__ vc, long long cells) {
    const int row = blockIdx.x, seg = blockIdx.y, d = threadIdx.x;
    __shared__ float red[32];
    const long long p = *pos + row;
    const float tau = a.log_scale ? 1.0f + a.log_alpha * logf(fmaxf((float) (p + 1) * (1.0f / a.log_floor), 1.0f)) : 1.0f;
    const size_t cell = (size_t) (p % cells);
    if (seg < a.n_head) {   // Q
        float* x = q + ((size_t) row * a.n_head + seg) * 128;
        const float val = x[d] * q_scale[(size_t) seg * 128 + d];
        const float norm = sqrtf(block_sum(val * val, red)) + 1e-6f;
        const float f = fminf(fmaxf(a.qk_clip / norm, 0.0f), 1.0f);
        x[d] = val * f * tau;
    } else if (seg < a.n_head + a.n_kv) {   // K
        const int h = seg - a.n_head;
        const float val = k[((size_t) row * a.n_kv + h) * 128 + d];
        const float norm = sqrtf(block_sum(val * val, red)) + 1e-6f;
        const float f = fminf(fmaxf(a.qk_clip / norm, 0.0f), 1.0f);
        kc[(cell * a.n_kv + h) * 128 + d] = __float2half_rn(val * f);
    } else if (seg < a.n_head + 2 * a.n_kv) {   // V
        const int h = seg - a.n_head - a.n_kv;
        const float val = v[((size_t) row * a.n_kv + h) * 128 + d];
        const float ms = block_sum(val * val, red) / 128.0f;
        vc[(cell * a.n_kv + h) * 128 + d] = __float2half_rn(val * (1.0f / sqrtf(ms + a.eps)));
    } else {   // R
        const int h = seg - a.n_head - 2 * a.n_kv;
        float* x = rr + ((size_t) row * a.n_head + h) * a.d_rel;
        const float val = d < a.d_rel ? x[d] * r_scale[(size_t) h * a.d_rel + d] : 0.0f;
        const float norm = sqrtf(block_sum(val * val, red)) + 1e-6f;
        const float f = fminf(fmaxf(a.r_clip / norm, 0.0f), 1.0f);
        if (d < a.d_rel) x[d] = val * f * tau;
    }
}

// ----------------------------------------------------------------------------------------------- the routed experts
// One block, a thread per expert: each counts its entries, a scan over the block lays the groups out in expert order, and
// each expert writes its entries in row order.
__global__ void group_experts_kernel(const int32_t* __restrict__ ids, int n, int n_expert, const unsigned long long* __restrict__ ptr,
                                     const Exl3Rates* __restrict__ rates, unsigned long long* __restrict__ grp_ptr, Exl3Rates* __restrict__ grp_rates,
                                     int32_t* __restrict__ grp_start, int32_t* __restrict__ n_groups, int32_t* __restrict__ ent_dst,
                                     int32_t* __restrict__ ent_tok, int k) {
    __shared__ int wc[32], wp[32];
    const int e = threadIdx.x, lane = e & 31, warp = e >> 5, nw = blockDim.x >> 5;
    int c = 0;
    if (e < n_expert)
        for (int i = 0; i < n; ++i) c += ids[i] == e;
    const int has = c > 0;
    int sc = c, sp = has;   // inclusive scans over the warp
    for (int o = 1; o < 32; o <<= 1) {
        const int tc = __shfl_up_sync(0xffffffffu, sc, o), tp = __shfl_up_sync(0xffffffffu, sp, o);
        if (lane >= o) {
            sc += tc;
            sp += tp;
        }
    }
    if (lane == 31) {
        wc[warp] = sc;
        wp[warp] = sp;
    }
    __syncthreads();
    int at = sc - c, gi = sp - has;
    for (int w = 0; w < warp; ++w) {
        at += wc[w];
        gi += wp[w];
    }
    if (has) {
        grp_ptr[gi] = ptr[e];
        if (rates) grp_rates[gi] = rates[e];
        grp_start[gi] = at;
        for (int i = 0; i < n; ++i)
            if (ids[i] == e) {
                ent_dst[at] = i;
                ent_tok[at] = i / k;
                ++at;
            }
    }
    if (e == 0) {
        int tc = 0, tg = 0;
        for (int w = 0; w < nw; ++w) {
            tc += wc[w];
            tg += wp[w];
        }
        grp_start[tg] = tc;
        *n_groups = tg;
    }
}

__global__ void sum_experts_kernel(const float* __restrict__ out, const float* __restrict__ w, int k, int E, float* __restrict__ y) {
    const int r = blockIdx.y, e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) acc = acc + out[((size_t) r * k + j) * E + e] * w[(size_t) r * k + j];
    y[(size_t) r * E + e] = acc;
}

__global__ void cap_logits_kernel(const float* __restrict__ x, int ld, int n_vocab, float cap, float* __restrict__ y) {
    const int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vocab) return;
    y[(size_t) r * n_vocab + i] = cap * tanhf(x[(size_t) r * ld + i] * (1.0f / cap));
}

}  // namespace

void set_position(int64_t* pos, int64_t value, void* stream) {
    set_position_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(pos, (long long) value);
}

void depth_mix(const States& s, const float* queries, int nq, int rows, int n_embd, float eps, float* out, void* stream) {
    if (rows <= 0) return;
    // about 128 blocks for a short run of rows, a block per row and query for a long one
    const int whole = (n_embd + kMixThreads - 1) / kMixThreads;
    const int nslice = std::max(1, std::min(whole, (128 + rows * nq - 1) / (rows * nq)));
    const int slice = (whole + nslice - 1) / nslice * kMixThreads;
    depth_mix_kernel<<<dim3((unsigned) rows, (unsigned) nq, (unsigned) ((n_embd + slice - 1) / slice)), kMixThreads, 0, (cudaStream_t) stream>>>(
        s, queries, rows, n_embd, eps, slice, out);
}

void loop_control(const float* cur, const float* msg, const float* down, const float* gain, const float* forget, int rows, int n_embd,
                  int rank, float eps, float* g, float* out, void* stream) {
    if (rows <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    control_down_kernel<<<dim3((unsigned) rank, (unsigned) rows), kCtlThreads, 0, st>>>(cur, msg, down, n_embd, rank, eps, g);
    control_out_kernel<<<dim3((unsigned) ((n_embd + kCtlThreads / 32 - 1) / (kCtlThreads / 32)), (unsigned) rows), kCtlThreads, 0, st>>>(
        cur, msg, gain, forget, g, n_embd, rank, out);
}

void short_conv(const float* x, const float* w, const float* dyn, int head_width, const ConvRing& ring, int ch0, const int64_t* pos,
                int rows, int C, float scale, float* out, void* stream) {
    if (rows <= 0) return;
    const int n_dyn = dyn ? C / head_width * 4 : 0;
    short_conv_kernel<<<dim3((unsigned) ((C + 255) / 256), (unsigned) rows), 256, 0, (cudaStream_t) stream>>>(
        x, w, dyn, head_width, n_dyn, ring, ch0, pos, C, scale, out);
}

void conv_store(const float* x, const ConvRing& ring, int ch0, const int64_t* pos, int rows, int keep, int C, void* stream) {
    const int n = keep < rows ? keep : rows;
    if (n <= 0) return;
    conv_store_kernel<<<dim3((unsigned) ((C + 255) / 256), (unsigned) n), 256, 0, (cudaStream_t) stream>>>(x, ring, ch0, pos, rows - n, C);
}

void qkvr_prep(float* q, const float* k, const float* v, float* r, const float* q_scale, const float* r_scale, const PrepArgs& a,
               const int64_t* pos, int rows, uint16_t* k_cache, uint16_t* v_cache, int64_t cells, void* stream) {
    if (rows <= 0) return;
    qkvr_prep_kernel<<<dim3((unsigned) rows, (unsigned) (2 * a.n_head + 2 * a.n_kv)), 128, 0, (cudaStream_t) stream>>>(
        q, k, v, r, q_scale, r_scale, a, pos, (__half*) k_cache, (__half*) v_cache, cells);
}

void group_experts(const int32_t* ids, int rows, int k, int n_expert, const unsigned long long* expert_ptr, const Exl3Rates* expert_rates,
                   unsigned long long* grp_ptr, Exl3Rates* grp_rates, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok,
                   void* stream) {
    group_experts_kernel<<<1, (unsigned) ((n_expert + 31) / 32 * 32), 0, (cudaStream_t) stream>>>(
        ids, rows * k, n_expert, expert_ptr, expert_rates, grp_ptr, grp_rates, grp_start, n_groups, ent_dst, ent_tok, k);
}

void sum_experts(const float* out, const float* w, int rows, int k, int n_embd, float* y, void* stream) {
    if (rows <= 0) return;
    sum_experts_kernel<<<dim3((unsigned) ((n_embd + 255) / 256), (unsigned) rows), 256, 0, (cudaStream_t) stream>>>(out, w, k, n_embd, y);
}

void cap_logits(const float* x, int ld, int rows, int n_vocab, float cap, float* logits, void* stream) {
    if (rows <= 0) return;
    cap_logits_kernel<<<dim3((unsigned) ((n_vocab + 255) / 256), (unsigned) rows), 256, 0, (cudaStream_t) stream>>>(x, ld, n_vocab, cap, logits);
}

}  // namespace basalt::models::inkling2x::kernels
