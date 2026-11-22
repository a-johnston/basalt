// src/models/inkling2x/kernels/ops.cu - Inkling 2's row kernels: the depth mix, the norms with their q8_1 blocks, the loop
// controllers, the short convolutions (with the residual and the routed experts' sum), the attention's input
// preparation, the routed experts' grouping, and the capped logits.
#include "basalt/models/inkling2x/kernels.hpp"
#include "common.cuh"

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

// ------------------------------------------------------------------------------------------------ norms and q8_1
// dense_ops' block sum: the warps' sums added in order by every thread
__device__ __forceinline__ float block_sum_ordered(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += red[i];
    return t;
}

constexpr int kNormThreads = 512;   // rms_norm_rows's block
struct NormJobs {
    NormJob j[3];
};

// rms_norm_rows_kernel's arithmetic, then each warp's 32 values (n_embd a multiple of 32) quantized as
// quantize_q8_1_rows does.  Grid (rows, jobs).
__global__ void __launch_bounds__(kNormThreads) norm_quant_kernel(NormJobs a, int rows, int E, float eps, float* __restrict__ out,
                                                                  Q81* __restrict__ xq) {
    __shared__ float red[32];
    const int r = blockIdx.x, j = blockIdx.y;
    const float* xr = a.j[j].x + (size_t) r * E;
    const float* w = a.j[j].w;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < E; c += kNormThreads) ss = fmaf(xr[c], xr[c], ss);
    const float inv = rsqrtf(block_sum_ordered(ss, red) / (float) E + eps);
    float* yr = out + ((size_t) j * rows + r) * E;
    Q81* qr = xq ? xq + ((size_t) j * rows + r) * (E / 32) : nullptr;
    for (int c = threadIdx.x; c < E; c += kNormThreads) {
        const float v = xr[c] * inv * w[c];
        yr[c] = v;
        if (qr) q8_1_value(v, qr, c);
    }
}

// ---------------------------------------------------------------------------------------------------- loop controllers
// g[r][o] = D[o] . [rms(cur); rms(msg)]: a block per output and row, the norms' sums taken in the same pass as the products
// (the norm is a factor of each half: g = (D_a . a) inv_a + (D_b . b) inv_b).  blockIdx.z: the controller.
constexpr int kCtlThreads = 256;
struct CtlJobs {
    CtlJob j[3];
};

__global__ void __launch_bounds__(kCtlThreads) control_down_kernel(CtlJobs jobs, int rows, int E, int rank, float eps,
                                                                   float* __restrict__ g_all) {
    const int o = blockIdx.x, r = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ float part[kCtlThreads / 32][4];
    const CtlJob& jb = jobs.j[blockIdx.z];
    float* g = g_all + (size_t) blockIdx.z * rows * rank;
    const float* a = jb.cur + (size_t) r * E;
    const float* b = jb.msg + (size_t) r * E;
    const float* w = jb.down + (size_t) o * 2 * E;
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
__global__ void __launch_bounds__(kCtlThreads) control_out_kernel(CtlJobs jobs, int rows, int E, int rank, const float* __restrict__ g_all) {
    const int r = blockIdx.y, lane = threadIdx.x & 31, e = blockIdx.x * (kCtlThreads / 32) + (threadIdx.x >> 5);
    if (e >= E) return;
    const CtlJob& jb = jobs.j[blockIdx.z];
    const float *cur = jb.cur, *msg = jb.msg, *G = jb.gain, *F = jb.forget;
    float* out = jb.out;
    const float* gr = g_all + ((size_t) blockIdx.z * rows + r) * rank;
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
struct Firsts {
    long long p[kMaxMembers];
};
__global__ void set_positions_kernel(int64_t* pos, Members m, Firsts first) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m.start[m.n]) return;   // Members::rows() is the host's
    const int i = member_of(m, r);
    pos[r] = first.p[i] + (r - m.start[i]);
}

// Decode walks a channel's rows in order. A prompt block owns a set of channels, one thread per channel and row slice.
constexpr int kConvThreads = 64;
constexpr int kConvSlices = 8, kConvChannels = 16;
template<bool SUM>
__device__ __forceinline__ float conv_input(const ConvJob& j, const float* eout, const float* ew, int k, int r, int c) {
    if constexpr (SUM) {
        float acc = 0.0f;
        for (int q = 0; q < k; ++q) acc = acc + eout[((size_t) r * k + q) * j.C + c] * ew[(size_t) r * k + q];
        return acc;
    } else return j.x[(size_t) r * j.C + c];
}
template<bool SUM, bool RESID, bool SLICED>
__device__ __forceinline__ void conv_rows_stage(int c, int slice, ConvJob a, ConvJob b, int hw, ConvRing ring, Members mem,
                                                const int64_t* __restrict__ pos, const float* __restrict__ eout,
                                                const float* __restrict__ ew, int k, float* cur, float* save) {
    const bool second = c >= a.C;
    const ConvJob& j = second ? b : a;
    if (second) c -= a.C;
    if (c >= j.C) return;
    const int C = j.C, n_dyn = j.dyn ? C / hw * 4 : 0;
    float w4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) w4[i] = j.w[(size_t) c * 4 + i];
    for (int m = 0; m < mem.n; ++m) {
        const int r0 = mem.start[m], r1 = mem.start[m + 1];
        if (r0 >= r1) continue;
        float* rg = ring.data[m];
        const long long pos0 = pos[r0];
        const int keep = min(r1 - r0, ring.slots - 3);
        const int begin = SLICED ? r0 + (r1 - r0) * slice / kConvSlices : r0;
        const int end = SLICED ? r0 + (r1 - r0) * (slice + 1) / kConvSlices : r1;
        float h[3];   // x_(t-3), x_(t-2), x_(t-1)
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int rr = begin - 3 + i;
            const long long p = pos0 + (rr - r0);
            h[i] = SLICED && rr >= r0 ? conv_input<SUM>(j, eout, ew, k, rr, c) :
                   p >= 0 ? rg[(size_t) (p % ring.slots) * ring.width + j.ch0 + c] : 0.0f;
        }
        // All slices of a channel belong to this block. Read their three taps before any slice overwrites a ring cell.
        if constexpr (SLICED) __syncthreads();
        for (int r = begin; r < end; ++r) {
            const float xt = conv_input<SUM>(j, eout, ew, k, r, c);
            const float xv[4] = {h[0], h[1], h[2], xt};
            float acc = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) acc += xv[i] * w4[i];
            if (j.dyn) {
                const float* tap = j.dyn + (size_t) r * n_dyn + (size_t) (c / hw) * 4;
#pragma unroll
                for (int lag = 0; lag < 4; ++lag) acc = acc + xv[3 - lag] * tap[lag];
            }
            const float o = (xv[3] + acc) * j.scale;
            if (j.out) j.out[(size_t) r * C + c] = o;
            if constexpr (RESID) {
                const size_t at = (size_t) r * C + c;
                if (save) {
                    save[at] = cur[at];
                    cur[at] = o;
                } else {
                    cur[at] += o;
                }
            }
            if (r >= r1 - keep) rg[(size_t) ((pos0 + (r - r0)) % ring.slots) * ring.width + j.ch0 + c] = xt;
            h[0] = h[1];
            h[1] = h[2];
            h[2] = xt;
        }
    }
}

template<bool SUM, bool RESID>
__global__ void __launch_bounds__(kConvThreads) conv_rows_kernel(ConvJob a, ConvJob b, int hw, ConvRing ring, Members mem,
                                                                 const int64_t* __restrict__ pos, const float* __restrict__ eout,
                                                                 const float* __restrict__ ew, int k, float* cur, float* save) {
    conv_rows_stage<SUM, RESID, false>(blockIdx.x * blockDim.x + threadIdx.x, 0, a, b, hw, ring, mem, pos, eout, ew, k, cur, save);
}
template<bool SUM, bool RESID>
__global__ void __launch_bounds__(kConvChannels * kConvSlices) conv_prompt_kernel(ConvJob a, ConvJob b, int hw, ConvRing ring,
                                                                                 Members mem, const int64_t* pos, const float* eout,
                                                                                 const float* ew, int k, float* cur, float* save) {
    conv_rows_stage<SUM, RESID, true>(blockIdx.x * kConvChannels + threadIdx.x % kConvChannels, threadIdx.x / kConvChannels,
                                      a, b, hw, ring, mem, pos, eout, ew, k, cur, save);
}

// ------------------------------------------------------------------------------------------ the attention's inputs
// blockIdx.y: n_head Q heads, n_kv K heads, n_kv V heads, then n_head R heads; 128 threads.
__global__ void qkvr_prep_kernel(float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ rr,
                                 const float* __restrict__ q_scale, const float* __restrict__ r_scale, PrepArgs a, Members mem,
                                 const int64_t* __restrict__ pos, KvCaches caches, long long cells) {
    const int row = blockIdx.x, seg = blockIdx.y, d = threadIdx.x;
    __shared__ float red[32];
    const long long p = pos[row];
    const int mi = member_of(mem, row);
    __half* kc = (__half*) caches.k[mi];
    __half* vc = (__half*) caches.v[mi];
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

__global__ void cap_logits_kernel(const float* __restrict__ x, int ld, int n_vocab, float cap, float* __restrict__ y) {
    const int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vocab) return;
    y[(size_t) r * n_vocab + i] = cap * tanhf(x[(size_t) r * ld + i] * (1.0f / cap));
}

}  // namespace

void set_positions(int64_t* pos, const Members& m, const int64_t* first, void* stream) {
    if (m.rows() <= 0) return;
    Firsts f{};
    for (int i = 0; i < m.n && i < kMaxMembers; ++i) f.p[i] = (long long) first[i];
    set_positions_kernel<<<(unsigned) ((m.rows() + 127) / 128), 128, 0, (cudaStream_t) stream>>>(pos, m, f);
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

void norm_quant(const NormJob* jobs, int n, int rows, int n_embd, float eps, float* out, void* xq, void* stream) {
    if (rows <= 0 || n <= 0) return;
    NormJobs a{};
    for (int i = 0; i < n && i < 3; ++i) a.j[i] = jobs[i];
    norm_quant_kernel<<<dim3((unsigned) rows, (unsigned) std::min(n, 3)), kNormThreads, 0, (cudaStream_t) stream>>>(
        a, rows, n_embd, eps, out, (Q81*) xq);
}

void loop_control(const CtlJob* jobs, int n, int rows, int n_embd, int rank, float eps, float* g, void* stream) {
    if (rows <= 0 || n <= 0) return;
    CtlJobs a{};
    for (int i = 0; i < n && i < 3; ++i) a.j[i] = jobs[i];
    const unsigned nz = (unsigned) std::min(n, 3);
    cudaStream_t st = (cudaStream_t) stream;
    control_down_kernel<<<dim3((unsigned) rank, (unsigned) rows, nz), kCtlThreads, 0, st>>>(a, rows, n_embd, rank, eps, g);
    control_out_kernel<<<dim3((unsigned) ((n_embd + kCtlThreads / 32 - 1) / (kCtlThreads / 32)), (unsigned) rows, nz), kCtlThreads, 0, st>>>(
        a, rows, n_embd, rank, g);
}

template<bool SUM, bool RESID>
static void launch_conv(const ConvJob& a, const ConvJob& b, int hw, const ConvRing& ring, const Members& m, const int64_t* pos,
                        const float* eout, const float* ew, int k, float* cur, float* save, cudaStream_t stream) {
    const int C = a.C + b.C;
    if (m.n == 1 && m.rows() > kMaxMembers)
        conv_prompt_kernel<SUM, RESID><<<(C + kConvChannels - 1) / kConvChannels, kConvChannels * kConvSlices, 0, stream>>>(
            a, b, hw, ring, m, pos, eout, ew, k, cur, save);
    else
        conv_rows_kernel<SUM, RESID><<<(C + kConvThreads - 1) / kConvThreads, kConvThreads, 0, stream>>>(
            a, b, hw, ring, m, pos, eout, ew, k, cur, save);
}

void conv_rows(const ConvJob& a, const ConvJob* b, int head_width, const ConvRing& ring, const Members& m, const int64_t* pos,
               void* stream) {
    if (m.rows() <= 0) return;
    const ConvJob none;
    launch_conv<false, false>(a, b ? *b : none, head_width, ring, m, pos, nullptr, nullptr, 0, nullptr, nullptr, (cudaStream_t) stream);
}

void conv_residual(const ConvJob& a, const float* eout, const float* ew, int k, int head_width, const ConvRing& ring,
                   const Members& m, const int64_t* pos, float* cur, float* save, void* stream) {
    if (m.rows() <= 0) return;
    const ConvJob none;
    if (eout) launch_conv<true, true>(a, none, head_width, ring, m, pos, eout, ew, k, cur, save, (cudaStream_t) stream);
    else launch_conv<false, true>(a, none, head_width, ring, m, pos, nullptr, nullptr, 0, cur, save, (cudaStream_t) stream);
}

void qkvr_prep(float* q, const float* k, const float* v, float* r, const float* q_scale, const float* r_scale, const PrepArgs& a,
               const Members& m, const int64_t* pos, const KvCaches& caches, int64_t cells, void* stream) {
    if (m.rows() <= 0) return;
    qkvr_prep_kernel<<<dim3((unsigned) m.rows(), (unsigned) (2 * a.n_head + 2 * a.n_kv)), 128, 0, (cudaStream_t) stream>>>(
        q, k, v, r, q_scale, r_scale, a, m, pos, caches, cells);
}

void group_experts(const int32_t* ids, int rows, int k, int n_expert, const unsigned long long* expert_ptr, const Exl3Rates* expert_rates,
                   unsigned long long* grp_ptr, Exl3Rates* grp_rates, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok,
                   void* stream) {
    group_experts_kernel<<<1, (unsigned) ((n_expert + 31) / 32 * 32), 0, (cudaStream_t) stream>>>(
        ids, rows * k, n_expert, expert_ptr, expert_rates, grp_ptr, grp_rates, grp_start, n_groups, ent_dst, ent_tok, k);
}

void cap_logits(const float* x, int ld, int rows, int n_vocab, float cap, float* logits, void* stream) {
    if (rows <= 0) return;
    cap_logits_kernel<<<dim3((unsigned) ((n_vocab + 255) / 256), (unsigned) rows), 256, 0, (cudaStream_t) stream>>>(x, ld, n_vocab, cap, logits);
}

}  // namespace basalt::models::inkling2x::kernels
