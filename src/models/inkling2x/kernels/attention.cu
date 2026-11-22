// src/models/inkling2x/kernels/attention.cu - Inkling 2's attention: q . k / sqrt(128) plus the relative-position bias of
// each head's query (unscaled), the sink as one more logit, a window on the sliding layers, the output gate. A block of four
// warps takes a split of the keys of one row and one K/V head for all of that head's queries (the K and V rows read once);
// a warp takes four keys at a time, eight lanes a key with 16 of its values each. A second kernel joins the splits.
#include "basalt/models/inkling2x/kernels.hpp"
#include "basalt/log.hpp"
#include "q8_1.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>

namespace basalt::models::inkling2x::kernels {
namespace {

constexpr int kDecodeRows = 16;    // a launch of up to this many rows splits its keys in runs of kSplitDecode
constexpr int kSplitDecode = 64;   // so a one-row decode spreads over the card
constexpr int kSplitPrompt = 256;  // a prompt chunk's runs: its workspace grows with rows x splits
constexpr int kWarps = 4;
constexpr int kHead = 128;         // values per head
constexpr int kLanes = 8;          // lanes a key
constexpr int kPer = kHead / kLanes;
constexpr int kKeys = 32 / kLanes;   // keys a warp takes at once

__device__ __forceinline__ int64_t key_lo(int64_t p, int window) { return window > 0 && p >= window ? p - window + 1 : 0; }

__device__ __forceinline__ void unpack16(uint4 a, uint4 b, float (&f)[kPer]) {
    const __half2* ha = (const __half2*) &a;
    const __half2* hb = (const __half2*) &b;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 x = __half22float2(ha[i]), y = __half22float2(hb[i]);
        f[2 * i] = x.x;
        f[2 * i + 1] = x.y;
        f[8 + 2 * i] = y.x;
        f[8 + 2 * i + 1] = y.y;
    }
}

// ws: m and l of each [row][head][split], then its 128 accumulators.  Grid (rows, n_kv, nsplit); G queries a K/V head.
template<int G>
__global__ void __launch_bounds__(kWarps * 32) attn_split_kernel(const float* __restrict__ q, const float* __restrict__ r,
                                                                 const float* __restrict__ sinks, const float* __restrict__ rel_bias,
                                                                 const __half* __restrict__ kc, const __half* __restrict__ vc,
                                                                 AttnArgs a, const int64_t* __restrict__ pos, int rows, int nsplit,
                                                                 int split, float* __restrict__ ws) {
    const int row = blockIdx.x, g = blockIdx.y, sp = blockIdx.z;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, kg = lane / kLanes, sub = lane % kLanes;
    __shared__ float tab[G][kSplitPrompt];   // the bias minus the sink by distance over this split's distances (capped at extent)
    __shared__ float wm[kWarps][G], wl[kWarps][G], wacc[kWarps][G][kHead];
    const long long p = *pos + row;
    const long long lo = key_lo(p, a.window) + (long long) sp * split;
    const long long hi = lo + split - 1 < p ? lo + split - 1 : p;
    float* wsm = ws, *wsl = ws + (size_t) rows * a.n_head * nsplit, *wsa = ws + 2 * (size_t) rows * a.n_head * nsplit;
    const size_t slot0 = ((size_t) row * a.n_head + (size_t) g * G) * nsplit + sp;   // head g G + hh at slot0 + hh nsplit
    if (lo > hi) {
        for (int i = threadIdx.x; i < G * kHead; i += blockDim.x) {
            const size_t slot = slot0 + (size_t) (i / kHead) * nsplit;
            wsa[slot * kHead + i % kHead] = 0.0f;
            if (i % kHead == 0) {
                wsm[slot] = -INFINITY;
                wsl[slot] = 0.0f;
            }
        }
        return;
    }
    const long long t0 = p - hi < a.extent ? p - hi : a.extent, t1 = p - lo < a.extent ? p - lo : a.extent;
    const int nt = (int) (t1 - t0) + 1;   // at most split
    for (int i = threadIdx.x; i < G * nt; i += blockDim.x) {
        const int hh = i / nt, h = g * G + hh;
        const long long e = t0 + i % nt;
        const float* rq = r + ((size_t) row * a.n_head + h) * a.d_rel;
        float t = 0.0f;
        if (e < a.extent)
            for (int j = 0; j < a.d_rel; ++j) t += rel_bias[(size_t) e * a.d_rel + j] * rq[j];
        tab[hh][i % nt] = t - sinks[(size_t) row * a.n_head + h];
    }
    float qv[G][kPer];
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
        const float4* qh = (const float4*) (q + ((size_t) row * a.n_head + g * G + hh) * kHead + sub * kPer);
#pragma unroll
        for (int i = 0; i < kPer / 4; ++i) {
            const float4 v = qh[i];
            qv[hh][4 * i] = v.x;
            qv[hh][4 * i + 1] = v.y;
            qv[hh][4 * i + 2] = v.z;
            qv[hh][4 * i + 3] = v.w;
        }
    }
    __syncthreads();
    const float scale = 1.0f / sqrtf((float) kHead);
    const unsigned gmask = 0xffu << (kLanes * kg);   // the lanes of this key: the same keys, the same trips
    float m[G], l[G], acc[G][kPer];
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
        m[hh] = -INFINITY;
        l[hh] = 0.0f;
#pragma unroll
        for (int i = 0; i < kPer; ++i) acc[hh][i] = 0.0f;
    }
    for (long long j = lo + warp * kKeys + kg; j <= hi; j += kWarps * kKeys) {
        const size_t at = ((size_t) (j % a.cells) * a.n_kv + g) * kHead + sub * kPer;
        const uint4* kr = (const uint4*) (kc + at);
        const uint4* vr = (const uint4*) (vc + at);
        float kf[kPer], vf[kPer];
        unpack16(kr[0], kr[1], kf);
        unpack16(vr[0], vr[1], vf);
        const long long d = p - j;
        const int ti = (int) ((d < a.extent ? d : a.extent) - t0);
#pragma unroll
        for (int hh = 0; hh < G; ++hh) {
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < kPer; ++i) dot += qv[hh][i] * kf[i];
#pragma unroll
            for (int o = 1; o < kLanes; o <<= 1) dot += __shfl_xor_sync(gmask, dot, o);
            const float s = dot * scale + tab[hh][ti];
            const float mn = fmaxf(m[hh], s);
            const float c = expf(m[hh] - mn), e = expf(s - mn);
#pragma unroll
            for (int i = 0; i < kPer; ++i) acc[hh][i] = acc[hh][i] * c + e * vf[i];
            l[hh] = l[hh] * c + e;
            m[hh] = mn;
        }
    }
    // the warp's four keys' runs joined, then the warps'
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
        float mx = m[hh];
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
        const float c = m[hh] == -INFINITY ? 0.0f : expf(m[hh] - mx);
        float lt = l[hh] * c;
        lt += __shfl_xor_sync(0xffffffffu, lt, 8);
        lt += __shfl_xor_sync(0xffffffffu, lt, 16);
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            float v = acc[hh][i] * c;
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 16);
            if (kg == 0) wacc[warp][hh][sub * kPer + i] = v;
        }
        if (lane == 0) {
            wm[warp][hh] = mx;
            wl[warp][hh] = lt;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < G * kHead; i += blockDim.x) {
        const int hh = i / kHead, dd = i % kHead;
        float bm = -INFINITY;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) bm = fmaxf(bm, wm[w][hh]);
        float bl = 0.0f, ba = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            const float c = wm[w][hh] == -INFINITY ? 0.0f : expf(wm[w][hh] - bm);
            bl += wl[w][hh] * c;
            ba += wacc[w][hh][dd] * c;
        }
        const size_t slot = slot0 + (size_t) hh * nsplit;
        wsa[slot * kHead + dd] = ba;
        if (dd == 0) {
            wsm[slot] = bm;
            wsl[slot] = bl;
        }
    }
}

// out, and with xq set its q8_1 blocks (a head's 128 values are four blocks, a warp each)
__global__ void attn_join_kernel(const float* __restrict__ ws, const float* __restrict__ gate, int n_head, int rows, int nsplit,
                                 float* __restrict__ out, Q81* __restrict__ xq) {
    const int row = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const size_t base = ((size_t) row * n_head + h) * nsplit;
    const float* wsm = ws, *wsl = ws + (size_t) rows * n_head * nsplit, *wsa = ws + 2 * (size_t) rows * n_head * nsplit;
    float m = 0.0f;   // the sink's logit
    for (int s = 0; s < nsplit; ++s) m = fmaxf(m, wsm[base + s]);
    float l = expf(0.0f - m), o = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        if (wsm[base + s] == -INFINITY) continue;
        const float c = expf(wsm[base + s] - m);
        l += wsl[base + s] * c;
        o += wsa[(base + s) * kHead + d] * c;
    }
    const size_t at = ((size_t) row * n_head + h) * kHead + d;
    const float v = (o / l) * (1.0f / (1.0f + expf(-gate[at])));
    out[at] = v;
    if (xq) q8_1_value(v, xq, (long long) at);
}

int split_of(int rows) { return rows <= kDecodeRows ? kSplitDecode : kSplitPrompt; }
int splits_of(int64_t keys, int split) { return (int) ((keys + split - 1) / split); }
int64_t keys_of(const AttnArgs& a, int64_t p_last) { return a.window > 0 && p_last + 1 > a.window ? a.window : p_last + 1; }

}  // namespace

int attention_splits(const AttnArgs& a, int64_t p_last, int rows) { return splits_of(keys_of(a, p_last), split_of(rows)); }

size_t attention_workspace_bytes(int rows, int n_head, int64_t max_keys) {
    auto need = [&](int r) { return (size_t) r * n_head * splits_of(max_keys, split_of(r)) * (2 + kHead) * sizeof(float); };
    const size_t a = need(rows), b = need(rows < kDecodeRows ? rows : kDecodeRows);
    return a > b ? a : b;
}

void attention(const float* q, const float* r, const float* sinks, const float* gate, const float* rel_bias, const uint16_t* k_cache,
               const uint16_t* v_cache, const AttnArgs& a, const int64_t* pos, int64_t pos0, int rows, void* ws, float* out,
               void* out_q8_1, void* stream) {
    if (rows <= 0) return;
    const int split = split_of(rows), nsplit = attention_splits(a, pos0 + rows - 1, rows);
    cudaStream_t st = (cudaStream_t) stream;
    const dim3 grid((unsigned) rows, (unsigned) a.n_kv, (unsigned) nsplit);
    const __half *kc = (const __half*) k_cache, *vc = (const __half*) v_cache;
    switch (a.n_head / a.n_kv) {
#define X(G) case G: attn_split_kernel<G><<<grid, kWarps * 32, 0, st>>>(q, r, sinks, rel_bias, kc, vc, a, pos, rows, nsplit, split, (float*) ws); break;
        X(1) X(2) X(3) X(4) X(6) X(8)
#undef X
        default: basalt::log::error("basalt inkling2x", "attention: %d query heads a K/V head", a.n_head / a.n_kv); std::exit(1);
    }
    attn_join_kernel<<<dim3((unsigned) rows, (unsigned) a.n_head), kHead, 0, st>>>((const float*) ws, gate, a.n_head, rows, nsplit, out,
                                                                                (Q81*) out_q8_1);
}

}  // namespace basalt::models::inkling2x::kernels
