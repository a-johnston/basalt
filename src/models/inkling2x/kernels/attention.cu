// src/models/inkling2x/kernels/attention.cu - Inkling 2's attention: q . k / sqrt(128) plus the relative-position bias of
// each head's query (unscaled), the sink as one more logit, a window on the sliding layers, the output gate. A block of four warps takes kSplit keys of one row
// and head, a warp a key at a time with the head's 128 values over its lanes; a second kernel joins the splits.
#include "basalt/models/inkling2x/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace basalt::models::inkling2x::kernels {
namespace {

constexpr int kSplit = 256;   // keys a block takes
constexpr int kWarps = 4;
constexpr int kHead = 128;    // values per head

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ int64_t key_lo(int64_t p, int window) { return window > 0 && p >= window ? p - window + 1 : 0; }

// ws: m and l of each [row][head][split], then its 128 accumulators
__global__ void __launch_bounds__(kWarps * 32) attn_split_kernel(const float* __restrict__ q, const float* __restrict__ r,
                                                                 const float* __restrict__ sinks, const float* __restrict__ rel_bias,
                                                                 const __half* __restrict__ kc, const __half* __restrict__ vc,
                                                                 AttnArgs a, long long pos0, int rows, int nsplit,
                                                                 float* __restrict__ ws) {
    const int row = blockIdx.x, h = blockIdx.y, split = blockIdx.z;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ float tab[kSplit];   // the bias minus the sink by distance, over this split's distances (capped at extent)
    __shared__ float wm[kWarps], wl[kWarps], wacc[kWarps][kHead];
    const long long p = pos0 + row;
    const long long lo = key_lo(p, a.window) + (long long) split * kSplit;
    const long long hi = lo + kSplit - 1 < p ? lo + kSplit - 1 : p;
    const size_t slot = ((size_t) row * a.n_head + h) * nsplit + split;
    float* wsm = ws, *wsl = ws + (size_t) rows * a.n_head * nsplit, *wsa = ws + 2 * (size_t) rows * a.n_head * nsplit;
    if (lo > hi) {
        if (threadIdx.x == 0) { wsm[slot] = -INFINITY; wsl[slot] = 0.0f; }
        if (threadIdx.x < kHead) wsa[slot * kHead + threadIdx.x] = 0.0f;
        return;
    }
    const float* rq = r + ((size_t) row * a.n_head + h) * a.d_rel;
    const float sink = sinks[(size_t) row * a.n_head + h];
    const long long t0 = p - hi < a.extent ? p - hi : a.extent, t1 = p - lo < a.extent ? p - lo : a.extent;   // at most kSplit apart
    for (int i = threadIdx.x; i <= t1 - t0; i += blockDim.x) {
        const long long e = t0 + i;
        float t = 0.0f;
        if (e < a.extent)
            for (int j = 0; j < a.d_rel; ++j) t += rel_bias[(size_t) e * a.d_rel + j] * rq[j];
        tab[i] = t - sink;
    }
    const int g = h / (a.n_head / a.n_kv);
    const float* qh = q + ((size_t) row * a.n_head + h) * kHead;
    float qv[4];
    for (int i = 0; i < 4; ++i) qv[i] = qh[lane * 4 + i];
    __syncthreads();
    const float scale = 1.0f / sqrtf((float) kHead);
    float m = -INFINITY, l = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (long long j = lo + warp; j <= hi; j += kWarps) {
        const size_t cell = (size_t) (j % a.cells);
        const __half* kr = kc + (cell * a.n_kv + g) * kHead + lane * 4;
        const __half2 k01 = *(const __half2*) kr, k23 = *(const __half2*) (kr + 2);
        float dot = qv[0] * __low2float(k01) + qv[1] * __high2float(k01) + qv[2] * __low2float(k23) + qv[3] * __high2float(k23);
        dot = warp_sum(dot);
        const long long d = p - j;
        const float s = dot * scale + tab[(d < a.extent ? d : a.extent) - t0];
        const float mn = fmaxf(m, s);
        const float c = expf(m - mn), e = expf(s - mn);
        const __half* vr = vc + (cell * a.n_kv + g) * kHead + lane * 4;
        const __half2 v01 = *(const __half2*) vr, v23 = *(const __half2*) (vr + 2);
        acc[0] = acc[0] * c + e * __low2float(v01);
        acc[1] = acc[1] * c + e * __high2float(v01);
        acc[2] = acc[2] * c + e * __low2float(v23);
        acc[3] = acc[3] * c + e * __high2float(v23);
        l = l * c + e;
        m = mn;
    }
    if (lane == 0) { wm[warp] = m; wl[warp] = l; }
    for (int i = 0; i < 4; ++i) wacc[warp][lane * 4 + i] = acc[i];
    __syncthreads();
    if (threadIdx.x < kHead) {
        float bm = -INFINITY;
        for (int w = 0; w < kWarps; ++w) bm = fmaxf(bm, wm[w]);
        float bl = 0.0f, ba = 0.0f;
        for (int w = 0; w < kWarps; ++w) {
            const float c = wm[w] == -INFINITY ? 0.0f : expf(wm[w] - bm);
            bl += wl[w] * c;
            ba += wacc[w][threadIdx.x] * c;
        }
        wsa[slot * kHead + threadIdx.x] = ba;
        if (threadIdx.x == 0) { wsm[slot] = bm; wsl[slot] = bl; }
    }
}

__global__ void attn_join_kernel(const float* __restrict__ ws, const float* __restrict__ gate, int n_head, int rows, int nsplit,
                                 float* __restrict__ out) {
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
    out[at] = (o / l) * (1.0f / (1.0f + expf(-gate[at])));
}

int splits_of(int64_t keys) { return (int) ((keys + kSplit - 1) / kSplit); }

}  // namespace

size_t attention_workspace_bytes(int rows, int n_head, int64_t max_keys) {
    return (size_t) rows * n_head * splits_of(max_keys) * (2 + kHead) * sizeof(float);
}

void attention(const float* q, const float* r, const float* sinks, const float* gate, const float* rel_bias, const uint16_t* k_cache,
               const uint16_t* v_cache, const AttnArgs& a, int64_t pos0, int rows, void* ws, float* out, void* stream) {
    if (rows <= 0) return;
    const int64_t p_last = pos0 + rows - 1;
    const int64_t keys = a.window > 0 && p_last + 1 > a.window ? a.window : p_last + 1;
    const int nsplit = splits_of(keys);
    cudaStream_t st = (cudaStream_t) stream;
    attn_split_kernel<<<dim3((unsigned) rows, (unsigned) a.n_head, (unsigned) nsplit), kWarps * 32, 0, st>>>(q, r, sinks, rel_bias, (const __half*) k_cache, (const __half*) v_cache, a, pos0, rows, nsplit,
                              (float*) ws);
    attn_join_kernel<<<dim3((unsigned) rows, (unsigned) a.n_head), kHead, 0, st>>>((const float*) ws, gate, a.n_head, rows, nsplit, out);
}

}  // namespace basalt::models::inkling2x::kernels
