// A full prompt split older than every row's bias extent. Four warps tile the
// key columns of Q K^T, then the output columns of P V. The join consumes
// unnormalized partials, with the sink subtracted from the partial maximum.
#pragma once
#include <mma.h>

namespace basalt::models::inkling2x::kernels {
namespace {
constexpr int kHistory = 256;
struct HistoryTile {
    __align__(32) __half q[16][128];
    __align__(32) float scores[16][kHistory];
    __align__(32) __half weights[16][kHistory];
    __align__(32) float out[16][128];
};

__device__ void history_stage(HistoryTile& t, int row0, int head, int sp, const float* q, const float* sinks,
                              const __half* kc, const __half* vc, int rows, int heads, int kv, int splits, int64_t first, int extent, float* ws) {
    namespace w = nvcuda::wmma;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int stride = kv * 128;
    const int64_t cutoff = (int64_t) (sp + 1) * kHistory - 1 + extent;
    if (first + row0 + 15 < cutoff) return;
    const size_t kvbase = ((size_t) sp * kHistory * kv + head / (heads / kv)) * 128;
    for (int i = tid; i < 16 * 128; i += 128) {
        const int row = row0 + i / 128;
        t.q[i / 128][i % 128] = row < rows ? __float2half(q[((size_t) row * heads + head) * 128 + i % 128]) : __float2half(0.f);
    }
    __syncthreads();
    for (int j = warp * 16; j < kHistory; j += 64) {
        w::fragment<w::accumulator,16,16,16,float> s;
        w::fill_fragment(s, 0.f);
        for (int d = 0; d < 128; d += 16) {
            w::fragment<w::matrix_a,16,16,16,__half,w::row_major> a;
            w::fragment<w::matrix_b,16,16,16,__half,w::col_major> b;
            w::load_matrix_sync(a, &t.q[0][d], 128);
            w::load_matrix_sync(b, kc + kvbase + j * stride + d, stride);
            w::mma_sync(s, a, b, s);
        }
        w::store_matrix_sync(&t.scores[0][j], s, kHistory, w::mem_row_major);
    }
    __syncthreads();
    const size_t slots = (size_t) rows * heads * splits;
    for (int i = warp; i < 16; i += 4) {
        float mx = -INFINITY, sum = 0.f;
        for (int j = lane; j < kHistory; j += 32) mx = fmaxf(mx, t.scores[i][j]);
        for (int d = 16; d; d /= 2) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
        for (int j = lane; j < kHistory; j += 32) {
            const float e = expf((t.scores[i][j] - mx) * rsqrtf(128.f));
            t.weights[i][j] = __float2half(e);
            sum += e;
        }
        for (int d = 16; d; d /= 2) sum += __shfl_xor_sync(0xffffffff, sum, d);
        if (!lane && row0 + i < rows && first + row0 + i >= cutoff) {
            const size_t h = (size_t) (row0 + i) * heads + head, slot = h * splits + sp;
            ws[slot] = mx * rsqrtf(128.f) - (sinks ? sinks[h] : 0.f);
            ws[slots + slot] = sum;
        }
    }
    __syncthreads();
    for (int d = warp * 16; d < 128; d += 64) {
        w::fragment<w::accumulator,16,16,16,float> o;
        w::fill_fragment(o, 0.f);
        for (int j = 0; j < kHistory; j += 16) {
            w::fragment<w::matrix_a,16,16,16,__half,w::row_major> a;
            w::fragment<w::matrix_b,16,16,16,__half,w::row_major> b;
            w::load_matrix_sync(a, &t.weights[0][j], kHistory);
            w::load_matrix_sync(b, vc + kvbase + j * stride + d, stride);
            w::mma_sync(o, a, b, o);
        }
        w::store_matrix_sync(&t.out[0][d], o, 128, w::mem_row_major);
    }
    __syncthreads();
    for (int i = 0; i < 16 && row0 + i < rows; ++i) {
        if (first + row0 + i < cutoff) continue;
        const size_t slot = ((size_t) (row0 + i) * heads + head) * splits + sp;
        ws[2 * slots + slot * 128 + tid] = t.out[i][tid];
    }
}

}  // namespace
}  // namespace basalt::models::inkling2x::kernels
