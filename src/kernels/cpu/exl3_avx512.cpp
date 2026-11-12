// src/kernels/cpu/exl3_avx512.cpp - the EXL3 rows on AVX-512 (F, BW, VL, VNNI, VBMI).  See exl3_cpu.hpp for the format
// and the arithmetic; this file computes the same integers and the same float chain as the scalar kernel.
//
// One pass per block of 128 inputs (8 tiles) and group of 32 steps: a tile's group g is its 16 inputs against
// outputs g and g + 8.  Sixteen states come from one 16-word load of the tile (a copy that starts with the tile's
// last word, so the tail-biting windows need no branch): a dword permute puts the two words that hold a pair of
// neighbouring steps in each 64-bit lane, and one masked vpmultishiftqb takes both 16-bit states out of it; the byte
// sum of state * 0x83DCD12D is one vpdpbusd; two halves pack to 32 int16 sums, and one vpdpwssd per token
// multiplies them with the activation in the same order.  A block's eight groups are summed across lanes at once,
// and 510 times the block's activation sum comes off then.  128-bit lanes 0 and 2 of the pair sums belong to output
// g, lanes 1 and 3 to output g + 8.
#include "basalt/kernels/cpu/exl3_cpu.hpp"

#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace basalt::kernels::cpu::exl3_detail {
namespace {

inline __m512i states16(const uint32_t* buf, const Chunk& c) {
    const __m512i wz = _mm512_loadu_si512(buf + c.base);
    const __m512i w2 = _mm512_permutexvar_epi32(_mm512_load_si512(c.pair), wz);
    return _mm512_maskz_multishift_epi64_epi8(0x3333333333333333ull, _mm512_load_si512(c.ctrl), w2);
}

// bytesum(state * mul1) of 16 states, 0 .. 1020, as int32 (the -510 of the weight is taken off per block)
inline __m512i bytesums16(__m512i st) {
    const __m512i x = _mm512_mullo_epi32(st, _mm512_set1_epi32((int) kExl3Mul1));
    return _mm512_dpbusd_epi32(_mm512_setzero_si512(), x, _mm512_set1_epi32(0x01010101));
}

// The 32 int16 bytesums of tile `buf` for group g: steps 32 g .. 32 g + 31, in the activation's slot order.
inline __m512i group32(const uint32_t* buf, const Tables& tb, int g) {
    return _mm512_packs_epi32(bytesums16(states16(buf, tb.c[2 * g])), bytesums16(states16(buf, tb.c[2 * g + 1])));
}

// The 16 sums of a block from its eight groups' accumulators: lanes 0..3 and 8..11 of acc[g] (128-bit lanes 0 and
// 2) are output g's, the others output g + 8's.  Integer adds, so any order gives the same sum.
inline __m512i reduce8(const __m512i (&acc)[8]) {
    __m512i c[2];
    for (int h = 0; h < 2; ++h) {
        const __m512i* x = acc + 4 * h;
        const __m512i b01 = _mm512_add_epi32(_mm512_unpacklo_epi32(x[0], x[1]), _mm512_unpackhi_epi32(x[0], x[1]));
        const __m512i b23 = _mm512_add_epi32(_mm512_unpacklo_epi32(x[2], x[3]), _mm512_unpackhi_epi32(x[2], x[3]));
        const __m512i s = _mm512_add_epi32(_mm512_unpacklo_epi64(b01, b23), _mm512_unpackhi_epi64(b01, b23));
        // 128-bit lane j: the four groups' sums over lane j; lanes 0 + 2 are outputs 4 h .., lanes 1 + 3 outputs 8 + 4 h ..
        c[h] = _mm512_add_epi32(s, _mm512_shuffle_i64x2(s, s, 0x4E));
    }
    return _mm512_permutex2var_epi64(c[0], _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11), c[1]);
}

// hadamard_rotate's butterflies on one block of 128 floats in eight registers: the same additions and
// subtractions in the same order, then the same scale
inline void fwht128(__m512 r[8]) {
    const __m512i i1 = _mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
    const __m512i i2 = _mm512_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
    const __m512i i4 = _mm512_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11);
    const __m512i i8 = _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
    for (int i = 0; i < 8; ++i) {
        __m512 v = r[i], b;
        b = _mm512_permutexvar_ps(i1, v);
        v = _mm512_mask_blend_ps(0xAAAA, _mm512_add_ps(v, b), _mm512_sub_ps(b, v));
        b = _mm512_permutexvar_ps(i2, v);
        v = _mm512_mask_blend_ps(0xCCCC, _mm512_add_ps(v, b), _mm512_sub_ps(b, v));
        b = _mm512_permutexvar_ps(i4, v);
        v = _mm512_mask_blend_ps(0xF0F0, _mm512_add_ps(v, b), _mm512_sub_ps(b, v));
        b = _mm512_permutexvar_ps(i8, v);
        r[i] = _mm512_mask_blend_ps(0xFF00, _mm512_add_ps(v, b), _mm512_sub_ps(b, v));
    }
    for (int len = 1; len < 8; len <<= 1)
        for (int i = 0; i < 8; i += len << 1)
            for (int j = i; j < i + len; ++j) {
                const __m512 a = r[j], c = r[j + len];
                r[j] = _mm512_add_ps(a, c);
                r[j + len] = _mm512_sub_ps(a, c);
            }
    const __m512 scale = _mm512_set1_ps(1.0f / std::sqrt((float) kExl3Had));
    for (int i = 0; i < 8; ++i) r[i] = _mm512_mul_ps(r[i], scale);
}

inline __m512 f16x16(const uint8_t* p) { return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) p)); }

}  // namespace

void prep_avx512(const float* x, const uint8_t* suh_f16, int n, Exl3Act& a) {
    alignas(64) uint16_t slot[32];
    for (int s = 0; s < 32; ++s) slot[s] = kExl3Slot[s];
    const __m512i slots = _mm512_load_si512(slot);
    float sum = 0.0f;
    for (int b = 0; b < n / kExl3Had; ++b) {
        __m512 r[8];
        for (int i = 0; i < 8; ++i)
            r[i] = _mm512_mul_ps(_mm512_loadu_ps(x + b * kExl3Had + 16 * i), f16x16(suh_f16 + 2 * (b * kExl3Had + 16 * i)));
        fwht128(r);
        __m512 m = _mm512_abs_ps(r[0]);
        for (int i = 1; i < 8; ++i) m = _mm512_max_ps(m, _mm512_abs_ps(r[i]));
        const float amax = _mm512_reduce_max_ps(m);
        const __m512 inv = _mm512_set1_ps(amax > 0.0f ? 32767.0f / amax : 0.0f);
        a.d[b] = amax / 32767.0f;
        __m512i qs = _mm512_setzero_si512();
        for (int i = 0; i < 8; ++i) {
            const __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(r[i], inv));
            qs = _mm512_add_epi32(qs, q);
            const __m512i q16 = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(q));
            _mm512_store_si512(a.qp[8 * b + i], _mm512_permutexvar_epi16(slots, q16));
        }
        a.qs[b] = _mm512_reduce_add_epi32(qs);
        sum = std::fma((float) a.qs[b], a.d[b], sum);
    }
    a.off = sum * kExl3Offset;
}

void finish_avx512(float* v, const uint8_t* svh_f16, int r0, int r1) {
    for (int b = r0; b < r1; b += kExl3Had) {
        __m512 r[8];
        for (int i = 0; i < 8; ++i) r[i] = _mm512_loadu_ps(v + b + 16 * i);
        fwht128(r);
        for (int i = 0; i < 8; ++i) _mm512_storeu_ps(v + b + 16 * i, _mm512_mul_ps(r[i], f16x16(svh_f16 + 2 * (b + 16 * i))));
    }
}

// GS groups at a time, so the GS x NT accumulators and the GS weight vectors fit the registers.
template<int NT, int GS>
void rows_nt(int k2, const uint8_t* w, int n_in, const Exl3Act* const* a, float* const* out, int r0, int r1) {
    const Tables& tb = tables(k2);
    const int kt_n = n_in / 16, nw = tb.words;
    const size_t tile = (size_t) 4 * nw;
    constexpr int kBuf = 64 + 16;   // a tile of up to 64 words after its last word, and the load's reach
    alignas(64) uint32_t buf[8][kBuf] = {};
    for (int band = r0 / 16; band < r1 / 16; ++band) {
        const uint8_t* wb = w + (size_t) band * kt_n * tile;
        __m512 facc[NT];
        for (int t = 0; t < NT; ++t) facc[t] = _mm512_setzero_ps();
        for (int b = 0; b < kt_n / 8; ++b) {
            for (int i = 0; i < 8; ++i) {
                const uint8_t* tp = wb + (size_t) (8 * b + i) * tile;
                std::memcpy(&buf[i][0], tp + tile - 4, 4);
                std::memcpy(&buf[i][1], tp, tile);
            }
            __m512i blk[NT][8];
            for (int g0 = 0; g0 < 8; g0 += GS) {
                __m512i acc[NT][GS];
                for (int t = 0; t < NT; ++t)
                    for (int g = 0; g < GS; ++g) acc[t][g] = _mm512_setzero_si512();
                for (int i = 0; i < 8; ++i) {
                    __m512i w16[GS];
                    for (int g = 0; g < GS; ++g) w16[g] = group32(buf[i], tb, g0 + g);
                    for (int t = 0; t < NT; ++t) {
                        const __m512i q = _mm512_load_si512(a[t]->qp[8 * b + i]);
                        for (int g = 0; g < GS; ++g) acc[t][g] = _mm512_dpwssd_epi32(acc[t][g], w16[g], q);
                    }
                }
                for (int t = 0; t < NT; ++t)
                    for (int g = 0; g < GS; ++g) blk[t][g0 + g] = acc[t][g];
            }
            for (int t = 0; t < NT; ++t) {   // sum (S - 510) q = sum S q - 510 sum q, exact in int32 (modulo 2^32)
                const __m512i is = _mm512_sub_epi32(reduce8(blk[t]), _mm512_set1_epi32(510 * a[t]->qs[b]));
                facc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), _mm512_set1_ps(a[t]->d[b]), facc[t]);
            }
        }
        for (int t = 0; t < NT; ++t)
            _mm512_storeu_ps(out[t] + 16 * band, _mm512_fmadd_ps(facc[t], _mm512_set1_ps(kExl3KInv), _mm512_set1_ps(a[t]->off)));
    }
}

void rows_avx512(int k2, const uint8_t* w, int n_in, const Exl3Act* const* a, int nt, float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: rows_nt<1, 8>(k2, w, n_in, a, out, r0, r1); break;
        case 2: rows_nt<2, 8>(k2, w, n_in, a, out, r0, r1); break;
        case 3: rows_nt<3, 4>(k2, w, n_in, a, out, r0, r1); break;
        case 4: rows_nt<4, 4>(k2, w, n_in, a, out, r0, r1); break;
        case 5: rows_nt<5, 2>(k2, w, n_in, a, out, r0, r1); break;
        case 6: rows_nt<6, 2>(k2, w, n_in, a, out, r0, r1); break;
        case 7: rows_nt<7, 2>(k2, w, n_in, a, out, r0, r1); break;
        default: rows_nt<8, 2>(k2, w, n_in, a, out, r0, r1); break;
    }
}

}  // namespace basalt::kernels::cpu::exl3_detail
