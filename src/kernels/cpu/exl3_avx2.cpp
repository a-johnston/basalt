// src/kernels/cpu/exl3_avx2.cpp - the EXL3 rows on AVX2 + FMA + F16C.  See exl3_cpu.hpp for the format and the
// arithmetic; this file computes the same integers and the same float chain as the scalar kernel.
// Compiled for AVX2 only and reached behind cpu_avx2_ok(), so nothing here can fault on an older CPU.
//
// The AVX-512 kernel's scheme at half the width: per block of 128 inputs (8 tiles) and group of 32 steps, a tile's
// 16 inputs against outputs g and g + 8.  Eight states come from one 8-word load of the tile copy (the 16-step
// chunk tables reach at most five words past their base) with a permute and two variable shifts; the byte sum of
// state * 0x83DCD12D is a vpmaddubsw and a vpmaddwd against ones.  Four state vectors pack to the 32 int16 weights
// in the activation's slot order, as two registers: the low 128-bit lane of each belongs to output g, the high lane
// to output g + 8.
#include "basalt/kernels/cpu/exl3_cpu.hpp"

#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace basalt::kernels::cpu::exl3_detail {
namespace {

// states 8 h .. 8 h + 7 of a 16-step chunk
inline __m256i states8(const uint32_t* buf, const Chunk& c, int h) {
    const __m256i wz = _mm256_loadu_si256((const __m256i*) (buf + c.base));
    const __m256i hi = _mm256_load_si256((const __m256i*) (c.hi + 8 * h));
    const __m256i shr = _mm256_load_si256((const __m256i*) (c.shr + 8 * h));
    const __m256i hv = _mm256_permutevar8x32_epi32(wz, hi);
    const __m256i lv = _mm256_permutevar8x32_epi32(wz, _mm256_sub_epi32(hi, _mm256_set1_epi32(1)));
    const __m256i v = _mm256_or_si256(_mm256_srlv_epi32(hv, shr), _mm256_sllv_epi32(lv, _mm256_sub_epi32(_mm256_set1_epi32(32), shr)));
    return _mm256_and_si256(v, _mm256_set1_epi32(0xffff));
}

// bytesum(state * mul1) - 510 of 8 states, as int32
inline __m256i weights8(__m256i st) {
    const __m256i x = _mm256_mullo_epi32(st, _mm256_set1_epi32((int) kExl3Mul1));
    const __m256i s16 = _mm256_maddubs_epi16(x, _mm256_set1_epi8(1));
    return _mm256_sub_epi32(_mm256_madd_epi16(s16, _mm256_set1_epi16(1)), _mm256_set1_epi32(510));
}

// hadamard_rotate's butterflies on one block of 128 floats in sixteen registers: the same additions and
// subtractions in the same order, then the same scale
inline void fwht128(__m256 r[16]) {
    for (int i = 0; i < 16; ++i) {
        __m256 v = r[i], b;
        b = _mm256_permute_ps(v, 0xB1);
        v = _mm256_blend_ps(_mm256_add_ps(v, b), _mm256_sub_ps(b, v), 0xAA);
        b = _mm256_permute_ps(v, 0x4E);
        v = _mm256_blend_ps(_mm256_add_ps(v, b), _mm256_sub_ps(b, v), 0xCC);
        b = _mm256_permute2f128_ps(v, v, 1);
        r[i] = _mm256_blend_ps(_mm256_add_ps(v, b), _mm256_sub_ps(b, v), 0xF0);
    }
    for (int len = 1; len < 16; len <<= 1)
        for (int i = 0; i < 16; i += len << 1)
            for (int j = i; j < i + len; ++j) {
                const __m256 a = r[j], c = r[j + len];
                r[j] = _mm256_add_ps(a, c);
                r[j + len] = _mm256_sub_ps(a, c);
            }
    const __m256 scale = _mm256_set1_ps(1.0f / std::sqrt((float) kExl3Had));
    for (int i = 0; i < 16; ++i) r[i] = _mm256_mul_ps(r[i], scale);
}

inline __m256 f16x8(const uint8_t* p) { return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) p)); }

inline float max8(__m256 m) {
    __m128 v = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    v = _mm_max_ps(v, _mm_movehl_ps(v, v));
    return _mm_cvtss_f32(_mm_max_ss(v, _mm_shuffle_ps(v, v, 1)));
}

inline int32_t sum4(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4E));
    return _mm_cvtsi128_si32(_mm_add_epi32(v, _mm_shuffle_epi32(v, 0xB1)));
}

}  // namespace

void prep_avx2(const float* x, const uint8_t* suh_f16, int n, Exl3Act& a) {
    // A tile's 16 int16 after the pack below are the input pairs (2 p, 2 p + 1) in the order 0 1 4 5 2 3 6 7; a
    // slot pair of qp holds one input pair, so each half of qp is a permutation of those eight 32-bit lanes.
    constexpr int kPackedAt[8] = {0, 1, 4, 5, 2, 3, 6, 7};
    alignas(32) int32_t idx[2][8];
    for (int j = 0; j < 16; ++j) idx[j / 8][j % 8] = kPackedAt[kExl3Slot[2 * j] / 2];
    const __m256i ix0 = _mm256_load_si256((const __m256i*) idx[0]), ix1 = _mm256_load_si256((const __m256i*) idx[1]);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    float sum = 0.0f;
    for (int b = 0; b < n / kExl3Had; ++b) {
        __m256 r[16];
        for (int i = 0; i < 16; ++i)
            r[i] = _mm256_mul_ps(_mm256_loadu_ps(x + b * kExl3Had + 8 * i), f16x8(suh_f16 + 2 * (b * kExl3Had + 8 * i)));
        fwht128(r);
        __m256 m = _mm256_andnot_ps(sign, r[0]);
        for (int i = 1; i < 16; ++i) m = _mm256_max_ps(m, _mm256_andnot_ps(sign, r[i]));
        const float amax = max8(m);
        const __m256 inv = _mm256_set1_ps(amax > 0.0f ? 32767.0f / amax : 0.0f);
        a.d[b] = amax / 32767.0f;
        __m256i qs = _mm256_setzero_si256();
        for (int kt = 0; kt < 8; ++kt) {
            const __m256i q0 = _mm256_cvtps_epi32(_mm256_mul_ps(r[2 * kt], inv));
            const __m256i q1 = _mm256_cvtps_epi32(_mm256_mul_ps(r[2 * kt + 1], inv));
            qs = _mm256_add_epi32(qs, _mm256_add_epi32(q0, q1));
            const __m256i q16 = _mm256_packs_epi32(q0, q1);
            int16_t* qp = a.qp[8 * b + kt];
            _mm256_store_si256((__m256i*) qp, _mm256_permutevar8x32_epi32(q16, ix0));
            _mm256_store_si256((__m256i*) (qp + 16), _mm256_permutevar8x32_epi32(q16, ix1));
        }
        const int32_t qsum = sum4(_mm_add_epi32(_mm256_castsi256_si128(qs), _mm256_extracti128_si256(qs, 1)));
        a.qs[b] = qsum;
        sum = std::fma((float) qsum, a.d[b], sum);
    }
    a.off = sum * kExl3Offset;
}

void finish_avx2(float* v, const uint8_t* svh_f16, int r0, int r1) {
    for (int b = r0; b < r1; b += kExl3Had) {
        __m256 r[16];
        for (int i = 0; i < 16; ++i) r[i] = _mm256_loadu_ps(v + b + 8 * i);
        fwht128(r);
        for (int i = 0; i < 16; ++i) _mm256_storeu_ps(v + b + 8 * i, _mm256_mul_ps(r[i], f16x8(svh_f16 + 2 * (b + 8 * i))));
    }
}

void rows_avx2(int k2, const uint8_t* w, int n_in, const Exl3Act* const* a, int nt, float* const* out, int r0, int r1) {
    const Tables& tb = tables(k2);
    const int kt_n = n_in / 16, nw = tb.words;
    const size_t tile = (size_t) 4 * nw;
    constexpr int kBuf = 64 + 16;   // a tile of up to 64 words after its last word, and the load's reach
    alignas(32) uint32_t buf[8][kBuf] = {};
    alignas(32) int32_t cs[kExl3MaxTokens][16];
    for (int band = r0 / 16; band < r1 / 16; ++band) {
        const uint8_t* wb = w + (size_t) band * kt_n * tile;
        __m256 facc[kExl3MaxTokens][2];
        for (int t = 0; t < nt; ++t) facc[t][0] = facc[t][1] = _mm256_setzero_ps();
        for (int b = 0; b < kt_n / 8; ++b) {
            for (int i = 0; i < 8; ++i) {
                const uint8_t* tp = wb + (size_t) (8 * b + i) * tile;
                std::memcpy(&buf[i][0], tp + tile - 4, 4);
                std::memcpy(&buf[i][1], tp, tile);
            }
            for (int g = 0; g < 8; ++g) {
                __m256i acc[kExl3MaxTokens];
                for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_si256();
                for (int i = 0; i < 8; ++i) {
                    const Chunk& lo = tb.c[2 * g];
                    const Chunk& hi = tb.c[2 * g + 1];
                    const __m256i w0 = _mm256_packs_epi32(weights8(states8(buf[i], lo, 0)), weights8(states8(buf[i], hi, 0)));
                    const __m256i w1 = _mm256_packs_epi32(weights8(states8(buf[i], lo, 1)), weights8(states8(buf[i], hi, 1)));
                    for (int t = 0; t < nt; ++t) {
                        const int16_t* qp = a[t]->qp[8 * b + i];
                        const __m256i p0 = _mm256_madd_epi16(w0, _mm256_load_si256((const __m256i*) qp));
                        const __m256i p1 = _mm256_madd_epi16(w1, _mm256_load_si256((const __m256i*) (qp + 16)));
                        acc[t] = _mm256_add_epi32(acc[t], _mm256_add_epi32(p0, p1));
                    }
                }
                for (int t = 0; t < nt; ++t) {
                    cs[t][g] = sum4(_mm256_castsi256_si128(acc[t]));
                    cs[t][g + 8] = sum4(_mm256_extracti128_si256(acc[t], 1));
                }
            }
            for (int t = 0; t < nt; ++t) {
                const __m256 d = _mm256_set1_ps(a[t]->d[b]);
                facc[t][0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*) cs[t])), d, facc[t][0]);
                facc[t][1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*) (cs[t] + 8))), d, facc[t][1]);
            }
        }
        for (int t = 0; t < nt; ++t) {
            const __m256 kinv = _mm256_set1_ps(kExl3KInv), off = _mm256_set1_ps(a[t]->off);
            _mm256_storeu_ps(out[t] + 16 * band, _mm256_fmadd_ps(facc[t][0], kinv, off));
            _mm256_storeu_ps(out[t] + 16 * band + 8, _mm256_fmadd_ps(facc[t][1], kinv, off));
        }
    }
}

}  // namespace basalt::kernels::cpu::exl3_detail
