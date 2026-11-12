// src/kernels/cpu/exl3_cpu.cpp - see include/basalt/kernels/cpu/exl3_cpu.hpp: the scalar reference decoder and
// kernel, the activation transform and the run-time ISA choice.  Compiled for the baseline x86-64 (no AVX), so
// everything here runs on any CPU; the AVX-512 and AVX2 kernels are in their own translation units.
#include "basalt/kernels/cpu/exl3_cpu.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/hadamard.hpp"
#include "basalt/kernels/f16_bits.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace basalt::kernels::cpu {
namespace {

constexpr int kRowOf4[4] = {0, 1, 8, 9};

struct Perm {
    uint8_t p[256];
    constexpr Perm() : p() {
        for (int t = 0; t < 32; ++t)
            for (int j = 0; j < 8; ++j)
                p[t * 8 + j] = (uint8_t) (((t % 4) * 2 + kRowOf4[j % 4]) * 16 + t / 4 + 8 * (j / 4));
    }
};
// Slot 8 L + m of a 32-weight group holds the group's step 4 L + m (m < 4) or 16 + 4 L + m - 4: the order
// _mm512_packs_epi32 leaves two 16-step halves in.  A group's step s is tile input 2 (s / 8) + {0, 1, 8, 9}[s % 4].
struct Slot {
    uint8_t s[32];
    constexpr Slot() : s() {
        for (int l = 0; l < 4; ++l)
            for (int m = 0; m < 8; ++m) {
                const int step = m < 4 ? 4 * l + m : 16 + 4 * l + m - 4;
                s[8 * l + m] = (uint8_t) (2 * (step / 8) + kRowOf4[step % 4]);
            }
    }
};
constexpr Perm kPermV;
constexpr Slot kSlotV;

uint32_t ld32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;   // x86 is little-endian
}
float f16(const uint8_t* p, int i) {
    uint16_t v;
    std::memcpy(&v, p + 2 * i, 2);
    return f32_from_f16(v);
}

exl3_detail::Tables make_tables(int k2) {
    exl3_detail::Tables t{};
    t.words = 4 * k2;
    for (int c = 0; c < 16; ++c) {
        exl3_detail::Chunk& ch = t.c[c];
        ch.base = (exl3_step_end(k2, 16 * c) - 1) / 32;
        for (int i = 0; i < 16; ++i) {
            const int e = exl3_step_end(k2, 16 * c + i), hi = (e - 1) / 32;
            ch.hi[i] = (uint32_t) (hi + 1 - ch.base);
            ch.shr[i] = (uint32_t) ((hi + 1) * 32 - e);
        }
        for (int q = 0; q < 8; ++q) {   // steps 16 c + 2 q and 16 c + 2 q + 1 from the odd step's word and the one before
            const int e0 = exl3_step_end(k2, 16 * c + 2 * q), e1 = exl3_step_end(k2, 16 * c + 2 * q + 1);
            const int hi = (e1 - 1) / 32, s1 = (hi + 1) * 32 - e1, s0 = s1 + e1 - e0;
            ch.pair[2 * q] = (uint32_t) (hi + 1 - ch.base);
            ch.pair[2 * q + 1] = (uint32_t) (hi - ch.base);
            const uint8_t b[8] = {(uint8_t) s0, (uint8_t) (s0 + 8), 0, 0, (uint8_t) s1, (uint8_t) (s1 + 8), 0, 0};
            for (int j = 0; j < 8; ++j) ch.ctrl[8 * q + j] = b[j];
        }
    }
    return t;
}

}  // namespace

const uint8_t kExl3Perm[256] = {
#define P4(i) kPermV.p[i], kPermV.p[(i) + 1], kPermV.p[(i) + 2], kPermV.p[(i) + 3]
#define P16(i) P4(i), P4((i) + 4), P4((i) + 8), P4((i) + 12)
#define P64(i) P16(i), P16((i) + 16), P16((i) + 32), P16((i) + 48)
    P64(0), P64(64), P64(128), P64(192)};
#undef P64
#undef P16
#undef P4
const uint8_t kExl3Slot[32] = {
#define S4(i) kSlotV.s[i], kSlotV.s[(i) + 1], kSlotV.s[(i) + 2], kSlotV.s[(i) + 3]
    S4(0), S4(4), S4(8), S4(12), S4(16), S4(20), S4(24), S4(28)};
#undef S4

bool exl3_isa_ok(Exl3Isa isa) {
    return isa == Exl3Isa::scalar || (isa == Exl3Isa::avx2 ? cpu_avx2_ok() : cpu_avx512_ok());
}
Exl3Isa exl3_isa() {
    static const Exl3Isa isa = exl3_isa_ok(Exl3Isa::avx512) ? Exl3Isa::avx512
                               : exl3_isa_ok(Exl3Isa::avx2) ? Exl3Isa::avx2 : Exl3Isa::scalar;
    return isa;
}
const char* exl3_isa_name(Exl3Isa isa) {
    return isa == Exl3Isa::avx512 ? "avx512" : isa == Exl3Isa::avx2 ? "avx2" : "scalar";
}

int exl3_step_end(int k2, int t) noexcept {
    if (k2 % 2 == 0) return (t + 1) * (k2 / 2);
    return ((t + 1) / 2) * k2 + (t % 2 == 0 ? k2 / 2 : 0);
}

void exl3_tile_states(const uint8_t* tile, int k2, uint16_t st[256]) {
    const int nw = 4 * k2;
    for (int t = 0; t < 256; ++t) {
        const int e = exl3_step_end(k2, t), hi = (e - 1) / 32, lo = (hi + nw - 1) % nw;
        const uint64_t v = ((uint64_t) ld32(tile + 4 * lo) << 32) | ld32(tile + 4 * hi);
        st[t] = (uint16_t) (v >> ((hi + 1) * 32 - e));
    }
}

float exl3_mul1_value(uint32_t state) noexcept {
    // the product and the sum are exact in double, so the conversions round once, as the fma does
    const double v = (double) (1024 + exl3_bytesum(state & 0xffffu)) * (double) kExl3KInv + (double) kExl3KBias;
    return f32_from_f16(f16_from_f32((float) v));
}

void exl3_dequant_inner(int type, const uint8_t* w, int n_in, int n_out, float* out) {
    const int k2 = exl3_k2(type), kt_n = n_in / 16;
    const size_t tile = (size_t) 16 * k2;
    uint16_t st[256];
    for (int nt = 0; nt < n_out / 16; ++nt)
        for (int kt = 0; kt < kt_n; ++kt) {
            exl3_tile_states(w + ((size_t) nt * kt_n + kt) * tile, k2, st);
            for (int t = 0; t < 256; ++t)
                out[(size_t) (16 * nt + kExl3Perm[t] % 16) * n_in + 16 * kt + kExl3Perm[t] / 16] = exl3_mul1_value(st[t]);
        }
}

void exl3_prep(Exl3Isa isa, const float* x, const uint8_t* suh_f16, int n, Exl3Act& a) {
    if (isa == Exl3Isa::avx512) {
        exl3_detail::prep_avx512(x, suh_f16, n, a);
        return;
    }
    if (isa == Exl3Isa::avx2) {
        exl3_detail::prep_avx2(x, suh_f16, n, a);
        return;
    }
    alignas(64) float v[kExl3MaxN];
    for (int i = 0; i < n; ++i) v[i] = x[i] * f16(suh_f16, i);
    hadamard_rotate(v, n, kExl3Had, nullptr);
    float sum = 0.0f;
    for (int b = 0; b < n / kExl3Had; ++b) {
        const float* vb = v + b * kExl3Had;
        float amax = 0.0f;
        for (int i = 0; i < kExl3Had; ++i) {
            const float m = std::fabs(vb[i]);
            if (m > amax) amax = m;
        }
        const float inv = amax > 0.0f ? 32767.0f / amax : 0.0f;
        a.d[b] = amax / 32767.0f;
        int32_t qs = 0;
        for (int kt = 0; kt < 8; ++kt) {
            int16_t q[16];
            for (int i = 0; i < 16; ++i) {
                q[i] = (int16_t) std::lrintf(vb[16 * kt + i] * inv);
                qs += q[i];
            }
            int16_t* qp = a.qp[8 * b + kt];
            for (int s = 0; s < 32; ++s) qp[s] = q[kExl3Slot[s]];
        }
        a.qs[b] = qs;
        sum = std::fma((float) qs, a.d[b], sum);
    }
    a.off = sum * kExl3Offset;
}

void exl3_finish(Exl3Isa isa, float* v, const uint8_t* svh_f16, int r0, int r1) {
    if (isa == Exl3Isa::avx512) {
        exl3_detail::finish_avx512(v, svh_f16, r0, r1);
        return;
    }
    if (isa == Exl3Isa::avx2) {
        exl3_detail::finish_avx2(v, svh_f16, r0, r1);
        return;
    }
    hadamard_rotate(v + r0, r1 - r0, kExl3Had, nullptr);
    for (int r = r0; r < r1; ++r) v[r] *= f16(svh_f16, r);
}

namespace exl3_detail {

const Tables& tables(int k2) {
    static const std::unique_ptr<Tables[]> all = [] {
        std::unique_ptr<Tables[]> t(new Tables[17]);
        for (int k = 2; k <= 16; ++k) t[k] = make_tables(k);
        return t;
    }();
    return all[k2];
}

void rows_scalar(int k2, const uint8_t* w, int n_in, const Exl3Act* const* a, int nt, float* const* out, int r0, int r1) {
    const int kt_n = n_in / 16;
    const size_t tile = (size_t) 16 * k2;
    uint8_t slot_of[16];
    for (int s = 0; s < 32; ++s) slot_of[kExl3Slot[s]] = (uint8_t) s;
    uint16_t st[256];
    for (int band = r0 / 16; band < r1 / 16; ++band) {
        const uint8_t* wb = w + (size_t) band * kt_n * tile;
        float acc[kExl3MaxTokens][16] = {};
        for (int b = 0; b < kt_n / 8; ++b) {
            int32_t sum[kExl3MaxTokens][16] = {};
            for (int kt = 8 * b; kt < 8 * b + 8; ++kt) {
                exl3_tile_states(wb + (size_t) kt * tile, k2, st);
                for (int s = 0; s < 256; ++s) {
                    const int wv = exl3_bytesum(st[s]) - 510, in = kExl3Perm[s] / 16, o = kExl3Perm[s] % 16;
                    for (int t = 0; t < nt; ++t) sum[t][o] += wv * a[t]->qp[kt][slot_of[in]];
                }
            }
            for (int t = 0; t < nt; ++t)
                for (int o = 0; o < 16; ++o) acc[t][o] = std::fma((float) sum[t][o], a[t]->d[b], acc[t][o]);
        }
        for (int t = 0; t < nt; ++t)
            for (int o = 0; o < 16; ++o) out[t][16 * band + o] = std::fma(acc[t][o], kExl3KInv, a[t]->off);
    }
}

}  // namespace exl3_detail

void exl3_rows(Exl3Isa isa, int type, const uint8_t* w, int n_in, const Exl3Act* const* a, int nt, float* const* out,
               int r0, int r1) {
    const int k2 = exl3_k2(type);
    // caller bugs: a type or an ISA this CPU does not have, or a shape the kernels do not take
    if (k2 == 0 || !exl3_isa_ok(isa) || nt > kExl3MaxTokens || n_in > kExl3MaxN || n_in % kExl3Had || (r0 | r1) % 16)
        std::abort();
    if (isa == Exl3Isa::avx512) exl3_detail::rows_avx512(k2, w, n_in, a, nt, out, r0, r1);
    else if (isa == Exl3Isa::avx2) exl3_detail::rows_avx2(k2, w, n_in, a, nt, out, r0, r1);
    else exl3_detail::rows_scalar(k2, w, n_in, a, nt, out, r0, r1);
}

namespace {

struct Scratch {
    Exl3Act a[2][kExl3MaxTokens];
    alignas(64) float v[2][kExl3MaxTokens][kExl3MaxN];
};
Scratch& scratch() {
    thread_local std::unique_ptr<Scratch> s(new Scratch);
    return *s;
}
int up128(int r) { return (r + kExl3Had - 1) / kExl3Had * kExl3Had; }

}  // namespace

void exl3_gu_rows(Exl3Isa isa, const NativeFmt& f, const uint8_t* blob, const float* const* x, int nt,
                  float* const* ff, int r0, int r1) {
    r0 = up128(r0);
    r1 = up128(r1);
    if (r0 >= r1) return;
    const int ne = (int) f.n_embd, nf = (int) f.n_ff;
    const uint8_t* vec = blob + f.bytes - f.tail;
    const uint8_t* suh[2] = {vec, vec + 2 * (ne + nf)};
    const uint8_t* svh[2] = {vec + 2 * ne, vec + 2 * (2 * ne + nf)};
    Scratch& s = scratch();
    for (int m = 0; m < 2; ++m) {
        const Exl3Act* ap[kExl3MaxTokens];
        float* vp[kExl3MaxTokens];
        for (int t = 0; t < nt; ++t) {
            exl3_prep(isa, x[t], suh[m], ne, s.a[m][t]);
            ap[t] = &s.a[m][t];
            vp[t] = s.v[m][t];
        }
        exl3_rows(isa, f.gu_type, blob + (m ? f.up_off : 0), ne, ap, nt, vp, r0, r1);
        for (int t = 0; t < nt; ++t) exl3_finish(isa, s.v[m][t], svh[m], r0, r1);
    }
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r) {
            const float g = s.v[0][t][r], u = s.v[1][t][r];
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;   // native_gu_rows' SwiGLU expression
        }
}

void exl3_down_rows(Exl3Isa isa, const NativeFmt& f, const uint8_t* blob, const float* const* h, int nt,
                    float* const* out, int r0, int r1) {
    r0 = up128(r0);
    r1 = up128(r1);
    if (r0 >= r1) return;
    const int ne = (int) f.n_embd, nf = (int) f.n_ff;
    const uint8_t* vec = blob + f.bytes - f.tail + 4 * (ne + nf);   // suh_d, then svh_d
    Scratch& s = scratch();
    const Exl3Act* ap[kExl3MaxTokens];
    for (int t = 0; t < nt; ++t) {
        exl3_prep(isa, h[t], vec, nf, s.a[0][t]);
        ap[t] = &s.a[0][t];
    }
    exl3_rows(isa, f.d_type, blob + f.down_off, nf, ap, nt, out, r0, r1);
    for (int t = 0; t < nt; ++t) exl3_finish(isa, out[t], vec + 2 * nf, r0, r1);
}

}  // namespace basalt::kernels::cpu
