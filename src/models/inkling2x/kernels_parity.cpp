// src/models/inkling2x/kernels_parity.cpp - Inkling 2's kernels against plain CPU references in double, at the model's
// geometry: the depth mix, a loop controller, the short convolutions and their ring, the attention's inputs, the
// attention (a sliding layer past its window and a full layer past the bias extent), the routed experts with situ
// through native_expert_grouped (IQ3_XXS gate/up and IQ4_NL down, dequantized by ggml; EXL3 at 2.5 and 3 bpw, decoded by
// the CPU reference) and the logits.
// A row's result must not depend on the rows beside it: each check also runs its first row alone.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/exl3_cpu.hpp"
#include "basalt/kernels/exl3_kernels.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/models/inkling2x/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::models::inkling2x::kernels;
namespace bk = basalt::kernels;

namespace {

constexpr int E = 2304, H = 24, HKV = 8, D = 128, DREL = 16, EXT = 1024, RANK = 64, NEXP = 128, KTOP = 8, FF = 1152;
std::mt19937 rng(7);
int failures = 0;

std::vector<float> randn(size_t n, float s = 1.0f) {
    std::normal_distribution<float> d(0.0f, s);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}

// a device word holding a position, for the kernels that read theirs from one
const int64_t* dpos(int64_t p) {
    int64_t* d = nullptr;
    cudaMalloc(&d, sizeof(int64_t));
    cudaMemcpy(d, &p, sizeof(int64_t), cudaMemcpyHostToDevice);
    return d;
}

template <class T> T* up(const std::vector<T>& v) {
    T* p = nullptr;
    cudaMalloc(&p, v.size() * sizeof(T));
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T> T* zeros(size_t n) {
    T* p = nullptr;
    cudaMalloc(&p, n * sizeof(T));
    cudaMemset(p, 0, n * sizeof(T));
    return p;
}
template <class T> std::vector<T> down(const T* p, size_t n) {
    std::vector<T> v(n);
    cudaDeviceSynchronize();
    cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

void report(const char* what, const std::vector<float>& got, const std::vector<double>& ref, double tol) {
    double mabs = 0, mref = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        mabs = std::max(mabs, std::fabs((double) got[i] - ref[i]));
        mref = std::max(mref, std::fabs(ref[i]));
    }
    double l1 = 0, l1r = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        l1 += std::fabs((double) got[i] - ref[i]);
        l1r += std::fabs(ref[i]);
    }
    const double rel = mref > 0 ? mabs / mref : mabs;
    const bool ok = std::isfinite(l1) && std::isfinite(rel) && rel <= tol;
    std::printf("%-34s max abs %.3e  max rel %.3e  L1 rel %.3e  %s\n", what, mabs, rel, l1r > 0 ? l1 / l1r : l1, ok ? "ok" : "FAIL");
    failures += !ok;
}

void same(const char* what, const float* a, const float* b, size_t n) {
    const bool ok = std::memcmp(a, b, n * sizeof(float)) == 0;
    std::printf("%-34s %s\n", what, ok ? "a row alone: bit for bit" : "FAIL: a row alone differs");
    failures += !ok;
}

std::vector<double> rmsn(const float* x, int n, double eps) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const double iv = 1.0 / std::sqrt(ss / n + eps);
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) y[i] = x[i] * iv;
    return y;
}

// ---------------------------------------------------------------------------------------------------------- depth mix
void check_mix() {
    const int R = 3, NS = 5, NQ = 3;
    std::vector<std::vector<float>> st;
    k::States s;
    for (int i = 0; i < NS; ++i) {
        st.push_back(randn((size_t) R * E, 0.5f + i));
        s.p[i] = up(st.back());
    }
    s.n = NS;
    auto qh = randn((size_t) NQ * E, 0.05f);
    float* q = up(qh);
    float* out = zeros<float>((size_t) NQ * R * E);
    k::depth_mix(s, q, NQ, R, E, 1e-6f, out, nullptr);
    std::vector<double> ref((size_t) NQ * R * E);
    for (int qi = 0; qi < NQ; ++qi)
        for (int r = 0; r < R; ++r) {
            std::vector<std::vector<double>> nrm;
            std::vector<double> p(NS);
            double m = -1e300, sum = 0;
            for (int i = 0; i < NS; ++i) {
                nrm.push_back(rmsn(st[i].data() + (size_t) r * E, E, 1e-6));
                double d = 0;
                for (int e = 0; e < E; ++e) d += nrm[i][e] * qh[(size_t) qi * E + e];
                p[i] = d / std::sqrt((double) E);
                m = std::max(m, p[i]);
            }
            for (double& x : p) sum += (x = std::exp(x - m));
            for (int e = 0; e < E; ++e) {
                double acc = 0;
                for (int i = 0; i < NS; ++i) acc += p[i] / sum * nrm[i][e];
                ref[((size_t) qi * R + r) * E + e] = acc;
            }
        }
    const auto got = down(out, ref.size());
    report("depth mix (5 states, 3 queries)", got, ref, 1e-5);
    float* one = zeros<float>((size_t) NQ * E);
    k::depth_mix(s, q, NQ, 1, E, 1e-6f, one, nullptr);
    const auto g1 = down(one, (size_t) NQ * E);
    std::vector<float> row0;
    for (int qi = 0; qi < NQ; ++qi) row0.insert(row0.end(), got.begin() + (size_t) qi * R * E, got.begin() + (size_t) qi * R * E + E);
    same("depth mix", g1.data(), row0.data(), row0.size());
}

// ---------------------------------------------------------------------------------------------------- loop controller
void check_control() {
    const int R = 3;
    auto ch = randn((size_t) R * E), mh = randn((size_t) R * E, 2.0f);
    auto dh = randn((size_t) RANK * 2 * E, 0.02f), gh = randn((size_t) E * RANK, 0.1f), fh = randn((size_t) E * RANK, 0.1f);
    float *c = up(ch), *m = up(mh), *dw = up(dh), *gw = up(gh), *fw = up(fh);
    float* g = zeros<float>((size_t) R * RANK);
    float* out = zeros<float>((size_t) R * E);
    k::loop_control(c, m, dw, gw, fw, R, E, RANK, 1e-6f, g, out, nullptr);
    std::vector<double> ref((size_t) R * E);
    for (int r = 0; r < R; ++r) {
        auto a = rmsn(ch.data() + (size_t) r * E, E, 1e-6), b = rmsn(mh.data() + (size_t) r * E, E, 1e-6);
        std::vector<double> gg(RANK);
        for (int o = 0; o < RANK; ++o) {
            double acc = 0;
            for (int e = 0; e < E; ++e) acc += dh[(size_t) o * 2 * E + e] * a[e] + dh[(size_t) o * 2 * E + E + e] * b[e];
            gg[o] = acc;
        }
        for (int e = 0; e < E; ++e) {
            double ga = 0, fo = 0;
            for (int o = 0; o < RANK; ++o) {
                ga += gh[(size_t) e * RANK + o] * gg[o];
                fo += fh[(size_t) e * RANK + o] * gg[o];
            }
            const double cv = ch[(size_t) r * E + e], mv = mh[(size_t) r * E + e];
            ref[(size_t) r * E + e] = cv - std::tanh(fo) * cv + std::tanh(ga) * mv;
        }
    }
    const auto got = down(out, ref.size());
    report("loop controller", got, ref, 1e-5);
    float* one = zeros<float>(E);
    k::loop_control(c, m, dw, gw, fw, 1, E, RANK, 1e-6f, g, one, nullptr);
    same("loop controller", down(one, E).data(), got.data(), E);
}

// ----------------------------------------------------------------------------------------------- short convolutions
void check_conv() {
    const int R = 5, C = 1024, SLOTS = 32, WIDTH = 6656, CH0 = 1024, HW = 128;
    const int64_t pos0 = 37;
    auto xh = randn((size_t) R * C), wh = randn((size_t) C * 4, 0.3f), dyh = randn((size_t) R * (C / HW) * 4, 0.3f);
    auto ringh = randn((size_t) SLOTS * WIDTH);
    float *x = up(xh), *w = up(wh), *dy = up(dyh), *ringd = up(ringh), *out = zeros<float>((size_t) R * C);
    k::ConvRing ring{ringd, SLOTS, WIDTH};
    for (int dyn = 0; dyn < 2; ++dyn) {
        const float scale = 1.0f / std::sqrt((float) ((dyn ? 2 : 1) * 4 + 1));
        k::short_conv(x, w, dyn ? dy : nullptr, HW, ring, CH0, dpos(pos0), R, C, scale, out, nullptr);
        std::vector<double> ref((size_t) R * C);
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) {
                double xv[4];
                for (int kk = 0; kk < 4; ++kk) {
                    const int rr = r - kk;
                    xv[3 - kk] = rr >= 0 ? xh[(size_t) rr * C + c] : ringh[(size_t) ((pos0 + rr) % SLOTS) * WIDTH + CH0 + c];
                }
                double acc = 0;
                for (int i = 0; i < 4; ++i) acc += xv[i] * wh[(size_t) c * 4 + i];
                if (dyn)
                    for (int lag = 0; lag < 4; ++lag) acc += xv[3 - lag] * dyh[(size_t) r * (C / HW) * 4 + (c / HW) * 4 + lag];
                ref[(size_t) r * C + c] = (xv[3] + acc) * scale;
            }
        report(dyn ? "short conv, input taps" : "short conv", down(out, ref.size()), ref, 1e-6);
    }
    // position 1: the taps before position 0 are zero
    float* o1 = zeros<float>(C);
    k::short_conv(x, w, nullptr, HW, ring, CH0, dpos(1), 1, C, 1.0f, o1, nullptr);
    std::vector<double> r1(C);
    for (int c = 0; c < C; ++c)
        r1[c] = xh[c] + xh[c] * wh[(size_t) c * 4 + 3] + ringh[(size_t) 0 * WIDTH + CH0 + c] * wh[(size_t) c * 4 + 2];
    report("short conv at position 1", down(o1, C), r1, 1e-6);
    k::conv_store(x, ring, CH0, dpos(pos0), R, 3, C, nullptr);
    const auto rg = down(ringd, ringh.size());
    bool ok = true;
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; c += 97) {
            const float want = r >= R - 3 ? xh[(size_t) r * C + c] : ringh[(size_t) ((pos0 + r) % SLOTS) * WIDTH + CH0 + c];
            ok &= rg[(size_t) ((pos0 + r) % SLOTS) * WIDTH + CH0 + c] == want;
        }
    std::printf("%-34s %s\n", "conv ring keeps the last rows", ok ? "ok" : "FAIL");
    failures += !ok;
}

// --------------------------------------------------------------------------------- the attention's inputs and itself
struct Attn {
    std::vector<float> q, k, v, r, sink, gate, qs, rs, rb;
};

// R = 3 takes the decode's splits of the keys; R = 17 (past a decode window) a prompt chunk's
void check_attention(bool full, int R) {
    const int W = full ? 0 : EXT;
    const int64_t pos0 = full ? 1500 : 1200;   // past the window / past the bias extent
    const int64_t cells = full ? 16384 : W + 16;   // a full layer: a cell per position
    Attn h;
    h.q = randn((size_t) R * H * D);
    h.k = randn((size_t) R * HKV * D);
    h.v = randn((size_t) R * HKV * D);
    h.r = randn((size_t) R * H * DREL);
    h.sink = randn((size_t) R * H, 2.0f);
    h.gate = randn((size_t) R * H * D);
    h.qs = randn((size_t) H * D, 1.0f);
    h.rs = randn((size_t) H * DREL, 1.0f);
    h.rb = randn((size_t) EXT * DREL, 0.5f);
    // the cached rows before pos0, as fp16
    std::vector<__half> kc((size_t) cells * HKV * D), vc((size_t) cells * HKV * D);
    std::normal_distribution<float> nd(0.0f, 0.1f);
    for (size_t i = 0; i < kc.size(); ++i) { kc[i] = __float2half(nd(rng)); vc[i] = __float2half(nd(rng)); }
    float *q = up(h.q), *kd = up(h.k), *vd = up(h.v), *r = up(h.r), *sink = up(h.sink), *gate = up(h.gate), *qs = up(h.qs), *rs = up(h.rs),
          *rb = up(h.rb);
    __half *kcd = up(kc), *vcd = up(vc);
    k::PrepArgs pa;
    pa.qk_clip = 22.6274f;
    pa.r_clip = 8.0f;
    pa.log_scale = full;
    pa.log_alpha = 0.1f;
    pa.log_floor = 8192.0f;
    // a prompt past the log-scale floor: positions where tau > 1
    const int64_t ppos = full ? 9000 : pos0;
    k::qkvr_prep(q, kd, vd, r, qs, rs, pa, dpos(ppos), R, (uint16_t*) kcd, (uint16_t*) vcd, cells, nullptr);
    // the reference preparation
    std::vector<double> qr((size_t) R * H * D), rr((size_t) R * H * DREL);
    for (int row = 0; row < R; ++row) {
        const int64_t p = ppos + row;
        const double tau = full ? 1.0 + 0.1 * std::log(std::max(1.0, (double) (p + 1) / 8192.0)) : 1.0;
        for (int hh = 0; hh < H; ++hh) {
            double ss = 0;
            std::vector<double> t(D);
            for (int d = 0; d < D; ++d) ss += (t[d] = (double) h.q[((size_t) row * H + hh) * D + d] * h.qs[(size_t) hh * D + d]) * t[d];
            const double f = std::min(std::max(22.6274 / (std::sqrt(ss) + 1e-6), 0.0), 1.0);
            for (int d = 0; d < D; ++d) qr[((size_t) row * H + hh) * D + d] = t[d] * f * tau;
            ss = 0;
            std::vector<double> u(DREL);
            for (int d = 0; d < DREL; ++d) ss += (u[d] = (double) h.r[((size_t) row * H + hh) * DREL + d] * h.rs[(size_t) hh * DREL + d]) * u[d];
            const double fr = std::min(std::max(8.0 / (std::sqrt(ss) + 1e-6), 0.0), 1.0);
            for (int d = 0; d < DREL; ++d) rr[((size_t) row * H + hh) * DREL + d] = u[d] * fr * tau;
        }
    }
    report(full ? "Q prep, full layer (tau > 1)" : "Q prep, sliding layer", down(q, qr.size()), qr, 1e-5);
    report(full ? "R prep, full layer (tau > 1)" : "R prep, sliding layer", down(r, rr.size()), rr, 1e-5);
    // K and V in the cache, back as floats
    {
        const auto kb = down(kcd, kc.size()), vb = down(vcd, vc.size());
        std::vector<float> got;
        std::vector<double> ref;
        for (int row = 0; row < R; ++row)
            for (int g = 0; g < HKV; ++g) {
                const size_t cell = (size_t) ((ppos + row) % cells);
                double ss = 0, sv = 0;
                for (int d = 0; d < D; ++d) {
                    ss += (double) h.k[((size_t) row * HKV + g) * D + d] * h.k[((size_t) row * HKV + g) * D + d];
                    sv += (double) h.v[((size_t) row * HKV + g) * D + d] * h.v[((size_t) row * HKV + g) * D + d];
                }
                const double f = std::min(std::max(22.6274 / (std::sqrt(ss) + 1e-6), 0.0), 1.0), iv = 1.0 / std::sqrt(sv / D + 1e-6);
                for (int d = 0; d < D; ++d) {
                    got.push_back(__half2float(kb[(cell * HKV + g) * D + d]));
                    ref.push_back(h.k[((size_t) row * HKV + g) * D + d] * f);
                    got.push_back(__half2float(vb[(cell * HKV + g) * D + d]));
                    ref.push_back(h.v[((size_t) row * HKV + g) * D + d] * iv);
                }
            }
        report("K and V into the fp16 cache", got, ref, 1e-3);
    }
    // attention over the cache the prep left (the rows at ppos: their keys are the cache's)
    k::AttnArgs aa;
    aa.window = W;
    aa.cells = cells;
    const int64_t apos = ppos;
    const int64_t maxk = W > 0 ? W : apos + R;
    void* ws = nullptr;
    cudaMalloc(&ws, k::attention_workspace_bytes(R, H, maxk));
    float* out = zeros<float>((size_t) R * H * D);
    k::attention(q, r, sink, gate, rb, (const uint16_t*) kcd, (const uint16_t*) vcd, aa, dpos(apos), apos, R, ws, out, nullptr);
    const auto qg = down(q, qr.size()), rg = down(r, rr.size());
    const auto kb = down(kcd, kc.size()), vb = down(vcd, vc.size());
    std::vector<double> ref((size_t) R * H * D);
    for (int row = 0; row < R; ++row) {
        const int64_t p = apos + row;
        const int64_t lo = W > 0 && p >= W ? p - W + 1 : 0;
        for (int hh = 0; hh < H; ++hh) {
            const int g = hh / (H / HKV);
            std::vector<double> tab(EXT + 1, 0.0);
            for (int e = 0; e < EXT; ++e)
                for (int i = 0; i < DREL; ++i) tab[e] += (double) h.rb[(size_t) e * DREL + i] * rg[((size_t) row * H + hh) * DREL + i];
            for (double& t : tab) t -= h.sink[(size_t) row * H + hh];
            std::vector<double> s;
            double m = 0;   // the sink's logit
            for (int64_t j = lo; j <= p; ++j) {
                const size_t cell = (size_t) (j % cells);
                double dot = 0;
                for (int d = 0; d < D; ++d) dot += (double) qg[((size_t) row * H + hh) * D + d] * __half2float(kb[(cell * HKV + g) * D + d]);
                const int64_t dist = p - j;
                s.push_back(dot / std::sqrt((double) D) + tab[dist < EXT ? dist : EXT]);
                m = std::max(m, s.back());
            }
            double l = std::exp(0.0 - m);
            std::vector<double> o(D, 0.0);
            for (int64_t j = lo; j <= p; ++j) {
                const double e = std::exp(s[(size_t) (j - lo)] - m);
                l += e;
                const size_t cell = (size_t) (j % cells);
                for (int d = 0; d < D; ++d) o[d] += e * __half2float(vb[(cell * HKV + g) * D + d]);
            }
            for (int d = 0; d < D; ++d) {
                const size_t at = ((size_t) row * H + hh) * D + d;
                ref[at] = o[d] / l / (1.0 + std::exp(-(double) h.gate[at]));
            }
        }
    }
    const auto got = down(out, ref.size());
    if (R > 16) {   // a row alone takes the decode's splits: not the same sums
        report(full ? "attention, full layer, prompt splits" : "attention, sliding, prompt splits", got, ref, 1e-4);
        return;
    }
    report(full ? "attention, full layer (> extent)" : "attention, sliding (wrapped ring)", got, ref, 1e-4);
    float* one = zeros<float>((size_t) H * D);
    k::attention(q, r, sink, gate, rb, (const uint16_t*) kcd, (const uint16_t*) vcd, aa, dpos(apos), apos, 1, ws, one, nullptr);
    same(full ? "attention, full layer" : "attention, sliding", down(one, (size_t) H * D).data(), got.data(), (size_t) H * D);
}

// ----------------------------------------------------------------------------------------------- the routed experts
std::vector<uint8_t> random_blocks(int type, size_t rows, size_t cols) {
    const size_t bs = ggml_blck_size((ggml_type) type), ts = ggml_type_size((ggml_type) type);
    std::vector<uint8_t> b(rows * cols / bs * ts);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.0005f, 0.003f);   // gate and up of a few units: the caps' working range
    for (size_t i = 0; i < b.size(); i += ts) {
        const __half d = __float2half(sc(rng));
        std::memcpy(&b[i], &d, 2);
        for (size_t j = 2; j < ts; ++j) b[i + j] = (uint8_t) byte(rng);
    }
    return b;
}

void check_experts() {
    const int R = 3, GU = 18, DN = 20;   // IQ3_XXS gate/up, IQ4_NL down
    bk::NativeExpertLayout L = bk::native_expert_layout(GU, DN, E, FF);
    L.situ_beta = 4.0f;
    L.situ_up = 25.0f;
    std::vector<uint8_t> blobs;
    for (int e = 0; e < NEXP; ++e) {
        auto gu = random_blocks(GU, 2 * FF, E), dn = random_blocks(DN, E, FF);
        if (gu.size() + dn.size() != L.bytes) { std::printf("expert blob size %zu vs layout %zu: FAIL\n", gu.size() + dn.size(), L.bytes); ++failures; return; }
        blobs.insert(blobs.end(), gu.begin(), gu.end());
        blobs.insert(blobs.end(), dn.begin(), dn.end());
    }
    auto xh = randn((size_t) R * E);
    std::vector<int32_t> ids((size_t) R * KTOP);
    std::vector<float> wts((size_t) R * KTOP);
    for (int r = 0; r < R; ++r) {
        std::vector<int> pick(NEXP);
        for (int e = 0; e < NEXP; ++e) pick[e] = e;
        std::shuffle(pick.begin(), pick.end(), rng);
        if (r == 1) pick[0] = ids[0];   // a shared expert across rows
        for (int j = 0; j < KTOP; ++j) { ids[(size_t) r * KTOP + j] = pick[j]; wts[(size_t) r * KTOP + j] = 0.5f + 0.25f * j; }
    }
    uint8_t* base = up(blobs);
    std::vector<unsigned long long> ptrs(NEXP);
    for (int e = 0; e < NEXP; ++e) ptrs[(size_t) e] = (unsigned long long) (uintptr_t) (base + (size_t) e * L.bytes);
    unsigned long long* eptr = up(ptrs);
    float* x = up(xh);
    int32_t* idd = up(ids);
    float* wd = up(wts);
    const size_t q8 = (size_t) E / 32 * 36;
    void* xq = zeros<uint8_t>((size_t) R * q8);
    bk::quantize_q8_1_rows(x, R, E, xq, nullptr);
    auto* gp = zeros<unsigned long long>(NEXP);
    int32_t *gs = zeros<int32_t>(NEXP + 1), *ng = zeros<int32_t>(1), *dst = zeros<int32_t>((size_t) R * KTOP), *tok = zeros<int32_t>((size_t) R * KTOP);
    void* scratch = zeros<uint8_t>(bk::native_expert_scratch_bytes(L, (int64_t) R * KTOP));
    float* eo = zeros<float>((size_t) R * KTOP * E);
    float* y = zeros<float>((size_t) R * E);
    auto run = [&](int rows, float* yy) {
        k::group_experts(idd, rows, KTOP, NEXP, eptr, nullptr, gp, nullptr, gs, ng, dst, tok, nullptr);
        bk::native_expert_grouped(L, gp, gs, ng, dst, tok, NEXP, (int64_t) rows * KTOP, xq, scratch, eo, nullptr);
        k::sum_experts(eo, wd, rows, KTOP, E, yy, nullptr);
    };
    run(R, y);
    // the reference: ggml's dequantizer, situ and the weighted sum in double
    const auto* tg = ggml_get_type_traits((ggml_type) GU);
    const auto* td = ggml_get_type_traits((ggml_type) DN);
    std::vector<double> ref((size_t) R * E, 0.0);
    std::vector<float> wg((size_t) 2 * FF * E), wdn((size_t) E * FF);
    for (int r = 0; r < R; ++r)
        for (int j = 0; j < KTOP; ++j) {
            const int e = ids[(size_t) r * KTOP + j];
            const uint8_t* b = blobs.data() + (size_t) e * L.bytes;
            tg->to_float(b, wg.data(), (int64_t) 2 * FF * E);
            td->to_float(b + L.down_off, wdn.data(), (int64_t) E * FF);
            std::vector<double> hh(FF);
            for (int f = 0; f < FF; ++f) {
                double g = 0, u = 0;
                for (int c = 0; c < E; ++c) {
                    g += (double) wg[(size_t) f * E + c] * xh[(size_t) r * E + c];
                    u += (double) wg[(size_t) (FF + f) * E + c] * xh[(size_t) r * E + c];
                }
                hh[f] = 4.0 * std::tanh(g / 4.0) / (1.0 + std::exp(-g)) * 25.0 * std::tanh(u / 25.0);
            }
            for (int c = 0; c < E; ++c) {
                double acc = 0;
                for (int f = 0; f < FF; ++f) acc += (double) wdn[(size_t) c * FF + f] * hh[f];
                ref[(size_t) r * E + c] += wts[(size_t) r * KTOP + j] * acc;
            }
        }
    const auto got = down(y, ref.size());
    // q8_1 activations on both products: the measure is relative to the largest output
    report("routed experts with situ", got, ref, 3e-2);
    float* y1 = zeros<float>(E);
    run(1, y1);
    same("routed experts", down(y1, E).data(), got.data(), E);
}

// the normalized 128-wide Sylvester-Hadamard transform of each block of v
void hadamard(std::vector<double>& v) {
    for (size_t b = 0; b < v.size(); b += 128) {
        for (int h = 1; h < 128; h <<= 1)
            for (int i = 0; i < 128; i += 2 * h)
                for (int j = i; j < i + h; ++j) {
                    const double x = v[b + j], y = v[b + j + h];
                    v[b + j] = x + y;
                    v[b + j + h] = x - y;
                }
        for (int i = 0; i < 128; ++i) v[b + i] /= std::sqrt(128.0);
    }
}

// y = H (inner (H (suh * x))) * svh for one EXL3 matrix
std::vector<double> exl3_matvec(int type, const uint8_t* trellis, const __half* suh, const __half* svh, const std::vector<double>& x,
                                int n_in, int n_out) {
    std::vector<float> inner((size_t) n_in * n_out);
    basalt::kernels::cpu::exl3_dequant_inner(type, trellis, n_in, n_out, inner.data());
    std::vector<double> xr(n_in), y(n_out, 0.0);
    for (int k = 0; k < n_in; ++k) xr[k] = x[k] * (double) __half2float(suh[k]);
    hadamard(xr);
    for (int r = 0; r < n_out; ++r) {
        double acc = 0;
        for (int k = 0; k < n_in; ++k) acc += (double) inner[(size_t) r * n_in + k] * xr[k];
        y[r] = acc;
    }
    hadamard(y);
    for (int r = 0; r < n_out; ++r) y[r] *= (double) __half2float(svh[r]);
    return y;
}

// a mixed-rate layer as REXL3 quants have them: even experts 2.5 bpw gate/up and 3 bpw down, odd ones 3 and 3.5, their blobs
// back to back at their own sizes; one mixed call against one uniform call per rate pair
void check_exl3_experts(bool bench = false) {
    using basalt::kernels::cpu::kExl3TypeBase;
    const int R = bench ? 64 : 9, NE = bench ? NEXP : 16;
    const int GU[2] = {kExl3TypeBase + 5, kExl3TypeBase + 6}, DN[2] = {kExl3TypeBase + 6, kExl3TypeBase + 7};
    bk::NativeExpertLayout L[2];
    for (int c = 0; c < 2; ++c) {
        L[c] = bk::native_expert_layout(GU[c], DN[c], E, FF);
        L[c].situ_beta = 4.0f;
        L[c].situ_up = 25.0f;
    }
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> mag(0.5f, 1.5f);
    std::vector<uint8_t> blobs;
    std::vector<size_t> off(NE);
    std::vector<int32_t> cls(NE);
    for (int e = 0; e < NE; ++e) {
        const int c = e % 2;
        cls[(size_t) e] = c;
        off[(size_t) e] = blobs.size();
        const size_t vec = L[c].bytes - basalt::kernels::cpu::exl3_vec_bytes(E, FF);
        blobs.resize(blobs.size() + L[c].bytes);
        uint8_t* b = blobs.data() + off[(size_t) e];
        for (size_t i = 0; i < vec; ++i) b[i] = (uint8_t) byte(rng);
        __half* v = (__half*) (b + vec);
        for (int i = 0; i < 3 * (E + FF); ++i) v[i] = __float2half((byte(rng) & 1 ? 1.0f : -1.0f) * mag(rng));
    }
    auto xh = randn((size_t) R * E);
    std::vector<int32_t> ids((size_t) R * KTOP);
    std::vector<float> wts((size_t) R * KTOP);
    for (int r = 0; r < R; ++r) {
        std::vector<int> pick(NE);
        for (int e = 0; e < NE; ++e) pick[e] = e;
        std::shuffle(pick.begin(), pick.end(), rng);
        if (!bench) std::iter_swap(pick.begin(), std::find(pick.begin(), pick.end(), 0));   // a full 8-entry band and its tail
        for (int j = 0; j < KTOP; ++j) { ids[(size_t) r * KTOP + j] = pick[j]; wts[(size_t) r * KTOP + j] = 0.5f + 0.25f * j; }
    }
    uint8_t* base = up(blobs);
    std::vector<unsigned long long> ptrs(NE);
    for (int e = 0; e < NE; ++e) ptrs[(size_t) e] = (unsigned long long) (uintptr_t) (base + off[(size_t) e]);
    unsigned long long* eptr = up(ptrs);
    std::vector<bk::Exl3Rates> rates(NE);
    for (int e = 0; e < NE; ++e) rates[e] = {(uint8_t) (GU[cls[e]] - kExl3TypeBase), (uint8_t) (DN[cls[e]] - kExl3TypeBase)};
    auto* er = up(rates);
    auto* gr = zeros<bk::Exl3Rates>(NE);
    float* x = up(xh);
    int32_t* idd = up(ids);
    float* wd = up(wts);
    auto* gp = zeros<unsigned long long>(NE);
    int32_t *gs = zeros<int32_t>(NE + 1), *ng = zeros<int32_t>(1), *dst = zeros<int32_t>((size_t) R * KTOP), *tok = zeros<int32_t>((size_t) R * KTOP);
    void* scratch = zeros<uint8_t>(std::max(bk::native_expert_scratch_bytes(L[0], (int64_t) R * KTOP), bk::native_expert_scratch_bytes(L[1], (int64_t) R * KTOP)));
    float* eo = zeros<float>((size_t) R * KTOP * E);
    float* y = zeros<float>((size_t) R * E);
    auto run = [&](int row, int rows, float* yy) {
        k::group_experts(idd + row * KTOP, rows, KTOP, NE, eptr, er, gp, gr, gs, ng, dst, tok, nullptr);
        bk::exl3_expert_grouped(L[0], gp, gs, ng, dst, tok, std::min(NE, rows * KTOP), rows * KTOP, x + row * E,
                                scratch, eo, nullptr, 0, gr);
        k::sum_experts(eo, wd + row * KTOP, rows, KTOP, E, yy, nullptr);
    };
    if (bench) {   // a decode step, cycling through distinct picks so the blobs do not all fit in cache
        size_t bytes = 0;
        for (auto e : ids) bytes += L[cls[e]].bytes;
        for (int r = 0; r < R; ++r) run(r, 1, y);
        cudaDeviceSynchronize();
        constexpr int reps = 8;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i)
            for (int r = 0; r < R; ++r) run(r, 1, y);
        if (cudaDeviceSynchronize() != cudaSuccess) { ++failures; return; }
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / (R * reps);
        std::printf("EXL3 group + 8 experts + sum: %.1f us per call, %.0f GB/s of expert weights\n", us, (double) bytes / (R * us * 1e3));
        return;
    }
    run(0, R, y);
    std::vector<double> ref((size_t) R * E, 0.0);
    for (int r = 0; r < R; ++r) {
        const std::vector<double> xr(xh.begin() + (size_t) r * E, xh.begin() + (size_t) (r + 1) * E);
        for (int j = 0; j < KTOP; ++j) {
            const int e = ids[(size_t) r * KTOP + j], c = cls[(size_t) e];
            const uint8_t* b = blobs.data() + off[(size_t) e];
            const __half* v = (const __half*) (b + L[c].bytes - basalt::kernels::cpu::exl3_vec_bytes(E, FF));   // suh_g svh_g suh_u svh_u suh_d svh_d
            const auto g = exl3_matvec(GU[c], b, v, v + E, xr, E, FF);
            const auto u = exl3_matvec(GU[c], b + L[c].up_off, v + E + FF, v + 2 * E + FF, xr, E, FF);
            std::vector<double> hh(FF);
            for (int f = 0; f < FF; ++f) hh[f] = 4.0 * std::tanh(g[f] / 4.0) / (1.0 + std::exp(-g[f])) * 25.0 * std::tanh(u[f] / 25.0);
            const auto d = exl3_matvec(DN[c], b + L[c].down_off, v + 2 * (E + FF), v + 2 * (E + FF) + FF, hh, FF, E);
            for (int cc = 0; cc < E; ++cc) ref[(size_t) r * E + cc] += wts[(size_t) r * KTOP + j] * d[cc];
        }
    }
    const auto got = down(y, ref.size());
    report("EXL3 experts, two rate pairs, situ", got, ref, 1e-3);
    float* y1 = zeros<float>(E);
    for (int r = 0; r < R; ++r) {
        run(r, 1, y1);
        same("EXL3 experts", down(y1, E).data(), got.data() + (size_t) r * E, E);
    }
    for (int c = 0; c < 2; ++c) {
        auto picks = ids;
        for (auto& id : picks) if (cls[id] != c) id = -1;
        k::group_experts(up(picks), R, KTOP, NE, eptr, nullptr, gp, nullptr, gs, ng, dst, tok, nullptr);
        bk::native_expert_grouped(L[c], gp, gs, ng, dst, tok, NE, R * KTOP, x, scratch, eo, nullptr);
    }
    k::sum_experts(eo, wd, R, KTOP, E, y, nullptr);
    const bool identical = std::memcmp(down(y, got.size()).data(), got.data(), got.size() * sizeof(float)) == 0;
    std::printf("EXL3 mixed vs uniform groups: %s\n", identical ? "identical" : "FAIL");
    failures += !identical;
}

void check_logits() {
    const int R = 2, LD = 201024, NV = 200058;
    auto xh = randn((size_t) R * LD, 20.0f);
    float *x = up(xh), *out = zeros<float>((size_t) R * NV);
    k::cap_logits(x, LD, R, NV, 30.0f, out, nullptr);
    std::vector<double> ref((size_t) R * NV);
    for (int r = 0; r < R; ++r)
        for (int i = 0; i < NV; ++i) ref[(size_t) r * NV + i] = 30.0 * std::tanh(xh[(size_t) r * LD + i] / 30.0);
    report("capped logits", down(out, ref.size()), ref, 1e-6);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
        check_exl3_experts(true);
        return failures ? 1 : 0;
    }
    check_mix();
    check_control();
    check_conv();
    check_attention(false, 3);
    check_attention(true, 3);
    check_attention(false, 17);
    check_attention(true, 17);
    check_experts();
    check_exl3_experts();
    check_logits();
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    std::printf(failures ? "FAILED (%d)\n" : "all ok\n", failures);
    return failures ? 1 : 0;
}
