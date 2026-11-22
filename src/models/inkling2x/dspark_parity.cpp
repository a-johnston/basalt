// src/models/inkling2x/dspark_parity.cpp - the kernels of Inkling 2's DSpark drafter against plain CPU references in double,
// at the drafter's geometry: the features kept per member (bf16), their mixes with q8_1 blocks, the query and key heads'
// norms and RoPE into the fp16 caches, the residual norms, the gated MLP's q8_1 input, the mask rows, the block's attention
// (a local layer past its window, a global layer past 512 keys, and a global layer's ring under a window with a floor; no
// sink, no bias, no gate), and the Markov head: its
// context, a step, the previous token's row, the head's screen and the pick over the whole vocabulary.
// A row's result must not depend on the rows beside it: the row checks also run their first row alone.
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/models/inkling2x/dspark_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = basalt::models::inkling2x::kernels;
namespace bk = basalt::kernels;

namespace {

constexpr int E = 2304, H = 24, HKV = 8, D = 128, RANK = 256, NF = 4, B = 8;
constexpr float EPS = 1e-6f, THETA = 1e6f, CAP = 30.0f;
std::mt19937 rng(11);
int failures = 0;

std::vector<float> randn(size_t n, float s = 1.0f) {
    std::normal_distribution<float> d(0.0f, s);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}
template <class T> T* up(const std::vector<T>& v) {
    T* p = nullptr;
    cudaMalloc(&p, std::max<size_t>(v.size(), 1) * sizeof(T));
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T> T* zeros(size_t n) {
    T* p = nullptr;
    cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T));
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
    const double rel = mref > 0 ? mabs / mref : mabs;
    const bool ok = std::isfinite(rel) && rel <= tol;
    std::printf("%-40s max abs %.3e  max rel %.3e  %s\n", what, mabs, rel, ok ? "ok" : "FAIL");
    failures += !ok;
}
void verdict(const char* what, bool ok, const char* good) {
    std::printf("%-40s %s\n", what, ok ? good : "FAIL");
    failures += !ok;
}

// bf16 by round to nearest even, and back
uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
double unbf16(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
// rows of double, normed: x / rms(x) * w
std::vector<double> rmsn(const double* x, int n, const float* w) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    const double iv = 1.0 / std::sqrt(ss / n + (double) EPS);
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) y[i] = x[i] * iv * (w ? w[i] : 1.0);
    return y;
}

// the q8_1 blocks of `rows` rows of n as quantize_q8_1_rows makes them from the same floats
bool q8_same(const float* x_dev, int rows, int n, const void* xq) {
    const size_t bytes = (size_t) rows * n / 32 * 36;
    uint8_t* q = zeros<uint8_t>(bytes);
    bk::quantize_q8_1_rows(x_dev, rows, n, q, nullptr);
    const bool ok = down(q, bytes) == down((const uint8_t*) xq, bytes);
    cudaFree(q);
    return ok;
}

// ------------------------------------------------------------------------------------------------------ features
void check_features() {
    const int R0 = 3, R1 = 2, slot = 2;
    const auto xh = randn((size_t) (R0 + R1) * E, 3.0f);
    const float* x = up(xh);
    k::Members m;
    m.n = 2;
    m.start[1] = R0;
    m.start[2] = R0 + R1;
    uint16_t* d0 = zeros<uint16_t>((size_t) R0 * NF * E);
    uint16_t* d1 = zeros<uint16_t>((size_t) R1 * NF * E);
    k::FeatureDst dst;
    dst.p[0] = d0;
    dst.p[1] = d1;
    k::features_store(x, m, dst, NF, slot, E, nullptr);
    bool ok = true;
    for (int i = 0; i < 2; ++i) {
        const int rows = i ? R1 : R0, first = i ? R0 : 0;
        const auto got = down(i ? d1 : d0, (size_t) rows * NF * E);
        for (int r = 0; r < rows; ++r)
            for (int f = 0; f < NF; ++f)
                for (int c = 0; c < E; ++c) {
                    const uint16_t want = f == slot ? bf16(xh[(size_t) (first + r) * E + c]) : 0;
                    ok &= got[((size_t) r * NF + f) * E + c] == want;
                }
    }
    verdict("features to two members (bf16)", ok, "bit for bit (round to nearest even, other slots kept)");
}

void check_feature_mix() {
    const int R = 5, NM = 3;
    std::vector<uint16_t> fh((size_t) R * NF * E);
    for (int f = 0; f < NF; ++f) {
        const auto v = randn((size_t) R * E, 0.5f + f);
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < E; ++c) fh[((size_t) r * NF + f) * E + c] = bf16(v[(size_t) r * E + c]);
    }
    k::Mixes mx;
    mx.n = NM;
    std::vector<std::vector<float>> wh(NM);
    for (int j = 0; j < NM; ++j) {
        const auto a = randn(NF, 1.0f);
        double s = 0;
        for (int f = 0; f < NF; ++f) s += std::exp(a[(size_t) f]);
        for (int f = 0; f < NF; ++f) mx.alpha[j][f] = (float) (std::exp(a[(size_t) f]) / s);
        wh[j] = randn(E, 0.5f);
        mx.w[j] = up(wh[j]);
    }
    const uint16_t* feat = up(fh);
    float* out = zeros<float>((size_t) NM * R * E);
    void* xq = zeros<uint8_t>((size_t) NM * R * E / 32 * 36);
    k::feature_mix(feat, R, NF, mx, E, EPS, out, xq, nullptr);
    std::vector<double> ref((size_t) NM * R * E);
    for (int j = 0; j < NM; ++j)
        for (int r = 0; r < R; ++r) {
            std::vector<double> v(E, 0.0);
            for (int f = 0; f < NF; ++f)
                for (int c = 0; c < E; ++c) v[(size_t) c] += (double) mx.alpha[j][f] * unbf16(fh[((size_t) r * NF + f) * E + c]);
            const auto n = rmsn(v.data(), E, wh[j].data());
            std::copy(n.begin(), n.end(), ref.begin() + ((size_t) j * R + r) * E);
        }
    const auto got = down(out, ref.size());
    report("feature mixes (3 of 4 features)", got, ref, 1e-5);
    verdict("feature mixes' q8_1 blocks", q8_same(out, NM * R, E, xq), "bit for bit (quantize_q8_1_rows)");
    float* one = zeros<float>((size_t) NM * E);
    k::feature_mix(feat, 1, NF, mx, E, EPS, one, nullptr, nullptr);
    const auto g1 = down(one, (size_t) NM * E);
    bool ok = true;
    for (int j = 0; j < NM; ++j) ok &= std::memcmp(g1.data() + (size_t) j * E, got.data() + (size_t) j * R * E, E * 4) == 0;
    verdict("feature mixes, a row alone", ok, "bit for bit");
}

// ------------------------------------------------------------------------------------------ query and key heads
// neox RoPE of a head at position p in double (the kernel's angles are fp32, as vLLM's table: at p ~ 1100 they round by
// about 1e-4)
std::vector<double> rope(const std::vector<double>& x, int64_t p) {
    std::vector<double> y(D);
    for (int i = 0; i < D / 2; ++i) {
        const double a = (double) p / std::pow((double) THETA, 2.0 * i / D);
        y[(size_t) i] = x[(size_t) i] * std::cos(a) - x[(size_t) i + D / 2] * std::sin(a);
        y[(size_t) i + D / 2] = x[(size_t) i + D / 2] * std::cos(a) + x[(size_t) i] * std::sin(a);
    }
    return y;
}

void check_qk_rope() {
    const int R = B;
    const int64_t pos0 = 1090, cells = 1096;   // the block wraps the ring
    const auto qh = randn((size_t) R * H * D, 2.0f), kh = randn((size_t) R * HKV * D, 2.0f), vh = randn((size_t) R * HKV * D, 1.0f);
    const auto qw = randn(D, 1.0f), kw = randn(D, 1.0f);
    float *q = up(qh), *kd = up(kh), *vd = up(vh), *qwd = up(qw), *kwd = up(kw);
    uint16_t *kc = zeros<uint16_t>((size_t) cells * HKV * D), *vc = zeros<uint16_t>((size_t) cells * HKV * D);
    k::Members one;
    one.start[1] = R;
    int64_t* pos = zeros<int64_t>((size_t) R);
    k::set_positions(pos, one, &pos0, nullptr);
    k::QkArgs qa;
    qa.n_head = H;
    qa.n_kv = HKV;
    qa.theta = THETA;
    qa.eps = EPS;
    k::qk_norm_rope_store(q, qwd, kd, kwd, vd, qa, pos, R, kc, vc, cells, nullptr);
    std::vector<double> qr, kr, vr;
    std::vector<float> kg, vg;
    const auto kb = down(kc, (size_t) cells * HKV * D), vb = down(vc, (size_t) cells * HKV * D);
    for (int r = 0; r < R; ++r) {
        const int64_t p = pos0 + r;
        for (int h = 0; h < H; ++h) {
            std::vector<double> x(qh.begin() + ((size_t) r * H + h) * D, qh.begin() + ((size_t) r * H + h + 1) * D);
            const auto y = rope(rmsn(x.data(), D, qw.data()), p);
            qr.insert(qr.end(), y.begin(), y.end());
        }
        const size_t cell = (size_t) (p % cells);
        for (int h = 0; h < HKV; ++h) {
            std::vector<double> x(kh.begin() + ((size_t) r * HKV + h) * D, kh.begin() + ((size_t) r * HKV + h + 1) * D);
            const auto y = rope(rmsn(x.data(), D, kw.data()), p);
            kr.insert(kr.end(), y.begin(), y.end());
            for (int d = 0; d < D; ++d) {
                kg.push_back(__half2float(__ushort_as_half(kb[(cell * HKV + h) * D + d])));
                vg.push_back(__half2float(__ushort_as_half(vb[(cell * HKV + h) * D + d])));
                vr.push_back(vh[((size_t) r * HKV + h) * D + d]);
            }
        }
    }
    report("query heads: norm and RoPE", down(q, qr.size()), qr, 5e-4);
    report("key heads: norm and RoPE, fp16 cells", kg, kr, 1e-3);
    report("value heads, fp16 cells", vg, vr, 1e-3);
    // the context's keys alone (no queries): the same cells
    uint16_t *kc2 = zeros<uint16_t>((size_t) cells * HKV * D), *vc2 = zeros<uint16_t>((size_t) cells * HKV * D);
    k::qk_norm_rope_store(nullptr, nullptr, kd, kwd, vd, qa, pos, R, kc2, vc2, cells, nullptr);
    verdict("keys and values without queries", down(kc2, kb.size()) == kb && down(vc2, vb.size()) == vb, "bit for bit");
}

// ---------------------------------------------------------------------------------------- residual norms, MLP
void check_add_norm() {
    const int R = B;
    const auto rh = randn((size_t) R * E, 2.0f), ah = randn((size_t) R * E, 1.0f), wh = randn(E, 0.5f);
    float *res = up(rh), *add = up(ah), *w = up(wh), *out = zeros<float>((size_t) R * E);
    void* xq = zeros<uint8_t>((size_t) R * E / 32 * 36);
    k::add_norm_quant(res, add, w, R, E, EPS, out, xq, nullptr);
    const auto rg = down(res, rh.size());
    bool sum_ok = true;
    std::vector<double> ref((size_t) R * E);
    for (size_t i = 0; i < rh.size(); ++i) sum_ok &= rg[i] == rh[i] + ah[i];
    for (int r = 0; r < R; ++r) {
        std::vector<double> x(rg.begin() + (size_t) r * E, rg.begin() + (size_t) (r + 1) * E);
        const auto n = rmsn(x.data(), E, wh.data());
        std::copy(n.begin(), n.end(), ref.begin() + (size_t) r * E);
    }
    verdict("residual sum", sum_ok, "bit for bit (fp32 adds)");
    const auto got = down(out, ref.size());
    report("residual norm", got, ref, 1e-6);
    verdict("residual norm's q8_1 blocks", q8_same(out, R, E, xq), "bit for bit (quantize_q8_1_rows)");
    void* xq2 = zeros<uint8_t>((size_t) R * E / 32 * 36);
    k::add_norm_quant(res, nullptr, w, R, E, EPS, nullptr, xq2, nullptr);   // nothing to add, no floats out
    verdict("residual norm, no add, q8_1 only", down((uint8_t*) xq2, (size_t) R * E / 32 * 36) == down((uint8_t*) xq, (size_t) R * E / 32 * 36),
            "bit for bit");
    float* one = zeros<float>((size_t) E);
    float* res1 = up(std::vector<float>(rg.begin(), rg.begin() + E));
    k::add_norm_quant(res1, nullptr, w, 1, E, EPS, one, nullptr, nullptr);
    verdict("residual norm, a row alone", std::memcmp(down(one, (size_t) E).data(), got.data(), E * 4) == 0, "bit for bit");
}

void check_swiglu() {
    const int R = B, N = 9216;
    const auto gh = randn((size_t) R * N, 2.0f), uh = randn((size_t) R * N, 1.0f);
    void* xq = zeros<uint8_t>((size_t) R * N / 32 * 36);
    k::swiglu_quant(up(gh), up(uh), R, N, xq, nullptr);
    const auto qb = down((uint8_t*) xq, (size_t) R * N / 32 * 36);
    std::vector<float> got((size_t) R * N);
    std::vector<double> ref((size_t) R * N);
    bool sums = true;
    for (size_t b = 0; b < got.size() / 32; ++b) {
        uint16_t ds[2];
        std::memcpy(ds, qb.data() + b * 36, 4);
        const float d = __half2float(__ushort_as_half(ds[0]));
        double s = 0;
        for (int i = 0; i < 32; ++i) {
            const size_t at = b * 32 + (size_t) i;
            got[at] = d * (float) (int8_t) qb[b * 36 + 4 + (size_t) i];
            const double g = gh[at];
            ref[at] = g / (1.0 + std::exp(-g)) * uh[at];
            s += ref[at];
        }
        // the block's sum (of the values before rounding to int8), in fp16
        sums &= std::fabs((double) __half2float(__ushort_as_half(ds[1])) - s) <= 2e-3 * std::max(1.0, std::fabs(s)) + 32 * d;
    }
    report("silu(gate) up as q8_1 (dequantized)", got, ref, 4e-3);
    verdict("silu(gate) up, the blocks' sums", sums, "ok");
}

void check_mask_rows() {
    const auto xh = randn((size_t) B * E), mh = randn(E);
    float* x = up(xh);
    k::fill_mask_rows(x, up(mh), B, E, nullptr);
    const auto g = down(x, xh.size());
    bool ok = std::memcmp(g.data(), xh.data(), E * 4) == 0;
    for (int r = 1; r < B; ++r) ok &= std::memcmp(g.data() + (size_t) r * E, mh.data(), E * 4) == 0;
    verdict("mask rows", ok, "bit for bit (row 0 kept)");
}

// ------------------------------------------------------------------------------------------------ the block's attention
// Every row of the block at position p0..p0+7 attends the keys up to the block's last (p0 + 7): the context's and the
// block's own. A local layer sees the last `window` of them; a global layer's ring (--mtp-window) too, from its floor on.
enum class Layer { Local, Global, Ring };
void check_block_attention(Layer kind) {
    const bool local = kind == Layer::Local, ring = kind == Layer::Ring;
    const char* name = local ? "block attention, local" : ring ? "block attention, ring" : "block attention, global";
    // a local layer past its window; a global one past 512 keys; a ring wrapped several times, its floor inside the window
    const int64_t p0 = local ? 1500 : ring ? 9000 : 700;
    const int64_t plast = p0 + B - 1;
    const int window = local ? 1024 + B : ring ? 2048 + B : 0;
    const int64_t cells = local ? window + 64 : ring ? window + 128 : 4096;
    const int64_t floor = ring ? plast - 700 : 0;
    std::vector<uint16_t> kc((size_t) cells * HKV * D), vc((size_t) cells * HKV * D);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t i = 0; i < kc.size(); ++i) {
        kc[i] = __half_as_ushort(__float2half(nd(rng)));
        vc[i] = __half_as_ushort(__float2half(nd(rng)));
    }
    const auto qh = randn((size_t) B * H * D, 0.3f);
    k::KvCaches kv;
    kv.k[0] = up(kc);
    kv.v[0] = up(vc);
    k::AttnArgs aa;
    aa.n_head = H;
    aa.n_kv = HKV;
    aa.d_rel = 0;
    aa.extent = 0;
    aa.window = window;
    aa.cells = cells;
    aa.sink_logit = false;
    int64_t* d_floor = ring ? up(std::vector<int64_t>{floor}) : nullptr;
    aa.floor = d_floor;
    k::Members one, each;
    one.start[1] = B;
    each.n = B;
    for (int i = 0; i <= B; ++i) each.start[i] = i;
    const std::vector<int64_t> last((size_t) B, plast);
    int64_t* pos = zeros<int64_t>((size_t) B);
    k::set_positions(pos, each, last.data(), nullptr);
    void* ws = zeros<uint8_t>(k::attention_workspace_bytes(B, H, plast + 1));
    float* out = zeros<float>((size_t) B * H * D);
    void* oq = zeros<uint8_t>((size_t) B * H * D / 32 * 36);
    float* q = up(qh);
    k::attention(q, nullptr, nullptr, nullptr, nullptr, kv, aa, one, pos, plast, ws, out, oq, nullptr);
    const int64_t lo = std::max(window > 0 && plast + 1 > window ? plast - window + 1 : 0, floor);
    std::vector<double> ref((size_t) B * H * D);
    for (int r = 0; r < B; ++r)
        for (int h = 0; h < H; ++h) {
            const int g = h / (H / HKV);
            std::vector<double> s;
            double m = -1e300;
            for (int64_t j = lo; j <= plast; ++j) {
                const size_t cell = (size_t) (j % cells);
                double dot = 0;
                for (int d = 0; d < D; ++d)
                    dot += (double) qh[((size_t) r * H + h) * D + d] * __half2float(__ushort_as_half(kc[(cell * HKV + g) * D + d]));
                s.push_back(dot / std::sqrt((double) D));
                m = std::max(m, s.back());
            }
            double l = 0;
            std::vector<double> o(D, 0.0);
            for (int64_t j = lo; j <= plast; ++j) {
                const double e = std::exp(s[(size_t) (j - lo)] - m);
                l += e;
                const size_t cell = (size_t) (j % cells);
                for (int d = 0; d < D; ++d) o[(size_t) d] += e * __half2float(__ushort_as_half(vc[(cell * HKV + g) * D + d]));
            }
            for (int d = 0; d < D; ++d) ref[((size_t) r * H + h) * D + d] = o[(size_t) d] / l;
        }
    const auto got = down(out, ref.size());
    const std::string n(name);
    report((n + (local ? " (past window)" : ring ? " (floor in window)" : " (> 512 keys)")).c_str(), got, ref, 1e-4);
    verdict((n + ": q8_1 blocks").c_str(), q8_same(out, B, H * D, oq), "bit for bit (quantize_q8_1_rows)");
    k::Members m1;
    m1.start[1] = 1;
    float* o1 = zeros<float>((size_t) H * D);
    k::attention(q, nullptr, nullptr, nullptr, nullptr, kv, aa, m1, pos, plast, ws, o1, nullptr, nullptr);
    verdict((n + ": a row alone").c_str(), std::memcmp(down(o1, (size_t) H * D).data(), got.data(), (size_t) H * D * 4) == 0,
            "bit for bit");
    if (ring) {   // a floor below the window changes nothing: the launch without one, bit for bit
        cudaMemcpy(d_floor, std::vector<int64_t>{plast - window - 5}.data(), 8, cudaMemcpyHostToDevice);
        float* ob = zeros<float>((size_t) B * H * D), *on = zeros<float>((size_t) B * H * D);
        k::attention(q, nullptr, nullptr, nullptr, nullptr, kv, aa, one, pos, plast, ws, ob, nullptr, nullptr);
        aa.floor = nullptr;
        k::attention(q, nullptr, nullptr, nullptr, nullptr, kv, aa, one, pos, plast, ws, on, nullptr, nullptr);
        verdict((n + ": a floor below the window").c_str(),
                std::memcmp(down(ob, ref.size()).data(), down(on, ref.size()).data(), ref.size() * 4) == 0, "bit for bit (no floor)");
    }
}

// ---------------------------------------------------------------------------------------------------- Markov head
std::vector<float> Wh;   // rnn_proj [3 rank][2 rank + 2 E]
const int WC = 2 * RANK + 2 * E;

void check_markov_context() {
    const int n = B - 1;
    const auto bh = randn(3 * RANK, 0.1f), hh = randn((size_t) B * E), fh = randn(E);
    float* out = zeros<float>((size_t) n * 3 * RANK);
    const float* h = up(hh);
    k::markov_context(up(Wh), up(bh), h + E, E, up(fh), n, RANK, E, out, nullptr);   // the mask rows (row 1 on)
    std::vector<double> ref((size_t) n * 3 * RANK);
    for (int s = 0; s < n; ++s)
        for (int o = 0; o < 3 * RANK; ++o) {
            double a = bh[(size_t) o];
            for (int c = 0; c < E; ++c)
                a += (double) Wh[(size_t) o * WC + 2 * RANK + c] * hh[(size_t) (s + 1) * E + c] +
                     (double) Wh[(size_t) o * WC + 2 * RANK + E + c] * fh[(size_t) c];
            ref[(size_t) s * 3 * RANK + o] = a;
        }
    report("Markov context (7 steps)", down(out, ref.size()), ref, 1e-5);
}

void check_markov_step() {
    const auto ch = randn(3 * RANK, 1.0f), sh = randn(RANK, 0.5f), eh = randn(RANK, 0.5f);
    float *raw = zeros<float>(3 * RANK), *state = up(sh), *lat = zeros<float>(RANK);
    k::markov_step(up(ch), up(Wh), up(eh), RANK, E, raw, state, lat, nullptr);
    std::vector<double> sr(RANK), lr(RANK);
    for (int o = 0; o < RANK; ++o) {
        double r[3];
        for (int part = 0; part < 3; ++part) {
            const int row = part * RANK + o;
            double a = ch[(size_t) row];
            for (int i = 0; i < RANK; ++i)
                a += (double) Wh[(size_t) row * WC + i] * sh[(size_t) i] + (double) Wh[(size_t) row * WC + RANK + i] * eh[(size_t) i];
            r[part] = a;
        }
        const double g = 1.0 / (1.0 + std::exp(-r[0]));
        sr[(size_t) o] = g * sh[(size_t) o] + (1.0 - g) * std::tanh(r[1]);
        lr[(size_t) o] = std::tanh(r[2]);
    }
    report("Markov step: state", down(state, (size_t) RANK), sr, 1e-5);
    report("Markov step: latent", down(lat, (size_t) RANK), lr, 1e-5);
}

// ------------------------------------------------------------------------------------------- the head's screen
// A Q8_0 head of n rows of 2304 (random codes, fp16 scales) and its q8_1 activations, on the card.
struct Head {
    int n;
    std::vector<uint8_t> w;   // [n][72] blocks of {fp16 d, int8 qs[32]}
    uint8_t *dw = nullptr, *screen = nullptr;
};
Head make_head(int n, float scale) {
    Head h;
    h.n = n;
    h.w.resize((size_t) n * 72 * 34);
    std::uniform_int_distribution<int> q(-127, 127);
    std::uniform_real_distribution<float> d(0.5f * scale, 1.5f * scale);
    for (size_t b = 0; b < (size_t) n * 72; ++b) {
        const __half hd = __float2half(d(rng));
        std::memcpy(&h.w[b * 34], &hd, 2);
        for (int i = 0; i < 32; ++i) h.w[b * 34 + 2 + (size_t) i] = (uint8_t) (int8_t) q(rng);
    }
    h.dw = up(h.w);
    h.screen = zeros<uint8_t>(k::head_screen_bytes(n));
    k::head_screen_pack(h.dw, n, h.screen, nullptr);
    return h;
}
// q8_1 rows of x (rows x 2304) made on the card, and their values {d, qs} back
struct Act {
    void* dq = nullptr;
    std::vector<uint8_t> q;
};
Act make_act(int rows) {
    Act a;
    const float* x = up(randn((size_t) rows * E, 1.0f));
    a.dq = zeros<uint8_t>((size_t) rows * 72 * 36);
    bk::quantize_q8_1_rows(x, rows, E, a.dq, nullptr);
    a.q = down((const uint8_t*) a.dq, (size_t) rows * 72 * 36);
    return a;
}
double half_at(const uint8_t* p) {
    uint16_t u;
    std::memcpy(&u, p, 2);
    return __half2float(__ushort_as_half(u));
}
// the exact logit of head row r against activation row a, in double
double exact_logit(const Head& h, const Act& x, int r, int a) {
    double t = 0;
    for (int b = 0; b < 72; ++b) {
        const uint8_t* wb = &h.w[((size_t) r * 72 + b) * 34];
        const uint8_t* xb = &x.q[((size_t) a * 72 + b) * 36];
        long sumi = 0;
        for (int i = 0; i < 32; ++i) sumi += (long) (int8_t) wb[2 + i] * (long) (int8_t) xb[4 + i];
        t += half_at(wb) * half_at(xb) * (double) sumi;
    }
    return t;
}

// The screen's scores against the copy it reads, decoded on the host in double: head_proxy.cuh's layout (tiles of 32
// rows; a row's words two groups a uint4 at lane = row, levels 2 bits from bit 2 ii of byte bb for value c 16 + 4 ii +
// bb; an fp16 scale a group) times the activations' q8_1 values.  Then how far the copy is from the head itself.
void check_head_screen() {
    const int N = 1000, R = 3;   // a tile's tail
    Head h = make_head(N, 0.002f);
    Act x = make_act(R);
    float* out = zeros<float>((size_t) R * N);
    k::head_screen_scores(h.screen, x.dq, R, N, out, N, nullptr);
    const auto got = down(out, (size_t) R * N);
    const auto cp = down(h.screen, k::head_screen_bytes(N));
    const int tiles = (N + 31) / 32, nw4 = E / 64;
    const uint32_t* wq = (const uint32_t*) cp.data();
    const uint8_t* sc = cp.data() + (size_t) tiles * nw4 * 32 * 16;
    std::vector<double> ref((size_t) R * N);
    double err = 0, top = 0;
    for (int r = 0; r < N; ++r) {
        const int tile = r >> 5, lane = r & 31;
        for (int a = 0; a < R; ++a) {
            double t = 0;
            for (int g = 0; g < 72; ++g) {
                const int w4 = g / 2, hh = g & 1;
                const size_t at = ((size_t) tile * nw4 + w4) * 32 + lane;
                const double sg = half_at(sc + (at * 2 + hh) * 2);
                const uint8_t* xb = &x.q[((size_t) a * 72 + g) * 36];
                for (int c = 0; c < 2; ++c) {
                    const uint32_t word = wq[at * 4 + 2 * hh + c];
                    for (int ii = 0; ii < 4; ++ii)
                        for (int bb = 0; bb < 4; ++bb) {
                            const int lvl = (int) ((word >> (8 * bb + 2 * ii)) & 3u);
                            t += sg * (lvl - 1.5) * half_at(xb) * (double) (int8_t) xb[4 + c * 16 + 4 * ii + bb];
                        }
                }
            }
            ref[(size_t) a * N + r] = t;
            const double e = exact_logit(h, x, r, a);
            err = std::max(err, std::fabs(t - e));
            top = std::max(top, std::fabs(e));
        }
    }
    report("head screen: scores of its copy (3 rows)", got, ref, 1e-5);
    std::printf("%-40s the copy's largest error %.3f against logits up to %.3f (approximate by design)\n", "head screen vs the head", err, top);
}

// The pick on a 50003-token head: the screen's approximate base, the Markov bias, the best rows rescored on the head;
// against the exact capped logits of every token in double.
void check_markov_pick(bool last_wins) {
    const int NV = 50003;   // past whole sweeps of the grid, and past the last group of four rows
    Head h = make_head(NV, 0.002f);
    Act x = make_act(1);
    std::vector<uint16_t> fold((size_t) NV * RANK), w1((size_t) NV * RANK);
    {
        const auto f = randn(fold.size(), 0.05f), e = randn(w1.size(), 0.5f);
        for (size_t i = 0; i < fold.size(); ++i) {
            fold[i] = bf16(f[i]);
            w1[i] = bf16(e[i]);
        }
    }
    auto fb = randn(NV, 0.5f);
    if (last_wins) fb[(size_t) NV - 1] = 60.0f;   // the vocabulary's last token
    std::vector<float> lat(RANK);
    for (float& v : lat) v = std::tanh(randn(1, 1.0f)[0]);
    float* base = zeros<float>((size_t) NV);
    k::head_screen_scores(h.screen, x.dq, 1, NV, base, NV, nullptr);
    float* scratch = zeros<float>(k::markov_pick_scratch_floats(NV));
    int32_t *tok = zeros<int32_t>(1), *prev = zeros<int32_t>(1);
    float *prob = zeros<float>(1), *emb = zeros<float>(RANK);
    k::markov_pick(base, h.dw, x.dq, up(fold), up(fb), up(lat), NV, CAP, scratch, up(w1), tok, prob, prev, emb, nullptr);
    std::vector<double> xv((size_t) NV);
    int best = 0;
    for (int v = 0; v < NV; ++v) {
        double d = 0;
        for (int i = 0; i < RANK; ++i) d += unbf16(fold[(size_t) v * RANK + i]) * lat[(size_t) i];
        xv[(size_t) v] = CAP * std::tanh((exact_logit(h, x, v, 0) + d + fb[(size_t) v]) / CAP);
        if (xv[(size_t) v] > xv[(size_t) best]) best = v;
    }
    double second = -1e300, l = 0;
    for (int v = 0; v < NV; ++v) {
        if (v != best) second = std::max(second, xv[(size_t) v]);
        l += std::exp(xv[(size_t) v] - xv[(size_t) best]);
    }
    const int t = down(tok, 1)[0], pv = down(prev, 1)[0];
    const char* name = last_wins ? "Markov pick, screened (the last wins)" : "Markov pick, screened (50003 tokens)";
    // a pick within fp32's reach of the best (a gap under 1e-4) is a tie the reference cannot order
    const bool tok_ok = t == best || (t >= 0 && t < NV && xv[(size_t) best] - xv[(size_t) t] < 1e-4);
    std::printf("%-40s token %d (reference %d, gap to the second %.2e)  %s\n", name, t, best, xv[(size_t) best] - second, tok_ok ? "ok" : "FAIL");
    failures += !tok_ok;
    // the probability: the pick's exact logit against the copy's sum with the rescored rows' terms exact, within 10%
    const double p = down(prob, 1)[0], pr = 1.0 / l;
    const bool p_ok = p > 0 && p <= 1 && p / pr < 1.1 && pr / p < 1.1;
    std::printf("%-40s %.4f (reference %.4f)  %s\n", last_wins ? "  its probability (approximate)" : "  its probability (approximate)", p, pr,
                p_ok ? "ok" : "FAIL");
    failures += !p_ok;
    std::vector<double> er(RANK);
    for (int i = 0; i < RANK; ++i) er[(size_t) i] = unbf16(w1[(size_t) std::max(t, 0) * RANK + i]);
    const auto eg = down(emb, (size_t) RANK);
    bool emb_ok = pv == t;
    for (int i = 0; i < RANK; ++i) emb_ok &= eg[(size_t) i] == (float) er[(size_t) i];
    verdict("  its next inputs", emb_ok, "the token's row, bit for bit");
    float* e2 = zeros<float>(RANK);
    k::markov_embed(up(w1), tok, RANK, e2, nullptr);
    verdict("  Markov embed", down(e2, (size_t) RANK) == eg, "bit for bit");
}

}  // namespace

int main() {
    Wh = randn((size_t) 3 * RANK * WC, 0.02f);
    check_features();
    check_feature_mix();
    check_qk_rope();
    check_add_norm();
    check_swiglu();
    check_mask_rows();
    check_block_attention(Layer::Local);
    check_block_attention(Layer::Global);
    check_markov_context();
    check_markov_step();
    check_head_screen();
    check_markov_pick(false);
    check_markov_pick(true);
    check_block_attention(Layer::Ring);   // last: the checks before it keep their inputs
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    std::printf(failures ? "FAILED (%d)\n" : "all ok\n", failures);
    return failures ? 1 : 0;
}
