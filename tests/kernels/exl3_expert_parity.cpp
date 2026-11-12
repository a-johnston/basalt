// tests/kernels/exl3_expert_parity.cpp - the EXL3 experts (exl3_cpu.hpp), every path.
//
//     build/exl3_expert_parity                 random blobs at the model's geometry, every bitrate
//     build/exl3_expert_parity --cpu-bench     the CPU rows per ISA on one core: us per expert and GB/s
//     build/exl3_expert_parity --bench         the grouped decode kernels and the dequantizer: us per call, GB/s
//     build/exl3_expert_parity --io BLOB GU D OUT   one real expert blob (types GU / D) on fixed inputs: writes
//                                              x (NT x H floats) then y (NT x H floats) to OUT, for a comparison
//                                              with ExLlamaV3 (tools/convert/exl3_upstream_check.py --expert-io)
//
// (a) the decoder against checksums computed by tools/convert/exl3.py, which is checked against ExLlamaV3.
// (b) the reference: exl3_dequant_inner (the fp16 codebook values), the transforms and a SwiGLU expert in double.
// (c) the CPU path (native_quant_act / native_gu_rows / native_quant_h / native_down_rows): within a tolerance of
// (b) (int16 activations, the codebook before its fp16 rounding), every ISA bit for bit, a token alone against the
// same token in a group bit for bit, row ranges against the whole bit for bit.
// (d) the GPU path (native_expert_act_rows / native_expert_grouped: the fp16 codebook values, hi + lo fp16
// activations on the tensor cores) against (b); an entry alone against the same entry in a call of several groups,
// and one block row striding over the groups, bit for bit.  (e) the prompt path's dequantizer against the
// original-basis weights in double.
#include "basalt/kernels/cpu/exl3_cpu.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/exl3_kernels.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = basalt::kernels::cpu;
namespace k = basalt::kernels;

namespace {
constexpr int NT = 3;
constexpr int H = 2560, FF = 640;
constexpr int kRates[] = {2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16};

struct Err { double max_abs = 0, rel = 0; };
Err err_of(const std::vector<float>& a, const std::vector<double>& b) {
    Err e;
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = std::fabs((double) a[i] - b[i]);
        if (x > e.max_abs) e.max_abs = x;
        n += x;
        d += std::fabs(b[i]);
    }
    e.rel = n / (d + 1e-30);
    return e;
}

void put_f16(uint8_t* p, float v) {
    const uint16_t b = k::f16_from_f32(v);
    std::memcpy(p, &b, 2);
}
double get_f16(const uint8_t* p, int i) {
    uint16_t b;
    std::memcpy(&b, p + 2 * i, 2);
    return (double) k::f32_from_f16(b);
}

// any bit stream is a valid trellis; the vectors are signed, suh small and svh near 1, as in real files
void random_blob(const cpu::NativeFmt& f, uint8_t* blob, std::mt19937& rng) {
    const size_t tr = f.bytes - f.tail;
    for (size_t i = 0; i < tr; ++i) blob[i] = (uint8_t) rng();
    std::uniform_real_distribution<float> su(0.004f, 0.02f), sv(0.6f, 1.6f);
    uint8_t* v = blob + tr;
    const int n[6] = {H, FF, H, FF, FF, H};
    for (int m = 0; m < 6; ++m)
        for (int i = 0; i < n[m]; ++i, v += 2) put_f16(v, ((rng() & 1) ? -1.f : 1.f) * (m % 2 ? sv(rng) : su(rng)));
}

void hadamard_d(double* v, int n) {
    for (int b0 = 0; b0 < n; b0 += 128) {
        for (int len = 1; len < 128; len <<= 1)
            for (int i = 0; i < 128; i += len << 1)
                for (int j = i; j < i + len; ++j) {
                    const double a = v[b0 + j], c = v[b0 + j + len];
                    v[b0 + j] = a + c;
                    v[b0 + j + len] = a - c;
                }
        for (int i = 0; i < 128; ++i) v[b0 + i] /= std::sqrt(128.0);
    }
}

// y = (H (inner (H (suh * x)))) * svh in double; inner is [n_out][n_in]
void matrix_d(const std::vector<float>& inner, const uint8_t* suh, const uint8_t* svh, int n_in, int n_out,
              const double* x, double* y) {
    std::vector<double> xr((size_t) n_in);
    for (int i = 0; i < n_in; ++i) xr[(size_t) i] = x[i] * get_f16(suh, i);
    hadamard_d(xr.data(), n_in);
    for (int r = 0; r < n_out; ++r) {
        double s = 0;
        const float* w = inner.data() + (size_t) r * n_in;
        for (int i = 0; i < n_in; ++i) s += (double) w[i] * xr[(size_t) i];
        y[r] = s;
    }
    hadamard_d(y, n_out);
    for (int r = 0; r < n_out; ++r) y[r] *= get_f16(svh, r);
}

struct Ref {
    std::vector<float> g, u, d;   // inner matrices
    void decode(const cpu::NativeFmt& f, const uint8_t* blob) {
        g.resize((size_t) H * FF);
        u.resize((size_t) H * FF);
        d.resize((size_t) H * FF);
        cpu::exl3_dequant_inner(f.gu_type, blob, H, FF, g.data());
        cpu::exl3_dequant_inner(f.gu_type, blob + f.up_off, H, FF, u.data());
        cpu::exl3_dequant_inner(f.d_type, blob + f.down_off, FF, H, d.data());
    }
    void run(const cpu::NativeFmt& f, const uint8_t* blob, const float* x, double* y) const {
        const uint8_t* v = blob + f.bytes - f.tail;
        std::vector<double> xd(x, x + H), gg((size_t) FF), uu((size_t) FF), hh((size_t) FF);
        matrix_d(g, v, v + 2 * H, H, FF, xd.data(), gg.data());
        matrix_d(u, v + 2 * (H + FF), v + 2 * (2 * H + FF), H, FF, xd.data(), uu.data());
        for (int r = 0; r < FF; ++r) hh[(size_t) r] = gg[(size_t) r] / (1.0 + std::exp(-gg[(size_t) r])) * uu[(size_t) r];
        matrix_d(d, v + 4 * (H + FF), v + 4 * (H + FF) + 2 * FF, FF, H, hh.data(), y);
    }
};

// the CPU expert on `nt` tokens of x, gate/up rows in `parts` ranges and down rows in parts + 1
void cpu_expert(const cpu::NativeFmt& f, cpu::Exl3Isa isa, const uint8_t* blob, const float* x, int nt, int parts,
                std::vector<float>& out) {
    std::vector<std::vector<float>> ff((size_t) nt, std::vector<float>((size_t) FF));
    const float* xp[cpu::kExl3MaxTokens];
    const float* hp[cpu::kExl3MaxTokens];
    float* fp[cpu::kExl3MaxTokens];
    float* op[cpu::kExl3MaxTokens];
    out.assign((size_t) nt * H, 0.f);
    for (int t = 0; t < nt; ++t) {
        xp[t] = x + (size_t) t * H;
        fp[t] = ff[(size_t) t].data();
        hp[t] = fp[t];
        op[t] = out.data() + (size_t) t * H;
    }
    for (int p = 0; p < parts; ++p) cpu::exl3_gu_rows(isa, f, blob, xp, nt, fp, FF * p / parts, FF * (p + 1) / parts);
    for (int p = 0; p <= parts; ++p)
        cpu::exl3_down_rows(isa, f, blob, hp, nt, op, H * p / (parts + 1), H * (p + 1) / (parts + 1));
}

void inputs(std::vector<float>& x, int nt, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    x.resize((size_t) nt * H);
    for (auto& v : x) v = nd(rng);
    // a dominant component, as hidden states have: its Hadamard block is far larger than the others
    for (int t = 0; t < nt; ++t) x[(size_t) t * H + 700] = 60.f + (float) t;
}

#define CK(x) do { if ((x) != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(cudaGetLastError())); std::exit(1); } } while (0)

// device buffers for `groups` blobs and `cap` entries of up to `nt` tokens
struct Gpu {
    cpu::NativeFmt f;
    k::NativeExpertLayout L;
    int groups = 0, cap = 0;
    uint8_t* blobs = nullptr;
    void *act = nullptr, *scr = nullptr;
    float *x = nullptr, *out = nullptr;
    uint16_t *dq_gu = nullptr, *dq_d = nullptr;
    unsigned long long* ptr = nullptr;
    int32_t *start = nullptr, *count = nullptr, *dst = nullptr, *tok = nullptr;
    Gpu(const cpu::NativeFmt& fmt, int groups_, int cap_, int nt) : f(fmt), groups(groups_), cap(cap_) {
        L = k::native_expert_layout(f);
        CK(cudaMalloc(&blobs, f.bytes * groups));
        CK(cudaMalloc(&x, (size_t) nt * H * 4));
        CK(cudaMalloc(&act, (size_t) nt * k::native_expert_act_bytes(f)));
        CK(cudaMalloc(&scr, k::native_expert_scratch_bytes(L, cap)));
        CK(cudaMalloc(&out, (size_t) cap * H * 4));
        CK(cudaMalloc(&dq_gu, (size_t) 2 * FF * H * 2));
        CK(cudaMalloc(&dq_d, (size_t) FF * H * 2));
        CK(cudaMalloc(&ptr, (size_t) groups * 8));
        CK(cudaMalloc(&start, (size_t) (groups + 1) * 4));
        CK(cudaMalloc(&count, 4));
        CK(cudaMalloc(&dst, (size_t) cap * 4));
        CK(cudaMalloc(&tok, (size_t) cap * 4));
    }
    ~Gpu() {
        for (void* p : {(void*) blobs, act, scr, (void*) x, (void*) out, (void*) dq_gu, (void*) dq_d, (void*) ptr,
                        (void*) start, (void*) count, (void*) dst, (void*) tok})
            cudaFree(p);
    }
    // groups given as (blob index, its entries' tokens); the result row of entry i is i
    void run(const std::vector<std::pair<int, std::vector<int>>>& g, const float* xh, int nt, std::vector<float>& res,
             int64_t grid_groups, cudaStream_t s) {
        std::vector<unsigned long long> hp;
        std::vector<int32_t> hs, hd, ht;
        for (const auto& [b, toks] : g) {
            hp.push_back((unsigned long long) (blobs + f.bytes * b));
            hs.push_back((int32_t) ht.size());
            for (const int t : toks) { hd.push_back((int32_t) ht.size()); ht.push_back(t); }
        }
        hs.push_back((int32_t) ht.size());
        const int32_t ng = (int32_t) g.size();
        CK(cudaMemcpy(ptr, hp.data(), hp.size() * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(start, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(count, &ng, 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dst, hd.data(), hd.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(tok, ht.data(), ht.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(x, xh, (size_t) nt * H * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(out, 0, (size_t) cap * H * 4));
        k::native_expert_act_rows(f, x, nt, act, s);
        k::native_expert_grouped(L, ptr, start, count, dst, tok, groups, cap, act, scr, out, s, grid_groups);
        res.resize(ht.size() * H);
        CK(cudaMemcpyAsync(res.data(), out, res.size() * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
    }
};

// W = diag(suh) H inner H diag(svh) in double, w[r * n_in + k] (row r = output)
std::vector<double> weight_d(const std::vector<float>& inner, const uint8_t* suh, const uint8_t* svh, int n_in, int n_out) {
    std::vector<double> w(inner.begin(), inner.end()), col((size_t) n_out);
    for (int r = 0; r < n_out; ++r) {
        hadamard_d(w.data() + (size_t) r * n_in, n_in);
        for (int i = 0; i < n_in; ++i) w[(size_t) r * n_in + i] *= get_f16(suh, i);
    }
    for (int i = 0; i < n_in; ++i) {
        for (int r = 0; r < n_out; ++r) col[(size_t) r] = w[(size_t) r * n_in + i];
        hadamard_d(col.data(), n_out);
        for (int r = 0; r < n_out; ++r) w[(size_t) r * n_in + i] = col[(size_t) r] * get_f16(svh, r);
    }
    return w;
}

int gpu_parity(int k2, cudaStream_t s) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(cpu::kExl3TypeBase + k2, cpu::kExl3TypeBase + k2, H, FF, f, err)) return 1;
    if (!k::native_expert_supported(f.gu_type, f.d_type, H, FF)) { std::printf("  gpu: %.1f bpw not supported\n", k2 / 2.0); return 1; }
    int failures = 0;
    std::mt19937 rng(200 + (unsigned) k2);
    std::vector<uint8_t> blob[2] = {std::vector<uint8_t>(f.bytes), std::vector<uint8_t>(f.bytes)};
    Gpu g(f, 2, 8, NT);
    Ref ref[2];
    for (int b = 0; b < 2; ++b) {
        random_blob(f, blob[b].data(), rng);
        ref[b].decode(f, blob[b].data());
        CK(cudaMemcpy(g.blobs + f.bytes * b, blob[b].data(), f.bytes, cudaMemcpyHostToDevice));
    }
    std::vector<float> x;
    inputs(x, NT, rng);
    // five entries of one expert (a pass of four and one of one) and three of another (two and one)
    const std::vector<std::pair<int, std::vector<int>>> plan = {{0, {0, 1, 2, 0, 1}}, {1, {2, 0, 1}}};
    std::vector<double> want;
    for (const auto& [b, toks] : plan)
        for (const int t : toks) {
            want.resize(want.size() + H);
            ref[b].run(f, blob[b].data(), x.data() + (size_t) t * H, want.data() + want.size() - H);
        }
    std::vector<float> got, one, strided;
    g.run(plan, x.data(), NT, got, 0, s);
    const Err e = err_of(got, want);
    const bool ok = e.rel < 1e-4;
    std::printf("  gpu grouped vs double reference: max abs %.3e rel %.3e %s\n", e.max_abs, e.rel, ok ? "ok" : "FAIL");
    failures += !ok;
    g.run(plan, x.data(), NT, strided, 1, s);
    if (std::memcmp(strided.data(), got.data(), got.size() * 4) != 0) { std::printf("  gpu one block row striding over the groups: DIFFER\n"); ++failures; }
    size_t at = 0;
    for (const auto& [b, toks] : plan)
        for (const int t : toks) {
            g.run({{b, {t}}}, x.data(), NT, one, 0, s);
            if (std::memcmp(one.data(), got.data() + at, (size_t) H * 4) != 0) {
                std::printf("  gpu entry (blob %d, token %d) alone vs in the call: DIFFER\n", b, t);
                ++failures;
            }
            at += H;
        }
    // the prompt path's dequantizer: the original-basis weights, rounded once to fp16
    {
        const uint8_t* v = blob[0].data() + f.bytes - f.tail;
        k::exl3_dequant_gu_f16(g.L, g.blobs, g.dq_gu, s);
        k::exl3_dequant_down_f16(g.L, g.blobs, g.dq_d, s);
        std::vector<uint16_t> gu((size_t) 2 * FF * H), dn((size_t) FF * H);
        CK(cudaMemcpyAsync(gu.data(), g.dq_gu, gu.size() * 2, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(dn.data(), g.dq_d, dn.size() * 2, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        const std::vector<double> wg = weight_d(ref[0].g, v, v + 2 * H, H, FF);
        const std::vector<double> wu = weight_d(ref[0].u, v + 2 * (H + FF), v + 2 * (2 * H + FF), H, FF);
        const std::vector<double> wd = weight_d(ref[0].d, v + 4 * (H + FF), v + 4 * (H + FF) + 2 * FF, FF, H);
        // within an fp16 rounding of the double value (the float transforms add a little)
        size_t bad = 0;
        double worst = 0;
        auto cmp = [&](uint16_t bits, double w) {
            const double d = std::fabs((double) k::f32_from_f16(bits) - w), ulp = std::ldexp(1.0, std::max(-14, std::ilogb(std::fabs(w) + 1e-30)) - 10);
            worst = std::max(worst, d / ulp);
            bad += d > 0.52 * ulp + 1e-7;
        };
        for (int r = 0; r < FF; ++r)
            for (int i = 0; i < H; ++i) {
                cmp(gu[(size_t) (2 * r) * H + i], wg[(size_t) r * H + i]);
                cmp(gu[(size_t) (2 * r + 1) * H + i], wu[(size_t) r * H + i]);
            }
        for (int r = 0; r < H; ++r)
            for (int i = 0; i < FF; ++i) cmp(dn[(size_t) r * FF + i], wd[(size_t) r * FF + i]);
        std::printf("  gpu fp16 dequantizer vs the weights in double: worst %.3f ulp, %zu of %d beyond an fp16 rounding %s\n", worst,
                    bad, 3 * H * FF, bad ? "FAIL" : "ok");
        failures += bad != 0;
    }
    return failures;
}

// the grouped decode kernels as the verify window runs them: `groups` experts of `per` entries each, distinct blobs
void gpu_bench(int k2, int groups, int per, cudaStream_t s) {
    cpu::NativeFmt f;
    std::string err;
    cpu::native_fmt(cpu::kExl3TypeBase + k2, cpu::kExl3TypeBase + k2, H, FF, f, err);
    const int ne = groups * per;
    Gpu g(f, groups, ne, per);
    std::mt19937 rng(3);
    std::vector<uint8_t> blob(f.bytes);
    std::vector<std::pair<int, std::vector<int>>> plan;
    for (int e = 0; e < groups; ++e) {
        random_blob(f, blob.data(), rng);
        CK(cudaMemcpy(g.blobs + f.bytes * e, blob.data(), f.bytes, cudaMemcpyHostToDevice));
        plan.push_back({e, {}});
        for (int i = 0; i < per; ++i) plan.back().second.push_back(i);
    }
    std::vector<float> x, res;
    inputs(x, per, rng);
    g.run(plan, x.data(), per, res, 0, s);
    auto run = [&] {
        k::native_expert_act_rows(f, g.x, per, g.act, s);
        k::native_expert_grouped(g.L, g.ptr, g.start, g.count, g.dst, g.tok, groups, ne, g.act, g.scr, g.out, s);
    };
    for (int i = 0; i < 20; ++i) run();
    CK(cudaStreamSynchronize(s));
    const int it = 200;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < it; ++i) run();
    CK(cudaStreamSynchronize(s));
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
    std::printf("bench %.1f bpw: %d groups x %d entries: %.1f us per call, %.0f GB/s of weights\n", k2 / 2.0, groups, per, us,
                (double) f.bytes * groups / (us * 1e3));
    if (groups == 1 && per == 1) {
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < it; ++i) {
            k::exl3_dequant_gu_f16(g.L, g.blobs, g.dq_gu, s);
            k::exl3_dequant_down_f16(g.L, g.blobs, g.dq_d, s);
        }
        CK(cudaStreamSynchronize(s));
        us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
        std::printf("bench %.1f bpw: fp16 dequantizer %.1f us per expert, %.1f GB/s of fp16 written\n", k2 / 2.0, us,
                    3.0 * H * FF * 2 / (us * 1e3));
    }
}

int decoder_checks() {
    int failures = 0;
    static const struct { int k2; uint32_t sum; } want[] = {
        {2, 0x43dfa62bu}, {3, 0x4034d888u}, {4, 0x4292e875u}, {5, 0x4321551bu}, {6, 0x44099f02u}, {7, 0x43b817d5u},
        {8, 0x429e5cc4u}, {10, 0x429c8297u}, {12, 0x441e955fu}, {14, 0x4707120eu}, {16, 0x43903d80u}};
    for (const auto& w : want) {
        uint8_t tile[256];
        for (int i = 0; i < 16 * w.k2; ++i) tile[i] = (uint8_t) (((i * 37 + 11) & 0xff) ^ (((i * i) >> 3) & 0xff));
        uint16_t st[256];
        cpu::exl3_tile_states(tile, w.k2, st);
        uint32_t sum = 0;
        for (int t = 0; t < 256; ++t) sum += (uint32_t) (t + 1) * st[t];
        if (sum != w.sum) { std::printf("  states at %.1f bpw: checksum %08x, expected %08x\n", w.k2 / 2.0, sum, w.sum); ++failures; }
    }
    uint32_t sum = 0;
    for (uint32_t s = 0; s < 65536; ++s) sum += (s + 1) * k::f16_from_f32(cpu::exl3_mul1_value(s));
    if (sum != 0x9f37aa70u) { std::printf("  mul1 codebook: checksum %08x, expected 9f37aa70\n", sum); ++failures; }
    uint32_t psum = 0;
    for (int t = 0; t < 256; ++t) psum += (uint32_t) (t + 1) * cpu::kExl3Perm[t];
    if (psum != 4338624u) { std::printf("  tile permutation: checksum %u\n", psum); ++failures; }
    std::printf("decoder vs tools/convert/exl3.py (11 bitrates, the codebook, the tile order): %s\n", failures ? "DIFFER" : "equal");
    return failures;
}

int parity(int k2) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(cpu::kExl3TypeBase + k2, cpu::kExl3TypeBase + k2, H, FF, f, err)) {
        std::printf("%.1f bpw: %s\n", k2 / 2.0, err.c_str());
        return 1;
    }
    int failures = 0;
    std::mt19937 rng(100 + (unsigned) k2);
    std::vector<uint8_t> blob(f.bytes);
    random_blob(f, blob.data(), rng);
    std::vector<float> x;
    inputs(x, NT, rng);
    Ref ref;
    ref.decode(f, blob.data());
    std::vector<double> want((size_t) NT * H);
    for (int t = 0; t < NT; ++t) ref.run(f, blob.data(), x.data() + (size_t) t * H, want.data() + (size_t) t * H);
    std::vector<float> sc, got, one;
    cpu_expert(f, cpu::Exl3Isa::scalar, blob.data(), x.data(), NT, 1, sc);
    const Err e = err_of(sc, want);
    const bool ok = e.rel < 2e-3;
    std::printf("%.1f bpw (blob %zu B): scalar vs double reference max abs %.3e rel %.3e %s\n", k2 / 2.0, f.bytes, e.max_abs,
                e.rel, ok ? "ok" : "FAIL");
    failures += !ok;
    auto same = [&](const char* what, const std::vector<float>& a, const std::vector<float>& b, size_t n) {
        const bool eq = std::memcmp(a.data(), b.data(), n * sizeof(float)) == 0;
        if (!eq) { std::printf("  %s: DIFFER\n", what); ++failures; }
    };
    for (const cpu::Exl3Isa isa : {cpu::Exl3Isa::scalar, cpu::Exl3Isa::avx2, cpu::Exl3Isa::avx512}) {
        if (!cpu::exl3_isa_ok(isa)) { std::printf("  %s: not on this CPU\n", cpu::exl3_isa_name(isa)); continue; }
        const std::string n = cpu::exl3_isa_name(isa);
        cpu_expert(f, isa, blob.data(), x.data(), NT, 1, got);
        same((n + " vs scalar").c_str(), got, sc, got.size());
        cpu_expert(f, isa, blob.data(), x.data(), NT, 4, got);
        same((n + " in row ranges vs whole").c_str(), got, sc, got.size());
        for (int t = 0; t < NT; ++t) {
            cpu_expert(f, isa, blob.data(), x.data() + (size_t) t * H, 1, 1, one);
            if (std::memcmp(one.data(), sc.data() + (size_t) t * H, (size_t) H * sizeof(float)) != 0) {
                std::printf("  %s token %d alone vs in the group: DIFFER\n", n.c_str(), t);
                ++failures;
            }
        }
    }
    // every group size up to the largest, on the scalar kernel's results per token
    {
        std::vector<float> xs, all, mine;
        inputs(xs, cpu::kExl3MaxTokens, rng);
        cpu_expert(f, cpu::Exl3Isa::scalar, blob.data(), xs.data(), cpu::kExl3MaxTokens, 1, all);
        for (const cpu::Exl3Isa isa : {cpu::Exl3Isa::avx2, cpu::Exl3Isa::avx512}) {
            if (!cpu::exl3_isa_ok(isa)) continue;
            for (int nt = 1; nt <= cpu::kExl3MaxTokens; ++nt) {
                cpu_expert(f, isa, blob.data(), xs.data() + (size_t) (cpu::kExl3MaxTokens - nt) * H, nt, 1, mine);
                if (std::memcmp(mine.data(), all.data() + (size_t) (cpu::kExl3MaxTokens - nt) * H, mine.size() * sizeof(float)) != 0) {
                    std::printf("  %s group of %d vs scalar: DIFFER\n", cpu::exl3_isa_name(isa), nt);
                    ++failures;
                }
            }
        }
    }
    // the pool's entry points take the same path
    {
        std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(cpu::kNativeActBytes)),
            hq(NT, std::vector<uint8_t>(cpu::kNativeHBytes));
        std::vector<std::vector<float>> ff(NT, std::vector<float>((size_t) FF));
        const void* a[NT];
        const void* hp[NT];
        float* fp[NT];
        float* op[NT];
        got.assign((size_t) NT * H, 0.f);
        for (int t = 0; t < NT; ++t) {
            cpu::native_quant_act(f, x.data() + (size_t) t * H, act[(size_t) t].data());
            a[t] = act[(size_t) t].data();
            fp[t] = ff[(size_t) t].data();
            op[t] = got.data() + (size_t) t * H;
        }
        cpu::native_gu_rows(f, blob.data(), a, NT, fp, 0, FF);
        for (int t = 0; t < NT; ++t) { cpu::native_quant_h(f, fp[t], hq[(size_t) t].data()); hp[t] = hq[(size_t) t].data(); }
        cpu::native_down_rows(f, blob.data(), hp, NT, op, 0, H);
        same("native_gu_rows / native_down_rows vs the kernels", got, sc, got.size());
    }
    return failures;
}

void cpu_bench(int k2, cpu::Exl3Isa isa, int nt) {
    constexpr int ne = 24;
    cpu::NativeFmt f;
    std::string err;
    cpu::native_fmt(cpu::kExl3TypeBase + k2, cpu::kExl3TypeBase + k2, H, FF, f, err);
    std::vector<uint8_t> w(f.bytes * ne);
    std::mt19937 rng(1);
    for (int e = 0; e < ne; ++e) random_blob(f, w.data() + f.bytes * e, rng);
    std::vector<float> x, out;
    inputs(x, nt, rng);
    const int it = isa == cpu::Exl3Isa::scalar ? 1 : 4;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < it; ++i)
        for (int e = 0; e < ne; ++e) cpu_expert(f, isa, w.data() + f.bytes * e, x.data(), nt, 1, out);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / (it * ne);
    std::printf("cpu bench %.1f bpw %-6s %d token%s: %.1f us per expert, %.2f GB/s of weights, %.2f G weights/s per token\n",
                k2 / 2.0, cpu::exl3_isa_name(isa), nt, nt > 1 ? "s" : " ", us, (double) f.bytes / (us * 1e3),
                3.0 * H * FF * nt / (us * 1e3));
}

int io(const char* blob_path, int gu, int d, const char* out_path) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu, d, H, FF, f, err)) { std::printf("%s\n", err.c_str()); return 1; }
    std::vector<uint8_t> blob(f.bytes);
    FILE* in = std::fopen(blob_path, "rb");
    if (!in || std::fread(blob.data(), 1, f.bytes, in) != f.bytes) { std::printf("%s: not a blob of %zu bytes\n", blob_path, f.bytes); return 1; }
    std::fclose(in);
    std::mt19937 rng(9);
    std::vector<float> x, y;
    inputs(x, NT, rng);
    cpu_expert(f, cpu::exl3_isa(), blob.data(), x.data(), NT, 1, y);
    FILE* o = std::fopen(out_path, "wb");
    if (!o) return 1;
    std::fwrite(x.data(), 4, x.size(), o);
    std::fwrite(y.data(), 4, y.size(), o);
    std::fclose(o);
    std::printf("%s: %d tokens of %d inputs and outputs written to %s\n", cpu::exl3_isa_name(cpu::exl3_isa()), NT, H, out_path);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 6 && std::string(argv[1]) == "--io") return io(argv[2], std::atoi(argv[3]), std::atoi(argv[4]), argv[5]);
    if (argc > 1 && std::string(argv[1]) == "--cpu-bench") {
        for (const int k2 : {3, 4, 8})
            for (const cpu::Exl3Isa isa : {cpu::Exl3Isa::scalar, cpu::Exl3Isa::avx2, cpu::Exl3Isa::avx512})
                if (cpu::exl3_isa_ok(isa))
                    for (const int nt : {1, 4}) cpu_bench(k2, isa, nt);
        return 0;
    }
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        for (const int k2 : {3, 4, 8})
            for (const auto& [groups, per] : {std::pair<int, int>{1, 1}, {10, 1}, {30, 1}, {10, 4}, {40, 2}}) gpu_bench(k2, groups, per, s);
        return 0;
    }
    int failures = decoder_checks();
    for (const int k2 : kRates) {
        failures += parity(k2);
        failures += gpu_parity(k2, s);
    }
    std::printf("%s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
