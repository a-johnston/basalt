// tests/kernels/exl3_expert_parity.cpp - the EXL3 experts (exl3_cpu.hpp) on the CPU.
//
//     build/exl3_expert_parity                 random blobs at the model's geometry, every bitrate
//     build/exl3_expert_parity --cpu-bench     the CPU rows per ISA on one core: us per expert and GB/s
//     build/exl3_expert_parity --io BLOB GU D OUT   one real expert blob (types GU / D) on fixed inputs: writes
//                                              x (NT x H floats) then y (NT x H floats) to OUT, for a comparison
//                                              with ExLlamaV3 (tools/convert/exl3_upstream_check.py --expert-io)
//
// (a) the decoder against checksums computed by tools/convert/exl3.py, which is checked against ExLlamaV3.
// (b) the reference: exl3_dequant_inner (the fp16 codebook values), the transforms and a SwiGLU expert in double.
// (c) the CPU path (native_quant_act / native_gu_rows / native_quant_h / native_down_rows): within a tolerance of
// (b) (int16 activations, the codebook before its fp16 rounding), every ISA bit for bit, a token alone against the
// same token in a group bit for bit, row ranges against the whole bit for bit.
#include "basalt/kernels/cpu/exl3_cpu.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/f16_bits.hpp"

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
    int failures = decoder_checks();
    for (const int k2 : kRates) failures += parity(k2);
    std::printf("%s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
