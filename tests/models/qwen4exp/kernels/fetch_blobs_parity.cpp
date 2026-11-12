// tests/models/qwen4exp/kernels/fetch_blobs_parity.cpp - fetch_blobs / rebase_ptrs (verify_kernels.hpp): blobs of
// mapped host memory copied into a device staging area byte for byte, the pointers rebased onto it, for 0 to 6 blobs
// of expert sizes.
//
//     build/fetch_blobs_parity           the checks
//     build/fetch_blobs_parity --bench   GB/s over the link for 1 to 4 blobs, none of them in the L2
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = basalt::kernels;

#define CK(x) do { if ((x) != cudaSuccess) { std::printf("%s: %s\n", #x, cudaGetErrorString(cudaGetLastError())); std::exit(1); } } while (0)

namespace {

constexpr int kCap = 8;

struct Setup {
    int64_t bytes;
    uint8_t* host = nullptr;   // kCap blobs, mapped
    uint8_t* stage = nullptr;
    unsigned long long* ptr = nullptr;
    int32_t* n = nullptr;
    explicit Setup(int64_t b) : bytes(b) {
        CK(cudaHostAlloc((void**) &host, (size_t) (kCap * b), cudaHostAllocMapped));
        CK(cudaMalloc(&stage, (size_t) (kCap * b)));
        CK(cudaMalloc(&ptr, kCap * 8));
        CK(cudaMalloc(&n, 4));
    }
    ~Setup() {
        cudaFreeHost(host);
        cudaFree(stage);
        cudaFree(ptr);
        cudaFree(n);
    }
    // blobs `order` (indices into the host blobs) staged in that order
    void run(const std::vector<int>& order, cudaStream_t s) {
        std::vector<unsigned long long> hp(kCap, 0);
        for (size_t i = 0; i < order.size(); ++i) {
            void* d = nullptr;
            CK(cudaHostGetDevicePointer(&d, host + (size_t) order[i] * bytes, 0));
            hp[i] = (unsigned long long) d;
        }
        const int32_t cnt = (int32_t) order.size();
        CK(cudaMemcpy(ptr, hp.data(), kCap * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(n, &cnt, 4, cudaMemcpyHostToDevice));
        k::fetch_blobs(ptr, n, stage, bytes, kCap, s);
        k::rebase_ptrs(ptr, n, stage, bytes, s);
        CK(cudaStreamSynchronize(s));
    }
};

int check(int64_t bytes, cudaStream_t s) {
    Setup u(bytes);
    std::mt19937_64 rng((uint64_t) bytes);
    for (int64_t i = 0; i < kCap * bytes; i += 8) {
        const uint64_t v = rng();
        std::memcpy(u.host + i, &v, 8);
    }
    int failures = 0;
    for (int cnt = 0; cnt <= 6; ++cnt) {
        std::vector<int> order;
        for (int i = 0; i < cnt; ++i) order.push_back((5 * i + 3) % kCap);
        CK(cudaMemset(u.stage, 0xa5, (size_t) (kCap * bytes)));
        u.run(order, s);
        std::vector<uint8_t> got((size_t) (kCap * bytes));
        std::vector<unsigned long long> p(kCap);
        CK(cudaMemcpy(got.data(), u.stage, got.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(p.data(), u.ptr, kCap * 8, cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < cnt; ++i) {
            ok &= std::memcmp(got.data() + (size_t) i * bytes, u.host + (size_t) order[i] * bytes, (size_t) bytes) == 0;
            ok &= p[i] == (unsigned long long) (u.stage + (size_t) i * bytes);
        }
        for (int64_t i = (int64_t) cnt * bytes; i < kCap * bytes; ++i) ok &= got[(size_t) i] == 0xa5;   // nothing past the blobs
        if (!ok) {
            std::printf("  %lld-byte blobs, %d staged: FAIL\n", (long long) bytes, cnt);
            ++failures;
        }
    }
    std::printf("%lld-byte blobs, 0 to 6 staged: %s\n", (long long) bytes, failures ? "FAIL" : "ok");
    return failures;
}

// every copy reads blobs the GPU's L2 has not seen: a pool of host memory larger than the L2, in turn
void bench(int64_t bytes, cudaStream_t s) {
    const int pool = (int) ((128ll << 20) / bytes) + kCap;
    uint8_t* host = nullptr;
    CK(cudaHostAlloc((void**) &host, (size_t) (pool * bytes), cudaHostAllocMapped));
    std::memset(host, 1, (size_t) (pool * bytes));
    Setup u(bytes);
    std::vector<unsigned long long> dev((size_t) pool);
    for (int i = 0; i < pool; ++i) {
        void* d = nullptr;
        CK(cudaHostGetDevicePointer(&d, host + (size_t) i * bytes, 0));
        dev[(size_t) i] = (unsigned long long) d;
    }
    for (int cnt = 1; cnt <= 4; ++cnt) {
        const int it = pool / cnt;
        std::vector<unsigned long long> ptrs((size_t) (it * kCap), 0);
        for (int r = 0; r < it; ++r)
            for (int i = 0; i < cnt; ++i) ptrs[(size_t) (r * kCap + i)] = dev[(size_t) (r * cnt + i)];
        unsigned long long* dp = nullptr;
        CK(cudaMalloc(&dp, ptrs.size() * 8));
        CK(cudaMemcpy(dp, ptrs.data(), ptrs.size() * 8, cudaMemcpyHostToDevice));
        const int32_t c = cnt;
        CK(cudaMemcpy(u.n, &c, 4, cudaMemcpyHostToDevice));
        CK(cudaStreamSynchronize(s));
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < it; ++r) k::fetch_blobs(dp + (size_t) r * kCap, u.n, u.stage, bytes, kCap, s);
        CK(cudaStreamSynchronize(s));
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
        std::printf("bench %lld-byte blobs x %d: %.1f us per copy, %.1f GB/s\n", (long long) bytes, cnt, us,
                    (double) bytes * cnt / (us * 1e3));
        cudaFree(dp);
    }
    cudaFreeHost(host);
}

}  // namespace

int main(int argc, char** argv) {
    CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    // two expert-sized blobs and a small odd count of 16-byte words
    const int64_t sizes[3] = {940800, 1248000, 16 * 1001};
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        for (const int64_t b : sizes) bench(b, s);
        return 0;
    }
    int failures = 0;
    for (const int64_t b : sizes) failures += check(b, s);
    std::printf("%s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
