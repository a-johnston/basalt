// src/models/inkling2x/session.cpp - an Inkling 2 session: the K/V caches and convolution rings of its layer slots, its last
// window's logits and its draws. The forward of a run of rows (a prompt chunk, or a decode batch: the windows of several
// sessions as one run, the weights read once) goes through the stored layers before the loop, the loop's core once per
// pass (each pass after the first mixing in the one before through the controllers), the layers after it, and the output.
// Every expert of every layer is on the card, so a row's whole forward stays there.
//
// The sessions of one model run one after another on the serving thread, each call leaving its stream idle, so they share
// one Scratch: the forward's activations and workspaces, sized for a prompt chunk, and the recorded decode graphs. A
// decode batch replays a graph recorded for its members and the attention's split counts, which change every 512
// positions (kernels.hpp). A row's result does not depend on the rows beside it, so a session's tokens are the same alone
// or in a batch.
#include "basalt/core/graph.hpp"
#include "basalt/kernels/cpu/exl3_cpu.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/models/inkling2x/kernels.hpp"
#include "basalt/log.hpp"
#include "models/inkling2x/model.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::inkling2x {
namespace {

namespace bk = basalt::kernels;
namespace ik = inkling2x::kernels;

constexpr int kMaxWindow = ik::kMaxMembers;   // rows of a decode window, and of a batch of them
constexpr int kChunk = 128;                    // rows of a prompt chunk: the most a forward takes
constexpr int kRing = 32;                      // positions a convolution ring keeps: a window's rows and the three taps before them
constexpr int kHead = 128;
constexpr float kMixEps = 1e-6f;
constexpr size_t kGraphs = 8;                  // decode graphs kept (members and split counts)

// Device memory that goes back when its owner does; zeroed when taken.
struct Buffers {
    std::vector<void*> p;
    ~Buffers() {
        for (void* b : p) cudaFree(b);
    }
    template<class T>
    bool take(T*& out, size_t n, std::string& err, const char* what) {
        void* b = nullptr;
        if (cudaMalloc(&b, n * sizeof(T)) != cudaSuccess) {
            cudaGetLastError();
            err = std::string("out of device memory for ") + what + " (" + std::to_string((n * sizeof(T)) >> 20) + " MiB more)";
            return false;
        }
        cudaMemset(b, 0, n * sizeof(T));
        p.push_back(b);
        out = (T*) b;
        return true;
    }
};

struct Scratch {
    Buffers mem;
    int32_t* tok = nullptr;
    int64_t* pos = nullptr;   // each row's position, which the kernels read
    float *emb = nullptr, *cur = nullptr, *pre = nullptr, *out[2] = {nullptr, nullptr}, *mix = nullptr, *msg = nullptr, *ctl = nullptr,
          *nrm = nullptr;
    uint8_t *xq = nullptr, *attq = nullptr, *fxq = nullptr, *hq = nullptr, *escr = nullptr, *ws = nullptr;
    float *q = nullptr, *kraw = nullptr, *vraw = nullptr, *k = nullptr, *v = nullptr, *r = nullptr, *sink = nullptr, *gate = nullptr,
          *dk = nullptr, *dv = nullptr, *att = nullptr, *o = nullptr, *oc = nullptr, *fx = nullptr, *fc = nullptr, *fn = nullptr,
          *rlog = nullptr, *rw = nullptr, *eout = nullptr, *mc = nullptr, *ctlg = nullptr, *hn = nullptr, *hl = nullptr, *lp = nullptr;
    int32_t *ids = nullptr, *grp_start = nullptr, *n_groups = nullptr, *ent_dst = nullptr, *ent_tok = nullptr;
    unsigned long long* grp_ptr = nullptr;
    bk::Exl3Rates* grp_rates = nullptr;
    std::vector<float*> blk;   // per layer slot where a block starts: the stream before it
    struct Graph {
        std::vector<uintptr_t> key;
        core::CapturedGraph g;
        uint64_t used = 0;
    };
    std::vector<Graph> graphs;
    uint64_t tick = 0;

    bool init(Inkling2Model& m, std::string& err) {
        const Geometry& g = m.geo();
        const int E = g.n_embd, R = kChunk, K = g.n_used;
        const int nq = g.n_head * kHead, nkv = g.n_kv * kHead, nrel = g.n_head * g.d_rel;
        const size_t q8 = (size_t) E / 32 * 36, q8a = (size_t) nq / 32 * 36, RE = (size_t) R * E;
        const char* w = "the forward's scratch";
        size_t escr_bytes = 0;
        for (int l = 0; l < g.n_layers; ++l) escr_bytes = std::max(escr_bytes, bk::native_expert_scratch_bytes(m.expert_layout(l), (int64_t) R * K));
        bool ok = mem.take(tok, R, err, w) && mem.take(pos, R, err, w) && mem.take(emb, RE, err, w) && mem.take(cur, RE, err, w) &&
                  mem.take(pre, RE, err, w) && mem.take(out[0], RE, err, w) && mem.take(out[1], RE, err, w) && mem.take(mix, 3 * RE, err, w) &&
                  mem.take(msg, 4 * RE, err, w) && mem.take(ctl, 3 * RE, err, w) && mem.take(nrm, 3 * RE, err, w) &&
                  mem.take(xq, 3 * R * q8, err, w) && mem.take(q, (size_t) R * nq, err, w) && mem.take(kraw, (size_t) R * nkv, err, w) &&
                  mem.take(vraw, (size_t) R * nkv, err, w) && mem.take(k, (size_t) R * nkv, err, w) && mem.take(v, (size_t) R * nkv, err, w) &&
                  mem.take(r, (size_t) R * nrel, err, w) && mem.take(sink, (size_t) R * g.n_head, err, w) && mem.take(gate, (size_t) R * nq, err, w) &&
                  mem.take(dk, (size_t) R * g.n_kv * 4, err, w) && mem.take(dv, (size_t) R * g.n_kv * 4, err, w) &&
                  mem.take(att, (size_t) R * nq, err, w) && mem.take(attq, (size_t) R * q8a, err, w) && mem.take(o, RE, err, w) &&
                  mem.take(oc, RE, err, w) && mem.take(fx, RE, err, w) && mem.take(fc, RE, err, w) && mem.take(fn, RE, err, w) &&
                  mem.take(fxq, R * q8, err, w) && mem.take(rlog, (size_t) R * g.n_expert, err, w) && mem.take(ids, (size_t) R * K, err, w) &&
                  mem.take(rw, (size_t) R * K, err, w) && mem.take(grp_ptr, (size_t) g.n_expert, err, w) &&
                  mem.take(grp_rates, (size_t) g.n_expert, err, w) && mem.take(grp_start, (size_t) g.n_expert + 1, err, w) &&
                  mem.take(n_groups, 1, err, w) && mem.take(ent_dst, (size_t) R * K, err, w) && mem.take(ent_tok, (size_t) R * K, err, w) &&
                  mem.take(eout, (size_t) R * K * E, err, w) && mem.take(mc, RE, err, w) && mem.take(ctlg, 3 * (size_t) R * g.gate_rank, err, w) &&
                  mem.take(hn, (size_t) kMaxWindow * E, err, w) && mem.take(hq, kMaxWindow * q8, err, w) &&
                  mem.take(hl, (size_t) kMaxWindow * g.vocab_padded, err, w) &&
                  mem.take(lp, bk::row_logprobs_scratch_floats(kMaxWindow, g.n_vocab) + (size_t) kMaxWindow * bk::kLogprobStride, err, w) &&
                  mem.take(escr, escr_bytes, err, w) && mem.take(ws, ik::attention_workspace_bytes(R, g.n_head, m.context()), err, w);
        if (!ok) return false;
        blk.assign((size_t) g.slots(), nullptr);
        for (int s = 0; s < g.slots(); ++s)
            if (g.layer_of(s) % g.block == 0 && !mem.take(blk[(size_t) s], RE, err, w)) return false;
        return cudaDeviceSynchronize() == cudaSuccess || (err = "the forward's scratch failed", false);
    }
};

class Session;

// A run of rows: member i is session s[i] with rows [m.start[i], m.start[i + 1]) from position first[i]; head[i] of its
// last rows get logits (none in a prompt chunk).
struct Run {
    ik::Members m;
    Session* s[kMaxWindow] = {};
    int64_t first[kMaxWindow] = {};
    int head[kMaxWindow] = {};
    int rows() const { return m.rows(); }
    int64_t last() const {   // the largest position of a row
        int64_t p = 0;
        for (int i = 0; i < m.n; ++i) p = std::max<int64_t>(p, first[i] + (m.start[i + 1] - m.start[i]) - 1);
        return p;
    }
};

class Session final : public model::ModelSession {
public:
    Session(Inkling2Model& m, std::shared_ptr<Scratch> x) : m_(m), g_(m.geo()), x_(std::move(x)) {}
    ~Session() override {
        if (stream_) cudaStreamDestroy(stream_);
    }

    bool init(std::string& err) {
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) { err = "no CUDA stream"; return false; }
        const int nkv = g_.n_kv * kHead;
        width_ = 2 * nkv + 2 * g_.n_embd;   // K, V, the attention's output, the FFN's output
        ctx_ = m_.context();
        const char* w = "the session";
        if (!mem_.take(logits_, (size_t) kMaxWindow * g_.n_vocab, err, w) || !mem_.take(pick_, (size_t) 3 * kMaxWindow, err, w) ||
            !mem_.take(d_bias_, (size_t) g_.n_vocab, err, w) || !mem_.take(d_mask_, (size_t) g_.n_vocab, err, w))
            return false;
        // the layer slots: a K/V cache and a convolution ring each
        const int S = g_.slots();
        kc_.assign((size_t) S, nullptr);
        vc_.assign((size_t) S, nullptr);
        cells_.assign((size_t) S, 0);
        ring_.assign((size_t) S, nullptr);
        for (int s = 0; s < S; ++s) {
            cells_[(size_t) s] = g_.swa[(size_t) g_.layer_of(s)] ? g_.window + kChunk : ctx_;
            const size_t cb = (size_t) cells_[(size_t) s] * nkv;
            if (!mem_.take(kc_[(size_t) s], cb, err, w) || !mem_.take(vc_[(size_t) s], cb, err, w) ||
                !mem_.take(ring_[(size_t) s], (size_t) kRing * width_, err, w))
                return false;
        }
        return cudaDeviceSynchronize() == cudaSuccess || (err = "the session's buffers failed", false);
    }

    model::Result decode_window(const model::DecodeRequest& r) override {
        model::ModelSession* self = this;
        return decode_batch(&self, &r, 1);
    }

    int batch_rows() const override { return kMaxWindow; }

    model::Result decode_batch(model::ModelSession* const* sessions, const model::DecodeRequest* reqs, int n) override {
        if (n < 1 || n > kMaxWindow) return {model::Status::Error, "inkling2x: a batch of " + std::to_string(n) + " windows"};
        Run run;
        run.m.n = n;
        std::vector<int32_t> tokens;
        for (int i = 0; i < n; ++i) {
            const model::DecodeRequest& r = reqs[i];
            if (r.n_tokens < 1 || run.m.start[i] + r.n_tokens > kMaxWindow)
                return {model::Status::Error, "inkling2x: a batch of more than " + std::to_string(kMaxWindow) + " rows"};
            if (r.pos + r.n_tokens > ctx_) return {model::Status::Refused, "inkling2x: the window passes the context of " + std::to_string(ctx_)};
            run.s[i] = static_cast<Session*>(sessions[i]);
            run.first[i] = r.pos;
            run.head[i] = r.row_tokens ? r.n_tokens : 1;
            run.m.start[i + 1] = run.m.start[i] + r.n_tokens;
            tokens.insert(tokens.end(), r.tokens, r.tokens + r.n_tokens);
        }
        std::string err;
        if (!run_rows(run, tokens.data(), true, err)) return {model::Status::Error, "inkling2x: " + err};
        // each member's draw from its own logits, then one wait
        for (int i = 0; i < n; ++i) run.s[i]->draw(reqs[i], run.head[i], stream_);
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "inkling2x: the window failed"};
        for (int i = 0; i < n; ++i) {
            Session& s = *run.s[i];
            const model::DecodeRequest& r = reqs[i];
            if (r.row_tokens)
                for (int t = 0; t < r.n_tokens; ++t) r.row_tokens[t] = s.picks_[(size_t) t];
            if (r.next_token) *r.next_token = s.picks_[(size_t) run.head[i] - 1];
            if (r.logits) cudaMemcpy(r.logits, s.logits_ + (size_t) (run.head[i] - 1) * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost);
            s.last_rows_ = run.head[i];
            s.pos_ = r.pos + r.n_tokens;
        }
        return {};
    }

    model::Result prefill_chunk(const model::PrefillRequest& r) override {
        if (r.pos + r.n_tokens > ctx_) return {model::Status::Refused, "inkling2x: the prompt passes the context of " + std::to_string(ctx_)};
        std::string err;
        for (int t0 = 0; t0 < r.n_tokens; t0 += kChunk) {
            if (r.stop && r.stop()) break;
            Run run;
            run.m.n = 1;
            run.m.start[1] = std::min(kChunk, r.n_tokens - t0);
            run.s[0] = this;
            run.first[0] = r.pos + t0;
            if (!run_rows(run, r.tokens + t0, false, err)) return {model::Status::Error, "inkling2x: " + err};
            pos_ = r.pos + t0 + run.rows();
            if (r.progress) {
                cudaStreamSynchronize(stream_);
                r.progress(t0 + run.rows());
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "inkling2x: the prompt failed"};
        return {};
    }

    model::Result set_logit_bias(const float* bias) override {
        bias_on_ = bias != nullptr;
        if (bias_on_ && cudaMemcpyAsync(d_bias_, bias, (size_t) g_.n_vocab * 4, cudaMemcpyHostToDevice, stream_) != cudaSuccess)
            return {model::Status::Error, "inkling2x: the logit bias upload failed"};
        return cudaStreamSynchronize(stream_) == cudaSuccess ? model::Result{} : model::Result{model::Status::Error, "inkling2x: the logit bias upload failed"};
    }

    // a draw of one row of the last window again, with the tokens the mask bans
    model::Result resample_row(int row, const float* bias, int32_t& token) override {
        if (row < 0 || row >= last_rows_) return {model::Status::Error, "inkling2x: the row is not in the last window"};
        if (cudaMemcpyAsync(d_mask_, bias, (size_t) g_.n_vocab * 4, cudaMemcpyHostToDevice, stream_) != cudaSuccess)
            return {model::Status::Error, "inkling2x: the mask upload failed"};
        bk::SamplerParams sp = sampling_;
        sp.logit_bias = d_mask_;
        sp.counter += (uint64_t) row;
        int32_t* one = pick_ + 2 * kMaxWindow;
        bk::sample_tokens(logits_ + (size_t) row * g_.n_vocab, 1, g_.n_vocab, nullptr, 0, sp, one, stream_);
        if (cudaMemcpyAsync(&token, one, 4, cudaMemcpyDeviceToHost, stream_) != cudaSuccess || cudaStreamSynchronize(stream_) != cudaSuccess)
            return {model::Status::Error, "inkling2x: the masked draw failed"};
        return {};
    }

    model::Result row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) override {
        if (row0 < 0 || n < 1 || row0 + n > last_rows_) return {model::Status::Error, "inkling2x: the session keeps the last window's logits only"};
        float* d_out = x_->lp + bk::row_logprobs_scratch_floats(n, g_.n_vocab);
        int32_t* d_pick = pick_ + kMaxWindow;
        cudaMemcpyAsync(d_pick, picks, (size_t) n * 4, cudaMemcpyHostToDevice, stream_);
        bk::row_logprobs(logits_ + (size_t) row0 * g_.n_vocab, n, g_.n_vocab, d_pick, top, d_out, x_->lp, stream_);
        cudaMemcpyAsync(out, d_out, (size_t) n * bk::kLogprobStride * 4, cudaMemcpyDeviceToHost, stream_);
        return cudaStreamSynchronize(stream_) == cudaSuccess ? model::Result{} : model::Result{model::Status::Error, "inkling2x: logprobs failed"};
    }

    // ---- checkpoints: the full layers' caches are by position; the sliding layers' K/V rings and every slot's convolution
    // ring are the running state. A ring keeps kRing (convolutions) or window + kChunk (K/V) positions, so a checkpoint at
    // `pos` is intact while the session has not run more than kRing - 3 rows past it.
    size_t running_bytes() const override {
        size_t n = 0;
        for (int s = 0; s < g_.slots(); ++s) n += (size_t) kRing * width_ * 4 + (g_.swa[(size_t) g_.layer_of(s)] ? 2 * kv_bytes_of(s) : 0);
        return n;
    }

    model::Result running_state(int64_t pos, bool save, core::RunningState& st) override {
        if (save && pos > pos_) return {model::Status::Error, "inkling2x: a checkpoint past the session's position"};
        if (save && pos_ - pos > kRing - 3)
            return {model::Status::Error, "inkling2x: the rings have moved on more than " + std::to_string(kRing - 3) + " positions past the checkpoint"};
        for (int s = 0; s < g_.slots(); ++s) {
            st.regions.push_back({ring_[(size_t) s], (size_t) kRing * width_ * 4});
            if (g_.swa[(size_t) g_.layer_of(s)]) {
                st.regions.push_back({kc_[(size_t) s], kv_bytes_of(s)});
                st.regions.push_back({vc_[(size_t) s], kv_bytes_of(s)});
            }
        }
        st.stream = stream_;
        return {};
    }

    model::Result running_restored(int64_t pos, const uint8_t*) override {
        pos_ = pos;
        return {};
    }

    model::Result copy_logits(int row, float* out) override {
        if (row < 0 || row >= last_rows_) return {model::Status::Error, "inkling2x: no logits for row " + std::to_string(row)};
        cudaMemcpy(out, logits_ + (size_t) row * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost);
        return {};
    }

private:
    // diagnostic (--diag-dump): a tensor of this call appended to <dir>/<name>.f32
    bool diag() const { return !m_.diag_dump().empty(); }
    void dump(const std::string& name, const float* d, int64_t n) {
        std::vector<float> h((size_t) n);
        cudaMemcpyAsync(h.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        const std::string path = m_.diag_dump() + "/" + name + ".f32";
        std::FILE* f = std::fopen(path.c_str(), "ab");
        if (!f) {
            basalt::log::warn("basalt inkling2x", "--diag-dump: cannot write %s", path.c_str());
            return;
        }
        std::fwrite(h.data(), 4, h.size(), f);
        std::fclose(f);
    }

    size_t kv_bytes_of(int s) const { return (size_t) cells_[(size_t) s] * g_.n_kv * kHead * 2; }

    const DevTensor* T(int l, const char* s) const { return m_.tensor(l, s); }
    const float* F(int l, const char* s) const { return (const float*) m_.tensor(l, s)->ptr; }

    // y = W x for the q8_1 rows xq (R of n_in)
    void gemv(const DevTensor* w, const void* xq, float* y, int n_in, int n_out, int R) {
        bk::iq_mmvq(w->type, w->ptr, xq, y, n_in, n_out, R, stream_);
    }

    // projections of the q8_1 inputs xq[in]: those of one input and one type in one launch (each bitwise its own gemv)
    struct Proj {
        const DevTensor* t;
        float* y;
        int n_out, in;
    };
    void gemv_sets(const Proj* p, int n, const void* const* xq, int n_in, int R) {
        bool done[bk::kMmvqSetMax] = {};
        for (int i = 0; i < n; ++i) {
            if (done[i]) continue;
            bk::MmvqSet set;
            for (int j = i; j < n; ++j)
                if (!done[j] && xq[p[j].in] == xq[p[i].in] && p[j].t->type == p[i].t->type) {
                    set.add(p[j].t->ptr, p[j].y, p[j].n_out);
                    done[j] = true;
                }
            bk::iq_mmvq_set(p[i].t->type, set, xq[p[i].in], n_in, R, stream_);
        }
    }
    size_t q8(int n) const { return (size_t) n / 32 * 36; }

    ik::States states(const std::vector<const float*>& blocks, const float* cur) const {
        ik::States s;
        for (const float* b : blocks) s.p[s.n++] = b;
        s.p[s.n++] = cur;
        return s;
    }

    // the members' state of layer slot s
    ik::ConvRing ring_of(const Run& run, int s) const {
        ik::ConvRing ring;
        for (int i = 0; i < run.m.n; ++i) ring.data[i] = run.s[i]->ring_[(size_t) s];
        ring.slots = kRing;
        ring.width = width_;
        return ring;
    }
    ik::KvCaches caches_of(const Run& run, int s) const {
        ik::KvCaches c;
        for (int i = 0; i < run.m.n; ++i) {
            c.k[i] = run.s[i]->kc_[(size_t) s];
            c.v[i] = run.s[i]->vc_[(size_t) s];
        }
        return c;
    }

    // one layer slot over the run's rows: slot s of stored layer l; `bank` (the previous pass's states) is set on the loop's
    // later passes
    void layer(const Run& run, int s, int l, std::vector<const float*>& blocks, const std::vector<const float*>* bank) {
        Scratch& x = *x_;
        const int R = run.rows(), E = g_.n_embd, nq = g_.n_head * kHead, nkv = g_.n_kv * kHead, nrel = g_.n_head * g_.d_rel;
        const bool swa = g_.swa[(size_t) l];
        const size_t RE = (size_t) R * E;
        const float* in[3] = {x.cur, x.cur, x.cur};
        if (l > 0) {   // the depth mix: one input on a sliding layer, one each for Q, K and V on a full one
            ik::depth_mix(states(blocks, x.cur), F(l, "attn_res_query.weight"), swa ? 1 : 3, R, E, kMixEps, x.mix, stream_);
            for (int j = 0; j < 3; ++j) in[j] = x.mix + (swa ? 0 : j) * RE;
        }
        if (bank) {   // the messages of the previous pass, mixed into Q, K and V
            ik::States bs;
            for (const float* p : *bank) bs.p[bs.n++] = p;
            ik::depth_mix(bs, F(l, "loop_msg_query.weight"), 4, R, E, kMixEps, x.msg, stream_);
            static const char* ctl[3][3] = {{"loop_q_ctrl_down.weight", "loop_q_ctrl_gain.weight", "loop_q_ctrl_forget.weight"},
                                            {"loop_k_ctrl_down.weight", "loop_k_ctrl_gain.weight", "loop_k_ctrl_forget.weight"},
                                            {"loop_v_ctrl_down.weight", "loop_v_ctrl_gain.weight", "loop_v_ctrl_forget.weight"}};
            ik::CtlJob cj[3];
            for (int j = 0; j < 3; ++j) {
                cj[j] = {in[j], x.msg + j * RE, F(l, ctl[j][0]), F(l, ctl[j][1]), F(l, ctl[j][2]), x.ctl + j * RE};
                in[j] = x.ctl + j * RE;
            }
            ik::loop_control(cj, 3, R, E, g_.gate_rank, kMixEps, x.ctlg, stream_);
        }
        // the input norms and their q8_1 blocks: one input on a sliding layer outside the loop's later passes, else three
        const int distinct = swa && !bank ? 1 : 3;
        static const char* full[3] = {"attn_q_inp_norm.weight", "attn_k_inp_norm.weight", "attn_v_inp_norm.weight"};
        ik::NormJob nj[3];
        for (int j = 0; j < distinct; ++j) nj[j] = {in[j], F(l, swa ? "attn_norm.weight" : full[j])};
        ik::norm_quant(nj, distinct, R, E, g_.rms_eps, x.nrm, x.xq, stream_);
        const void* xq[3];
        for (int j = 0; j < 3; ++j) xq[j] = x.xq + (distinct == 1 ? 0 : j) * R * q8(E);
        Proj pr[8] = {{T(l, "attn_q.weight"), x.q, nq, 0},          {T(l, "attn_k.weight"), x.kraw, nkv, 1},
                      {T(l, "attn_v.weight"), x.vraw, nkv, 2},      {T(l, "attn_rel.weight"), x.r, nrel, 0},
                      {T(l, "attn_sink_proj.weight"), x.sink, g_.n_head, 0}, {T(l, "attn_gate.weight"), x.gate, nq, 0}};
        int n_pr = 6;
        if (!swa) {
            pr[n_pr++] = {T(l, "attn_k_conv_proj.weight"), x.dk, g_.n_kv * 4, 1};
            pr[n_pr++] = {T(l, "attn_v_conv_proj.weight"), x.dv, g_.n_kv * 4, 2};
        }
        gemv_sets(pr, n_pr, xq, E, R);
        const ik::ConvRing ring = ring_of(run, s);
        const ik::KvCaches caches = caches_of(run, s);
        const float kv_scale = 1.0f / std::sqrt((float) ((swa ? 1 : 2) * g_.n_conv + 1));
        const float post_scale = 1.0f / std::sqrt((float) (g_.n_conv + 1));
        ik::ConvJob kj, vj;
        kj.x = x.kraw;
        kj.w = F(l, "attn_k_conv.weight");
        kj.dyn = swa ? nullptr : x.dk;
        kj.out = x.k;
        kj.C = nkv;
        kj.scale = kv_scale;
        vj = kj;
        vj.x = x.vraw;
        vj.w = F(l, "attn_v_conv.weight");
        vj.dyn = swa ? nullptr : x.dv;
        vj.out = x.v;
        vj.ch0 = nkv;
        ik::conv_rows(kj, &vj, kHead, ring, run.m, x.pos, stream_);
        ik::PrepArgs pa;
        pa.n_head = g_.n_head;
        pa.n_kv = g_.n_kv;
        pa.d_rel = g_.d_rel;
        pa.qk_clip = g_.qk_clip;
        pa.r_clip = g_.rel_clip;
        pa.eps = g_.rms_eps;
        pa.log_scale = !swa;
        pa.log_alpha = g_.log_alpha;
        pa.log_floor = g_.log_floor;
        ik::qkvr_prep(x.q, x.k, x.v, x.r, F(l, "attn_q_clip.weight"), F(l, "attn_rel_clip.weight"), pa, run.m, x.pos, caches, cells_[(size_t) s],
                      stream_);
        if (diag()) {
            dump("norm-" + std::to_string(s), x.nrm, (int64_t) RE);
            dump("Qcur-" + std::to_string(s), x.q, (int64_t) R * nq);
            dump("Rcur-" + std::to_string(s), x.r, (int64_t) R * nrel);
            dump("Kconv-" + std::to_string(s), x.k, (int64_t) R * nkv);
            dump("Vconv-" + std::to_string(s), x.v, (int64_t) R * nkv);
            dump("sinks-" + std::to_string(s), x.sink, (int64_t) R * g_.n_head);
            dump("gate-" + std::to_string(s), x.gate, (int64_t) R * nq);
        }
        ik::attention(x.q, x.r, x.sink, x.gate, F(l, "attn_rel_bias.weight"), caches, attn_args(s, l), run.m, x.pos, run.last(), x.ws, x.att,
                      x.attq, stream_);
        if (diag()) dump("att-" + std::to_string(s), x.att, (int64_t) R * nq);
        gemv(T(l, "attn_output.weight"), x.attq, x.o, nq, E, R);
        // the output's convolution into the residual: a block's first layer starts a new stream (the one before joins the
        // mixed states), the others add
        ik::ConvJob oj;
        oj.x = x.o;
        oj.w = F(l, "attn_post_conv.weight");
        oj.out = diag() ? x.oc : nullptr;
        oj.ch0 = 2 * nkv;
        oj.C = E;
        oj.scale = post_scale;
        const bool start = l % g_.block == 0;
        ik::conv_residual(oj, nullptr, nullptr, 0, kHead, ring, run.m, x.pos, x.cur, start ? x.blk[(size_t) s] : nullptr, stream_);
        if (start) blocks.push_back(x.blk[(size_t) s]);
        if (diag()) dump("attn_out-" + std::to_string(s), x.oc, (int64_t) RE);
        if (diag()) dump("ffn_inp-" + std::to_string(s), x.cur, (int64_t) RE);
        // the FFN over the states mixed with its own query
        ik::depth_mix(states(blocks, x.cur), F(l, "ffn_res_query.weight"), 1, R, E, kMixEps, x.fx, stream_);
        const float* fin = x.fx;
        if (bank) {
            const ik::CtlJob fj{x.fx, x.msg + 3 * RE, F(l, "loop_ffn_ctrl_down.weight"), F(l, "loop_ffn_ctrl_gain.weight"),
                                F(l, "loop_ffn_ctrl_forget.weight"), x.fc};
            ik::loop_control(&fj, 1, R, E, g_.gate_rank, kMixEps, x.ctlg, stream_);
            fin = x.fc;
        }
        const auto& lay = m_.expert_layout(l);
        const bool exl3 = bk::cpu::exl3_supported(lay.gu_type);   // EXL3 experts read the normed rows in fp32, the others q8_1
        const ik::NormJob nf{fin, F(l, "ffn_norm.weight")};
        ik::norm_quant(&nf, 1, R, E, g_.rms_eps, x.fn, exl3 ? nullptr : x.fxq, stream_);
        bk::RouterArgs ra;
        ra.n_expert = g_.n_expert;
        ra.n_embd = E;
        ra.k = g_.n_used;
        ra.weight_scale = g_.expert_scale;
        ra.gate = bk::RouterGate::Sigmoid;
        bk::router(x.fn, F(l, "ffn_gate_inp.weight"), F(l, "exp_probs_b.bias"), R, ra, x.rlog, x.ids, x.rw, stream_);
        const auto* rates = m_.expert_rates(l);
        const int entries = R * g_.n_used, groups = std::min(g_.n_expert, entries);
        ik::group_experts(x.ids, R, g_.n_used, g_.n_expert, m_.expert_ptrs(l), rates, x.grp_ptr, x.grp_rates, x.grp_start, x.n_groups,
                          x.ent_dst, x.ent_tok, stream_);
        if (exl3) {
            bk::exl3_expert_grouped(lay, x.grp_ptr, x.grp_start, x.n_groups, x.ent_dst, x.ent_tok, groups, entries, x.fn, x.escr, x.eout,
                                    stream_, 0, rates ? x.grp_rates : nullptr);
        } else {
            bk::native_expert_grouped(lay, x.grp_ptr, x.grp_start, x.n_groups, x.ent_dst, x.ent_tok, groups, entries, x.fxq, x.escr, x.eout,
                                      stream_);
        }
        // the experts' weighted sum, its convolution, into the residual
        ik::ConvJob mj;
        mj.w = F(l, "ffn_post_conv.weight");
        mj.out = diag() ? x.mc : nullptr;
        mj.ch0 = 2 * nkv + E;
        mj.C = E;
        mj.scale = post_scale;
        ik::conv_residual(mj, x.eout, x.rw, g_.n_used, kHead, ring, run.m, x.pos, x.cur, nullptr, stream_);
        if (diag()) dump("l_out-" + std::to_string(s), x.cur, (int64_t) RE);
    }

    ik::AttnArgs attn_args(int s, int l) const {
        const bool swa = g_.swa[(size_t) l];
        ik::AttnArgs aa;
        aa.n_head = g_.n_head;
        aa.n_kv = g_.n_kv;
        aa.d_rel = g_.d_rel;
        aa.extent = swa ? g_.window : g_.rel_extent;
        aa.window = swa ? g_.window : 0;
        aa.cells = cells_[(size_t) s];
        return aa;
    }

    // A decode graph's key: the rows, each member's session and rows, and the attention's split counts (full layers, then
    // sliding ones) at the run's largest position.
    std::vector<uintptr_t> graph_key(const Run& run) const {
        std::vector<uintptr_t> key{(uintptr_t) run.m.n};
        for (int i = 0; i < run.m.n; ++i) {
            key.push_back((uintptr_t) run.s[i]);
            key.push_back((uintptr_t) run.m.start[i + 1]);
            key.push_back((uintptr_t) run.head[i]);
        }
        int full = 0, sliding = 0;
        for (int l = 0; l < g_.n_layers; ++l)
            (g_.swa[(size_t) l] ? sliding : full) = ik::attention_splits(attn_args(l, l), run.last(), run.rows());
        key.push_back((uintptr_t) full);
        key.push_back((uintptr_t) sliding);
        return key;
    }

    // The run's tokens and positions go to the card, then its forward: a decode batch's as a graph (recorded for its key
    // when there is none), a prompt chunk's as launches.
    bool run_rows(const Run& run, const int32_t* tokens, bool decode, std::string& err) {
        Scratch& x = *x_;
        cudaMemcpyAsync(x.tok, tokens, (size_t) run.rows() * 4, cudaMemcpyHostToDevice, stream_);
        ik::set_positions(x.pos, run.m, run.first, stream_);
        if (!decode || diag()) return forward(run, decode, err);
        const std::vector<uintptr_t> key = graph_key(run);
        auto it = std::find_if(x.graphs.begin(), x.graphs.end(), [&](const Scratch::Graph& g) { return g.key == key; });
        if (it == x.graphs.end()) {
            if (x.graphs.size() >= kGraphs)   // the least recently used goes
                x.graphs.erase(std::min_element(x.graphs.begin(), x.graphs.end(),
                                                [](const Scratch::Graph& a, const Scratch::Graph& b) { return a.used < b.used; }));
            Scratch::Graph g;
            g.key = key;
            if (!g.g.begin(stream_, err)) return false;
            const bool ok = forward(run, true, err);
            if (!g.g.end(stream_, err) || !ok) {
                err = "the decode graph: " + err;
                return false;
            }
            x.graphs.push_back(std::move(g));
            it = x.graphs.end() - 1;
        }
        it->used = ++x.tick;
        return it->g.launch(stream_, err);
    }

    // the forward of the run's rows (their positions in x.pos); with `decode`, each member's last head[i] rows' logits go to
    // its session
    bool forward(const Run& run, bool decode, std::string& err) {
        Scratch& x = *x_;
        const int R = run.rows(), E = g_.n_embd;
        const size_t RE = (size_t) R * E;
        const DevTensor* te = m_.tensor("token_embd.weight");
        bk::iq_embed_rows(te->type, te->ptr, bk::iq_row_bytes(te->type, E), x.tok, R, E, x.cur, stream_);
        bk::rms_norm_rows(x.cur, (const float*) m_.tensor("token_embd_norm.weight")->ptr, x.emb, R, E, g_.rms_eps, stream_);
        cudaMemcpyAsync(x.cur, x.emb, RE * 4, cudaMemcpyDeviceToDevice, stream_);
        if (diag()) dump("inp_norm", x.emb, (int64_t) RE);
        std::vector<const float*> blocks;
        for (int l = 0; l < g_.loop_start; ++l) layer(run, l, l, blocks, nullptr);
        cudaMemcpyAsync(x.pre, x.cur, RE * 4, cudaMemcpyDeviceToDevice, stream_);
        const std::vector<const float*> prelude = blocks;
        std::vector<const float*> bank;
        for (int p = 0; p < g_.passes; ++p) {
            if (p > 0) {   // the previous pass's blocks after its first core block, and its output
                bank.assign(blocks.begin() + (ptrdiff_t) prelude.size() + 1, blocks.end());
                bank.push_back(x.out[(p - 1) % 2]);
                cudaMemcpyAsync(x.cur, x.pre, RE * 4, cudaMemcpyDeviceToDevice, stream_);
                blocks = prelude;
            }
            for (int l = g_.loop_start; l < g_.loop_end; ++l) {
                const int s = p == 0 ? l : g_.n_layers + (p - 1) * g_.core() + (l - g_.loop_start);
                layer(run, s, l, blocks, p > 0 ? &bank : nullptr);
            }
            cudaMemcpyAsync(x.out[p % 2], x.cur, RE * 4, cudaMemcpyDeviceToDevice, stream_);
        }
        for (int l = g_.loop_end; l < g_.n_layers; ++l) layer(run, l, l, blocks, nullptr);
        if (decode) {   // the output: the states mixed by its own query, normed, the head, and each member's rows capped
            ik::depth_mix(states(blocks, x.cur), (const float*) m_.tensor("output_res_query.weight")->ptr, 1, R, E, kMixEps, x.fx, stream_);
            const ik::NormJob hj{x.fx, (const float*) m_.tensor("output_norm.weight")->ptr};
            ik::norm_quant(&hj, 1, R, E, g_.rms_eps, x.hn, x.hq, stream_);
            gemv(m_.tensor("output.weight"), x.hq, x.hl, E, g_.vocab_padded, R);
            for (int i = 0; i < run.m.n; ++i) {
                const int row = run.m.start[i + 1] - run.head[i];
                ik::cap_logits(x.hl + (size_t) row * g_.vocab_padded, g_.vocab_padded, run.head[i], g_.n_vocab, g_.logit_cap, run.s[i]->logits_,
                               stream_);
            }
        }
        if (cudaGetLastError() != cudaSuccess) {
            err = "a kernel failed to launch";
            return false;
        }
        return true;
    }

    // The window's draw from this session's logits on `stream`: its picks reach picks_ once the stream is done.
    void draw(const model::DecodeRequest& r, int rows, cudaStream_t stream) {
        bk::SamplerParams sp;
        if (r.sampling) sp = *r.sampling;
        else sp.greedy = true;
        if (!r.sampling) sp.temperature = 0.f;
        sp.logit_bias = bias_on_ ? d_bias_ : nullptr;
        sampling_ = sp;
        bk::sample_tokens(logits_, rows, g_.n_vocab, nullptr, 0, sp, pick_, stream);
        picks_.assign((size_t) rows, 0);
        cudaMemcpyAsync(picks_.data(), pick_, (size_t) rows * 4, cudaMemcpyDeviceToHost, stream);
    }

    Inkling2Model& m_;
    const Geometry& g_;
    std::shared_ptr<Scratch> x_;
    Buffers mem_;
    cudaStream_t stream_ = nullptr;
    int64_t ctx_ = 0;
    int width_ = 0, last_rows_ = 0;
    int64_t pos_ = 0;   // the next position: what the caches hold runs to just before it
    bool bias_on_ = false;
    bk::SamplerParams sampling_;
    float *logits_ = nullptr, *d_bias_ = nullptr, *d_mask_ = nullptr;
    int32_t* pick_ = nullptr;
    std::vector<uint16_t*> kc_, vc_;
    std::vector<int64_t> cells_;
    std::vector<float*> ring_;
    std::vector<int32_t> picks_;
};

}  // namespace

model::Result make_sessions(Inkling2Model& model, int n, std::vector<std::unique_ptr<model::ModelSession>>& sessions) {
    std::string err;
    auto x = std::make_shared<Scratch>();
    if (!x->init(model, err)) return {model::Status::Error, "inkling2x: " + err};
    for (int i = 0; i < n; ++i) {
        auto s = std::make_unique<Session>(model, x);
        if (!s->init(err)) return {model::Status::Error, "inkling2x: " + err};
        sessions.push_back(std::move(s));
    }
    return {};
}

}  // namespace basalt::models::inkling2x
