// src/models/inkling2x/session.cpp - an Inkling 2 session: the K/V caches and convolution rings of its layer slots, and the
// forward of a run of rows (a decode window or a prompt chunk): the stored layers before the loop, the loop's core once
// per pass (each pass after the first mixing in the one before through the controllers), the layers after it, and the
// output. Every expert of every layer is on the card, so a row's whole forward stays there. A one-row decode replays the
// forward as a CUDA graph, recorded again when the attention's split count changes (every 256 positions).
#include "basalt/core/graph.hpp"
#include "basalt/kernels/cpu/exl3_cpu.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/elementwise.hpp"
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
#include <utility>
#include <cstdio>
#include <string>
#include <vector>

namespace basalt::models::inkling2x {
namespace {

namespace bk = basalt::kernels;
namespace ik = inkling2x::kernels;

constexpr int kMaxWindow = 16;   // rows of a decode window
constexpr int kChunk = 128;      // rows of a prompt chunk: the most a forward takes
constexpr int kRing = 32;        // positions a convolution ring keeps: a window's rows and the three taps before them
constexpr int kHead = 128;
constexpr float kMixEps = 1e-6f;

class Session final : public model::ModelSession {
public:
    explicit Session(Inkling2Model& m) : m_(m), g_(m.geo()) {}
    ~Session() override {
        for (void* p : bufs_) cudaFree(p);
        if (stream_) cudaStreamDestroy(stream_);
    }

    bool init(std::string& err) {
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) { err = "no CUDA stream"; return false; }
        const int E = g_.n_embd, R = kChunk, K = g_.n_used;
        const int nq = g_.n_head * kHead, nkv = g_.n_kv * kHead, nrel = g_.n_head * g_.d_rel;
        width_ = 2 * nkv + 2 * E;   // K, V, the attention's output, the FFN's output
        ctx_ = m_.context();
        auto f = [&](float*& p, size_t n) { return alloc((void**) &p, n * sizeof(float), err); };
        auto b = [&](void*& p, size_t n) { return alloc(&p, n, err); };
        const size_t q8 = (size_t) E / 32 * 36, q8a = (size_t) nq / 32 * 36;
        bool ok = alloc((void**) &tok_, (size_t) R * 4, err) && f(emb_, (size_t) R * E) && f(cur_, (size_t) R * E) &&
                  f(pre_, (size_t) R * E) && f(out_[0], (size_t) R * E) && f(out_[1], (size_t) R * E) &&
                  f(mix_, 3 * (size_t) R * E) && f(msg_, 4 * (size_t) R * E) && f(ctl_, 3 * (size_t) R * E) &&
                  f(nrm_, 3 * (size_t) R * E) && b(xq_, 3 * (size_t) R * q8) && f(q_, (size_t) R * nq) && f(kraw_, (size_t) R * nkv) &&
                  f(vraw_, (size_t) R * nkv) && f(k_, (size_t) R * nkv) && f(v_, (size_t) R * nkv) && f(r_, (size_t) R * nrel) &&
                  f(sink_, (size_t) R * g_.n_head) && f(gate_, (size_t) R * nq) && f(dk_, (size_t) R * g_.n_kv * 4) &&
                  f(dv_, (size_t) R * g_.n_kv * 4) && f(att_, (size_t) R * nq) && b(attq_, (size_t) R * q8a) && f(o_, (size_t) R * E) &&
                  f(oc_, (size_t) R * E) && f(fx_, (size_t) R * E) && f(fc_, (size_t) R * E) && f(fn_, (size_t) R * E) &&
                  b(fxq_, (size_t) R * q8) && f(rlog_, (size_t) R * g_.n_expert) && alloc((void**) &ids_, (size_t) R * K * 4, err) &&
                  f(rw_, (size_t) R * K) && alloc((void**) &grp_ptr_, g_.n_expert * sizeof(*grp_ptr_), err) &&
                  alloc((void**) &grp_rates_, g_.n_expert * sizeof(*grp_rates_), err) && alloc((void**) &grp_start_, (g_.n_expert + 1) * 4, err) &&
                  alloc((void**) &n_groups_, 4, err) &&
                  alloc((void**) &ent_dst_, (size_t) R * K * 4, err) && alloc((void**) &ent_tok_, (size_t) R * K * 4, err) &&
                  f(eout_, (size_t) R * K * E) && f(moe_, (size_t) R * E) && f(mc_, (size_t) R * E) && f(ctlg_, (size_t) R * g_.gate_rank) &&
                  f(hn_, (size_t) R * E) && b(hq_, (size_t) kMaxWindow * q8) && f(hl_, (size_t) kMaxWindow * g_.vocab_padded) &&
                  f(logits_, (size_t) kMaxWindow * g_.n_vocab) && alloc((void**) &pick_, (size_t) (3 * kMaxWindow) * 4, err) &&
                  f(lp_, bk::row_logprobs_scratch_floats(kMaxWindow, g_.n_vocab) + (size_t) kMaxWindow * bk::kLogprobStride) &&
                  f(d_bias_, (size_t) g_.n_vocab) && f(d_mask_, (size_t) g_.n_vocab) && alloc((void**) &d_pos_, sizeof(int64_t), err);
        if (!ok) return false;
        size_t escr = 0;
        for (int l = 0; l < g_.n_layers; ++l)
            escr = std::max(escr, bk::native_expert_scratch_bytes(m_.expert_layout(l), (int64_t) R * K));
        if (!b(escr_, escr) || !b(ws_, ik::attention_workspace_bytes(R, g_.n_head, ctx_))) return false;
        // the layer slots: a K/V cache and a convolution ring each, and a block buffer where a block starts
        const int S = g_.slots();
        kc_.assign((size_t) S, nullptr);
        vc_.assign((size_t) S, nullptr);
        cells_.assign((size_t) S, 0);
        ring_.assign((size_t) S, nullptr);
        blk_.assign((size_t) S, nullptr);
        for (int s = 0; s < S; ++s) {
            const int l = g_.layer_of(s);
            cells_[(size_t) s] = g_.swa[(size_t) l] ? g_.window + kChunk : ctx_;
            const size_t cb = (size_t) cells_[(size_t) s] * nkv * 2;
            if (!b((void*&) kc_[(size_t) s], cb) || !b((void*&) vc_[(size_t) s], cb) || !f(ring_[(size_t) s], (size_t) kRing * width_)) return false;
            if (l % g_.block == 0 && !f(blk_[(size_t) s], (size_t) R * E)) return false;
        }
        return cudaStreamSynchronize(stream_) == cudaSuccess || (err = "the session's buffers failed", false);
    }

    model::Result decode_window(const model::DecodeRequest& r) override {
        if (r.n_tokens < 1 || r.n_tokens > kMaxWindow) return {model::Status::Error, "inkling2x: a window of " + std::to_string(r.n_tokens) + " rows"};
        if (r.pos + r.n_tokens > ctx_) return {model::Status::Refused, "inkling2x: the window passes the context of " + std::to_string(ctx_)};
        std::string err;
        const int head = r.row_tokens ? r.n_tokens : 1;
        if (!run_rows(r.tokens, r.n_tokens, r.pos, head, err) || !pick(r, head, err)) return {model::Status::Error, "inkling2x: " + err};
        if (r.row_tokens)
            for (int i = 0; i < r.n_tokens; ++i) r.row_tokens[i] = picks_[(size_t) i];
        if (r.next_token) *r.next_token = picks_[(size_t) head - 1];
        if (r.logits) cudaMemcpyAsync(r.logits, logits_ + (size_t) (head - 1) * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost, stream_);
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "inkling2x: the window failed"};
        last_rows_ = head;
        return {};
    }

    model::Result prefill_chunk(const model::PrefillRequest& r) override {
        if (r.pos + r.n_tokens > ctx_) return {model::Status::Refused, "inkling2x: the prompt passes the context of " + std::to_string(ctx_)};
        std::string err;
        for (int t0 = 0; t0 < r.n_tokens; t0 += kChunk) {
            if (r.stop && r.stop()) break;
            const int n = std::min(kChunk, r.n_tokens - t0);
            if (!run_rows(r.tokens + t0, n, r.pos + t0, 0, err)) return {model::Status::Error, "inkling2x: " + err};
            if (r.progress) {
                cudaStreamSynchronize(stream_);
                r.progress(t0 + n);
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "inkling2x: the prompt failed"};
        return {};
    }

    model::Result set_logit_bias(const float* bias) override {
        bias_on_ = bias != nullptr;
        if (bias_on_ && cudaMemcpyAsync(d_bias_, bias, (size_t) g_.n_vocab * 4, cudaMemcpyHostToDevice, stream_) != cudaSuccess)
            return {model::Status::Error, "inkling2x: the logit bias upload failed"};
        return {};
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
        float* d_out = lp_ + bk::row_logprobs_scratch_floats(n, g_.n_vocab);
        int32_t* d_pick = pick_ + kMaxWindow;
        cudaMemcpyAsync(d_pick, picks, (size_t) n * 4, cudaMemcpyHostToDevice, stream_);
        bk::row_logprobs(logits_ + (size_t) row0 * g_.n_vocab, n, g_.n_vocab, d_pick, top, d_out, lp_, stream_);
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
    bool alloc(void** p, size_t n, std::string& err) {
        if (cudaMalloc(p, n) != cudaSuccess) {
            cudaGetLastError();
            err = "out of device memory for the session (" + std::to_string(n >> 20) + " MiB more)";
            return false;
        }
        cudaMemsetAsync(*p, 0, n, stream_);
        bufs_.push_back(*p);
        return true;
    }

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
    void quant(const float* x, int R, int n, void* xq) { bk::quantize_q8_1_rows(x, R, n, xq, stream_); }

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

    // one layer slot over the R rows at pos0: slot s of stored layer l; `bank` (the previous pass's states) is set on the
    // loop's later passes
    void layer(int s, int l, int64_t pos0, int R, std::vector<const float*>& blocks, const std::vector<const float*>* bank) {
        const int E = g_.n_embd, nq = g_.n_head * kHead, nkv = g_.n_kv * kHead, nrel = g_.n_head * g_.d_rel;
        const bool swa = g_.swa[(size_t) l];
        const size_t RE = (size_t) R * E;
        const float* in[3] = {cur_, cur_, cur_};
        if (l > 0) {   // the depth mix: one input on a sliding layer, one each for Q, K and V on a full one
            ik::depth_mix(states(blocks, cur_), F(l, "attn_res_query.weight"), swa ? 1 : 3, R, E, kMixEps, mix_, stream_);
            for (int j = 0; j < 3; ++j) in[j] = mix_ + (swa ? 0 : j) * RE;
        }
        if (bank) {   // the messages of the previous pass, mixed into Q, K and V
            ik::States bs;
            for (const float* p : *bank) bs.p[bs.n++] = p;
            ik::depth_mix(bs, F(l, "loop_msg_query.weight"), 4, R, E, kMixEps, msg_, stream_);
            static const char* ctl[3][3] = {{"loop_q_ctrl_down.weight", "loop_q_ctrl_gain.weight", "loop_q_ctrl_forget.weight"},
                                            {"loop_k_ctrl_down.weight", "loop_k_ctrl_gain.weight", "loop_k_ctrl_forget.weight"},
                                            {"loop_v_ctrl_down.weight", "loop_v_ctrl_gain.weight", "loop_v_ctrl_forget.weight"}};
            for (int j = 0; j < 3; ++j) {
                ik::loop_control(in[j], msg_ + j * RE, F(l, ctl[j][0]), F(l, ctl[j][1]), F(l, ctl[j][2]), R, E, g_.gate_rank, kMixEps, ctlg_,
                                 ctl_ + j * RE, stream_);
                in[j] = ctl_ + j * RE;
            }
        }
        // the input norms, and each distinct normed input quantized once
        int distinct = 3;
        if (swa && !bank) {
            bk::rms_norm_rows(in[0], F(l, "attn_norm.weight"), nrm_, R, E, g_.rms_eps, stream_);
            distinct = 1;
        } else {
            static const char* full[3] = {"attn_q_inp_norm.weight", "attn_k_inp_norm.weight", "attn_v_inp_norm.weight"};
            for (int j = 0; j < 3; ++j) bk::rms_norm_rows(in[j], F(l, swa ? "attn_norm.weight" : full[j]), nrm_ + j * RE, R, E, g_.rms_eps, stream_);
        }
        for (int j = 0; j < distinct; ++j) quant(nrm_ + j * RE, R, E, (uint8_t*) xq_ + j * R * q8(E));
        const void* xq[3];
        for (int j = 0; j < 3; ++j) xq[j] = (uint8_t*) xq_ + (distinct == 1 ? 0 : j) * R * q8(E);
        Proj pr[8] = {{T(l, "attn_q.weight"), q_, nq, 0},          {T(l, "attn_k.weight"), kraw_, nkv, 1},
                      {T(l, "attn_v.weight"), vraw_, nkv, 2},      {T(l, "attn_rel.weight"), r_, nrel, 0},
                      {T(l, "attn_sink_proj.weight"), sink_, g_.n_head, 0}, {T(l, "attn_gate.weight"), gate_, nq, 0}};
        int n_pr = 6;
        if (!swa) {
            pr[n_pr++] = {T(l, "attn_k_conv_proj.weight"), dk_, g_.n_kv * 4, 1};
            pr[n_pr++] = {T(l, "attn_v_conv_proj.weight"), dv_, g_.n_kv * 4, 2};
        }
        gemv_sets(pr, n_pr, xq, E, R);
        const ik::ConvRing ring{ring_[(size_t) s], kRing, width_};
        const float kv_scale = 1.0f / std::sqrt((float) ((swa ? 1 : 2) * g_.n_conv + 1));
        const float post_scale = 1.0f / std::sqrt((float) (g_.n_conv + 1));
        const int keep = std::min(R, kRing - 3);
        ik::short_conv(kraw_, F(l, "attn_k_conv.weight"), swa ? nullptr : dk_, kHead, ring, 0, d_pos_, R, nkv, kv_scale, k_, stream_);
        ik::short_conv(vraw_, F(l, "attn_v_conv.weight"), swa ? nullptr : dv_, kHead, ring, nkv, d_pos_, R, nkv, kv_scale, v_, stream_);
        ik::conv_store(kraw_, ring, 0, d_pos_, R, keep, nkv, stream_);
        ik::conv_store(vraw_, ring, nkv, d_pos_, R, keep, nkv, stream_);
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
        ik::qkvr_prep(q_, k_, v_, r_, F(l, "attn_q_clip.weight"), F(l, "attn_rel_clip.weight"), pa, d_pos_, R, kc_[(size_t) s], vc_[(size_t) s],
                      cells_[(size_t) s], stream_);
        if (diag()) {
            dump("norm-" + std::to_string(s), nrm_, (int64_t) RE);
            dump("Qcur-" + std::to_string(s), q_, (int64_t) R * nq);
            dump("Rcur-" + std::to_string(s), r_, (int64_t) R * nrel);
            dump("Kconv-" + std::to_string(s), k_, (int64_t) R * nkv);
            dump("Vconv-" + std::to_string(s), v_, (int64_t) R * nkv);
            dump("sinks-" + std::to_string(s), sink_, (int64_t) R * g_.n_head);
            dump("gate-" + std::to_string(s), gate_, (int64_t) R * nq);
        }
        const ik::AttnArgs aa = attn_args(s, l);
        ik::attention(q_, r_, sink_, gate_, F(l, "attn_rel_bias.weight"), kc_[(size_t) s], vc_[(size_t) s], aa, d_pos_, pos0, R, ws_, att_,
                      stream_);
        if (diag()) dump("att-" + std::to_string(s), att_, (int64_t) R * nq);
        quant(att_, R, nq, attq_);
        gemv(T(l, "attn_output.weight"), attq_, o_, nq, E, R);
        ik::short_conv(o_, F(l, "attn_post_conv.weight"), nullptr, kHead, ring, 2 * nkv, d_pos_, R, E, post_scale, oc_, stream_);
        ik::conv_store(o_, ring, 2 * nkv, d_pos_, R, keep, E, stream_);
        if (diag()) dump("attn_out-" + std::to_string(s), oc_, (int64_t) RE);
        // a block's first layer starts a new residual stream; the one before joins the mixed states
        if (l % g_.block == 0) {
            cudaMemcpyAsync(blk_[(size_t) s], cur_, RE * 4, cudaMemcpyDeviceToDevice, stream_);
            blocks.push_back(blk_[(size_t) s]);
            cudaMemcpyAsync(cur_, oc_, RE * 4, cudaMemcpyDeviceToDevice, stream_);
        } else {
            bk::add_inplace(cur_, oc_, (int64_t) RE, stream_);
        }
        if (diag()) dump("ffn_inp-" + std::to_string(s), cur_, (int64_t) RE);
        // the FFN over the states mixed with its own query
        ik::depth_mix(states(blocks, cur_), F(l, "ffn_res_query.weight"), 1, R, E, kMixEps, fx_, stream_);
        const float* fin = fx_;
        if (bank) {
            ik::loop_control(fx_, msg_ + 3 * RE, F(l, "loop_ffn_ctrl_down.weight"), F(l, "loop_ffn_ctrl_gain.weight"),
                             F(l, "loop_ffn_ctrl_forget.weight"), R, E, g_.gate_rank, kMixEps, ctlg_, fc_, stream_);
            fin = fc_;
        }
        bk::rms_norm_rows(fin, F(l, "ffn_norm.weight"), fn_, R, E, g_.rms_eps, stream_);
        bk::RouterArgs ra;
        ra.n_expert = g_.n_expert;
        ra.n_embd = E;
        ra.k = g_.n_used;
        ra.weight_scale = g_.expert_scale;
        ra.gate = bk::RouterGate::Sigmoid;
        bk::router(fn_, F(l, "ffn_gate_inp.weight"), F(l, "exp_probs_b.bias"), R, ra, rlog_, ids_, rw_, stream_);
        const auto& lay = m_.expert_layout(l);
        const auto* rates = m_.expert_rates(l);
        const int entries = R * g_.n_used, groups = std::min(g_.n_expert, entries);
        ik::group_experts(ids_, R, g_.n_used, g_.n_expert, m_.expert_ptrs(l), rates, grp_ptr_, grp_rates_, grp_start_, n_groups_,
                          ent_dst_, ent_tok_, stream_);
        if (bk::cpu::exl3_supported(lay.gu_type)) {
            bk::exl3_expert_grouped(lay, grp_ptr_, grp_start_, n_groups_, ent_dst_, ent_tok_, groups, entries, fn_, escr_, eout_, stream_,
                                    0, rates ? grp_rates_ : nullptr);
        } else {
            quant(fn_, R, E, fxq_);
            bk::native_expert_grouped(lay, grp_ptr_, grp_start_, n_groups_, ent_dst_, ent_tok_, groups, entries, fxq_, escr_, eout_, stream_);
        }
        ik::sum_experts(eout_, rw_, R, g_.n_used, E, moe_, stream_);
        ik::short_conv(moe_, F(l, "ffn_post_conv.weight"), nullptr, kHead, ring, 2 * nkv + E, d_pos_, R, E, post_scale, mc_, stream_);
        ik::conv_store(moe_, ring, 2 * nkv + E, d_pos_, R, keep, E, stream_);
        bk::add_inplace(cur_, mc_, (int64_t) RE, stream_);
        if (diag()) dump("l_out-" + std::to_string(s), cur_, (int64_t) RE);
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

    // The attention's split counts of one row at p_last, full layers then sliding ones: a graph replays where they match.
    std::pair<int, int> splits_at(int64_t p_last) const {
        int full = 0, sliding = 0;
        for (int l = 0; l < g_.n_layers; ++l) (g_.swa[(size_t) l] ? sliding : full) = ik::attention_splits(attn_args(l, l), p_last, 1);
        return {full, sliding};
    }

    // The R rows at pos0 (the last `head` get logits): the tokens and the position go to the card, then the forward, as
    // the graph for a one-row decode
    bool run_rows(const int32_t* tokens, int R, int64_t pos0, int head, std::string& err) {
        cudaMemcpyAsync(tok_, tokens, (size_t) R * 4, cudaMemcpyHostToDevice, stream_);
        ik::set_position(d_pos_, pos0, stream_);
        if (R != 1 || head != 1 || diag()) {
            if (!forward(R, pos0, head, err)) return false;
        } else {
            const std::pair<int, int> key = splits_at(pos0);
            if (!graph_.valid() || key != graph_key_) {
                graph_ = core::CapturedGraph();
                if (!graph_.begin(stream_, err)) return false;
                const bool ok = forward(1, pos0, 1, err);
                if (!graph_.end(stream_, err) || !ok) {
                    graph_ = core::CapturedGraph();
                    err = "the decode graph: " + err;
                    return false;
                }
                graph_key_ = key;
            }
            if (!graph_.launch(stream_, err)) return false;
        }
        pos_ = pos0 + R;
        return true;
    }

    // the forward of R rows at pos0 (in d_pos_); the last `head` rows get logits
    bool forward(int R, int64_t pos0, int head, std::string& err) {
        const int E = g_.n_embd;
        const size_t RE = (size_t) R * E;
        const DevTensor* te = m_.tensor("token_embd.weight");
        bk::iq_embed_rows(te->type, te->ptr, bk::iq_row_bytes(te->type, E), tok_, R, E, cur_, stream_);
        bk::rms_norm_rows(cur_, (const float*) m_.tensor("token_embd_norm.weight")->ptr, emb_, R, E, g_.rms_eps, stream_);
        cudaMemcpyAsync(cur_, emb_, RE * 4, cudaMemcpyDeviceToDevice, stream_);
        if (diag()) dump("inp_norm", emb_, (int64_t) RE);
        std::vector<const float*> blocks;
        for (int l = 0; l < g_.loop_start; ++l) layer(l, l, pos0, R, blocks, nullptr);
        cudaMemcpyAsync(pre_, cur_, RE * 4, cudaMemcpyDeviceToDevice, stream_);
        const std::vector<const float*> prelude = blocks;
        std::vector<const float*> bank;
        for (int p = 0; p < g_.passes; ++p) {
            if (p > 0) {   // the previous pass's blocks after its first core block, and its output
                bank.assign(blocks.begin() + (ptrdiff_t) prelude.size() + 1, blocks.end());
                bank.push_back(out_[(p - 1) % 2]);
                cudaMemcpyAsync(cur_, pre_, RE * 4, cudaMemcpyDeviceToDevice, stream_);
                blocks = prelude;
            }
            for (int l = g_.loop_start; l < g_.loop_end; ++l) {
                const int s = p == 0 ? l : g_.n_layers + (p - 1) * g_.core() + (l - g_.loop_start);
                layer(s, l, pos0, R, blocks, p > 0 ? &bank : nullptr);
            }
            cudaMemcpyAsync(out_[p % 2], cur_, RE * 4, cudaMemcpyDeviceToDevice, stream_);
        }
        for (int l = g_.loop_end; l < g_.n_layers; ++l) layer(l, l, pos0, R, blocks, nullptr);
        if (head > 0) {   // the output: the states mixed by its own query, normed, the head and the cap, for the last `head` rows
            ik::depth_mix(states(blocks, cur_), (const float*) m_.tensor("output_res_query.weight")->ptr, 1, R, E, kMixEps, fx_, stream_);
            const float* last = fx_ + (size_t) (R - head) * E;
            bk::rms_norm_rows(last, (const float*) m_.tensor("output_norm.weight")->ptr, hn_, head, E, g_.rms_eps, stream_);
            quant(hn_, head, E, hq_);
            gemv(m_.tensor("output.weight"), hq_, hl_, E, g_.vocab_padded, head);
            ik::cap_logits(hl_, g_.vocab_padded, head, g_.n_vocab, g_.logit_cap, logits_, stream_);
        }
        if (cudaGetLastError() != cudaSuccess) {
            err = "a kernel failed to launch";
            return false;
        }
        return true;
    }

    bool pick(const model::DecodeRequest& r, int rows, std::string& err) {
        bk::SamplerParams sp;
        if (r.sampling) sp = *r.sampling;
        else sp.greedy = true;
        if (!r.sampling) sp.temperature = 0.f;
        sp.logit_bias = bias_on_ ? d_bias_ : nullptr;
        sampling_ = sp;
        bk::sample_tokens(logits_, rows, g_.n_vocab, nullptr, 0, sp, pick_, stream_);
        picks_.assign((size_t) rows, 0);
        if (cudaMemcpyAsync(picks_.data(), pick_, (size_t) rows * 4, cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            err = "the pick failed";
            return false;
        }
        return true;
    }

    Inkling2Model& m_;
    const Geometry& g_;
    cudaStream_t stream_ = nullptr;
    std::vector<void*> bufs_;
    int64_t ctx_ = 0;
    int width_ = 0, last_rows_ = 0;
    int64_t pos_ = 0;   // the next position: what the caches hold runs to just before it
    bool bias_on_ = false;
    bk::SamplerParams sampling_;
    float *lp_ = nullptr, *d_bias_ = nullptr, *d_mask_ = nullptr;
    int32_t* tok_ = nullptr;
    int64_t* d_pos_ = nullptr;   // the first row's position, which the kernels read
    core::CapturedGraph graph_;  // the one-row decode
    std::pair<int, int> graph_key_{-1, -1};
    float *emb_ = nullptr, *cur_ = nullptr, *pre_ = nullptr, *out_[2] = {nullptr, nullptr}, *mix_ = nullptr, *msg_ = nullptr, *ctl_ = nullptr,
          *nrm_ = nullptr;
    void *xq_ = nullptr, *attq_ = nullptr, *fxq_ = nullptr, *hq_ = nullptr, *escr_ = nullptr, *ws_ = nullptr;
    float *q_ = nullptr, *kraw_ = nullptr, *vraw_ = nullptr, *k_ = nullptr, *v_ = nullptr, *r_ = nullptr, *sink_ = nullptr, *gate_ = nullptr,
          *dk_ = nullptr, *dv_ = nullptr, *att_ = nullptr, *o_ = nullptr, *oc_ = nullptr, *fx_ = nullptr, *fc_ = nullptr, *fn_ = nullptr,
          *rlog_ = nullptr, *rw_ = nullptr, *eout_ = nullptr, *moe_ = nullptr, *mc_ = nullptr, *ctlg_ = nullptr, *hn_ = nullptr, *hl_ = nullptr,
          *logits_ = nullptr;
    int32_t *ids_ = nullptr, *grp_start_ = nullptr, *n_groups_ = nullptr, *ent_dst_ = nullptr, *ent_tok_ = nullptr, *pick_ = nullptr;
    unsigned long long* grp_ptr_ = nullptr;
    bk::Exl3Rates* grp_rates_ = nullptr;
    std::vector<uint16_t*> kc_, vc_;
    std::vector<int64_t> cells_;
    std::vector<float*> ring_, blk_;
    std::vector<int32_t> picks_;
};

}  // namespace

std::unique_ptr<model::ModelSession> make_session(Inkling2Model& model, std::string& err) {
    auto s = std::make_unique<Session>(model);
    if (!s->init(err)) return nullptr;
    return s;
}

}  // namespace basalt::models::inkling2x
