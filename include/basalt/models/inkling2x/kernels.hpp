// basalt/models/inkling2x/kernels.hpp - Inkling 2's own kernels: the depth mix, the loop controllers, the short
// convolutions, the attention's Q/K/V/R preparation and its attention with a relative-position bias and a sink, the
// routed experts' grouping and sum, and the capped logits. Rows are tokens; every [rows][width] buffer is row-major.
#pragma once

#include <cstdint>

namespace basalt::models::inkling2x::kernels {

/// The states a depth mix reads: up to kMaxStates buffers of [rows][n_embd].
constexpr int kMaxStates = 8;
struct States {
    const float* p[kMaxStates] = {};
    int n = 0;
};

/// The depth mix: each state of a row RMS-normed (no weight, `eps`), scored against each of the `nq` queries ([nq][n_embd])
/// with 1/sqrt(n_embd), the softmax over the states, and their normed values summed by it into out[q][row][n_embd].
void depth_mix(const States& s, const float* queries, int nq, int rows, int n_embd, float eps, float* out, void* stream);

/// A loop controller: out = cur - tanh(F g) * cur + tanh(G g) * msg with g = D [rms(cur); rms(msg)] (the norms without a
/// weight, `eps`). D is [rank][2 n_embd], G (gain) and F (forget) [n_embd][rank], all F32. `g` is [rows][rank] scratch.
void loop_control(const float* cur, const float* msg, const float* down, const float* gain, const float* forget, int rows,
                  int n_embd, int rank, float eps, float* g, float* out, void* stream);

/// The last inputs of a layer slot's convolutions by position: `slots` rows of `width` floats, position p in row p % slots.
struct ConvRing {
    float* data = nullptr;
    int slots = 0;
    int width = 0;
};

/// A causal short convolution of 4 taps with its residual, over rows at positions pos0..: out = (x_t + sum_i w[c][i] x_(t-3+i)
/// + sum_lag dyn[t][h(c) 4 + lag] x_(t-lag)) * scale, h(c) = c / head_width. Inputs before pos0 come from the ring's channels
/// [ch0, ch0 + C) (zero before position 0). `dyn` (null: none) is [rows][C / head_width * 4].
void short_conv(const float* x, const float* w, const float* dyn, int head_width, const ConvRing& ring, int ch0, int64_t pos0,
                int rows, int C, float scale, float* out, void* stream);
/// The ring keeps the inputs of rows [rows - keep, rows) (keep at most ring.slots - 3) at their positions.
void conv_store(const float* x, const ConvRing& ring, int ch0, int64_t pos0, int rows, int keep, int C, void* stream);

/// The attention's inputs of rows at positions pos0..: Q ([rows][n_head][128]) scaled by q_scale ([n_head][128]) and clipped
/// to an L2 norm of qk_clip, K ([rows][n_kv][128]) clipped to qk_clip, V RMS-normed (no weight, eps), R ([rows][n_head][16])
/// scaled by r_scale and clipped to r_clip; on a full-attention layer Q and R times 1 + alpha log(max(1, (p + 1) / floor)).
/// Q and R are written in place, K and V as fp16 to the caches at cell (p % cells) of [cells][n_kv][128].
struct PrepArgs {
    int n_head = 24, n_kv = 8, d_rel = 16;
    float qk_clip = 0, r_clip = 0, eps = 1e-6f;
    bool log_scale = false;
    float log_alpha = 0, log_floor = 1;
};
void qkvr_prep(float* q, const float* k, const float* v, float* r, const float* q_scale, const float* r_scale, const PrepArgs& a,
               int64_t pos0, int rows, uint16_t* k_cache, uint16_t* v_cache, int64_t cells, void* stream);

/// Attention of rows at positions pos0.. over the cached keys: per head, score_j = q . k_j / sqrt(128) + T[min(d_j, extent)]
/// with d_j the distance back, T[e] = R . rel_bias[e] - sink (T[extent] = -sink): the bias and the sink are not scaled.
/// Keys at d_j >= window are out (window 0: none). The softmax takes the scores and one more logit of 0. Out ([rows][n_head][128]) is the attention times
/// sigmoid(gate) ([rows][n_head 128]). `rel_bias` is [extent][16]; `ws` is attention_workspace_bytes of scratch.
struct AttnArgs {
    int n_head = 24, n_kv = 8, d_rel = 16, extent = 1024, window = 0;
    int64_t cells = 0;   ///< the caches' cells: position p in cell p % cells
};
size_t attention_workspace_bytes(int rows, int n_head, int64_t max_keys);
void attention(const float* q, const float* r, const float* sinks, const float* gate, const float* rel_bias,
               const uint16_t* k_cache, const uint16_t* v_cache, const AttnArgs& a, int64_t pos0, int rows, void* ws, float* out,
               void* stream);

/// The routed experts of `rows` rows of k picks (`ids`, [rows][k]) as native_expert_grouped's groups. Expert e's blob is at
/// `expert_ptr[e]` and its rate pair is `expert_class[e]` of `n_class`: class c's groups (one per expert picked, in expert
/// order, its entries in row order) are grp_ptr + c n_expert, grp_start + c (n_expert + 1) and n_groups[c], their entries
/// in ent_dst (row k + j) and ent_tok (row) after those of the classes before, so one call per class reads its own.
void group_experts(const int32_t* ids, int rows, int k, int n_expert, const unsigned long long* expert_ptr,
                   const int32_t* expert_class, int n_class, unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups,
                   int32_t* ent_dst, int32_t* ent_tok, void* stream);
/// y[row] = sum_j w[row][j] out[row k + j], j in order.
void sum_experts(const float* out, const float* w, int rows, int k, int n_embd, float* y, void* stream);

/// logits[row][i] = cap tanh(x[row][i] / cap) for i < n_vocab of rows of `ld` values.
void cap_logits(const float* x, int ld, int rows, int n_vocab, float cap, float* logits, void* stream);

}  // namespace basalt::models::inkling2x::kernels
