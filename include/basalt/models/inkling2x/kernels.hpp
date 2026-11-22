// basalt/models/inkling2x/kernels.hpp - Inkling 2's own kernels: the depth mix, the norms with their q8_1 blocks, the loop
// controllers, the short convolutions (with the residual and the routed experts' sum), the attention's Q/K/V/R
// preparation and its attention with a relative-position bias and a sink, the routed experts' grouping, and the capped
// logits. Rows are tokens; every [rows][width] buffer is row-major.
// A run of rows may hold several conversations (`Members`, a decode batch): the kernels that touch a conversation's own
// state (its convolution rings and K/V caches) take each member's, and those that depend on the rows' positions read
// each row's from `pos`, a device array (set_positions), so a captured graph replays at new positions.
#pragma once

#include "basalt/kernels/exl3_kernels.hpp"

#include <cstdint>

namespace basalt::models::inkling2x::kernels {

/// The states a depth mix reads: up to kMaxStates buffers of [rows][n_embd].
constexpr int kMaxStates = 8;
struct States {
    const float* p[kMaxStates] = {};
    int n = 0;
};

/// The conversations of a run of rows: member m holds rows [start[m], start[m + 1]) at consecutive positions.
constexpr int kMaxMembers = 16;
struct Members {
    int n = 1;
    int start[kMaxMembers + 1] = {};
    int rows() const { return start[n]; }
};
/// pos[row] = first[m] + (row - start[m]) for the rows of each member m (first: host, m.n of them), on the stream.
void set_positions(int64_t* pos, const Members& m, const int64_t* first, void* stream);

/// The depth mix: each state of a row RMS-normed (no weight, `eps`), scored against each of the `nq` queries ([nq][n_embd])
/// with 1/sqrt(n_embd), the softmax over the states, and their normed values summed by it into out[q][row][n_embd].
void depth_mix(const States& s, const float* queries, int nq, int rows, int n_embd, float eps, float* out, void* stream);

/// RMS norms with their weights, out[j][row] = x_j[row] * rsqrt(mean(x_j[row]^2) + eps) * w_j, and their q8_1 blocks
/// xq[j][row] (null: none), for up to 3 inputs of [rows][n_embd] in one launch: rms_norm_rows's and quantize_q8_1_rows's
/// arithmetic, the same bits.
struct NormJob {
    const float* x = nullptr;
    const float* w = nullptr;
};
void norm_quant(const NormJob* jobs, int n, int rows, int n_embd, float eps, float* out, void* xq, void* stream);

/// Loop controllers: out = cur - tanh(F g) * cur + tanh(G g) * msg with g = D [rms(cur); rms(msg)] (the norms without a
/// weight, `eps`). D is [rank][2 n_embd], G (gain) and F (forget) [n_embd][rank], all F32. Up to 3 controllers in one
/// launch a phase; `g` is [n][rows][rank] scratch.
struct CtlJob {
    const float *cur = nullptr, *msg = nullptr, *down = nullptr, *gain = nullptr, *forget = nullptr;
    float* out = nullptr;
};
void loop_control(const CtlJob* jobs, int n, int rows, int n_embd, int rank, float eps, float* g, void* stream);

/// The last inputs of a layer slot's convolutions by position, a ring for each member: `slots` rows of `width` floats,
/// position p in row p % slots.
struct ConvRing {
    float* data[kMaxMembers] = {};
    int slots = 0;
    int width = 0;
};

/// A causal short convolution of 4 taps with its residual, over each member's rows: out = (x_t + sum_i w[c][i] x_(t-3+i)
/// + sum_lag dyn[t][h(c) 4 + lag] x_(t-lag)) * scale, h(c) = c / head_width. Inputs before a member's first row come from
/// its ring's channels [ch0, ch0 + C) (zero before position 0), and its ring keeps the inputs of its last rows (at most
/// ring.slots - 3) at their positions. `dyn` (null: none) is [rows][C / head_width * 4]; `out` [rows][C] (null: not
/// written). A thread takes a channel through every row in order, so its ring cells are its own and any number of rows is
/// one launch.
struct ConvJob {
    const float* x = nullptr;
    const float* w = nullptr;
    const float* dyn = nullptr;
    float* out = nullptr;
    int ch0 = 0, C = 0;
    float scale = 1.0f;
};
/// One or two convolutions (`b` null: one) in one launch.
void conv_rows(const ConvJob& a, const ConvJob* b, int head_width, const ConvRing& ring, const Members& m, const int64_t* pos,
               void* stream);
/// One convolution whose result is the residual: added to cur ([rows][C]), or, with `save` set (a block's first layer),
/// cur copied to save and replaced. With `eout` set its input is the routed experts' sum, x[row] = sum_j ew[row][j]
/// eout[row k + j] in j order (a.x unused).
void conv_residual(const ConvJob& a, const float* eout, const float* ew, int k, int head_width, const ConvRing& ring,
                   const Members& m, const int64_t* pos, float* cur, float* save, void* stream);

/// Each member's K/V cache of a layer slot: [cells][n_kv][128] fp16, position p in cell p % cells.
struct KvCaches {
    uint16_t* k[kMaxMembers] = {};
    uint16_t* v[kMaxMembers] = {};
};

/// The attention's inputs of the rows at positions pos[row]: Q ([rows][n_head][128]) scaled by q_scale ([n_head][128]) and clipped
/// to an L2 norm of qk_clip, K ([rows][n_kv][128]) clipped to qk_clip, V RMS-normed (no weight, eps), R ([rows][n_head][16])
/// scaled by r_scale and clipped to r_clip; on a full-attention layer Q and R times 1 + alpha log(max(1, (p + 1) / floor)).
/// Q and R are written in place, K and V as fp16 to each row's member's caches at cell (p % cells).
struct PrepArgs {
    int n_head = 24, n_kv = 8, d_rel = 16;
    float qk_clip = 0, r_clip = 0, eps = 1e-6f;
    bool log_scale = false;
    float log_alpha = 0, log_floor = 1;
};
void qkvr_prep(float* q, const float* k, const float* v, float* r, const float* q_scale, const float* r_scale, const PrepArgs& a,
               const Members& m, const int64_t* pos, const KvCaches& caches, int64_t cells, void* stream);

/// Attention of the rows at positions pos[row] over their members' cached keys: per head, score_j = q . k_j / sqrt(128) + T[min(d_j, extent)]
/// with d_j the distance back, T[e] = R . rel_bias[e] - sink (T[extent] = -sink): the bias and the sink are not scaled.
/// Keys at d_j >= window are out (window 0: none). The softmax takes the scores and one more logit of 0. Out ([rows][n_head][128]) is the attention times
/// sigmoid(gate) ([rows][n_head 128]). `rel_bias` is [extent][16]; `ws` is attention_workspace_bytes of scratch. `p_last`
/// (the last row's position, the largest of every member's) sizes the grid only: the keys are split in runs (64 keys for up
/// to 16 rows, 256 above), a row's splits past its keys are empty, and a decode launch rounds its split count up to a
/// multiple of 8, so it is good for every position with the same count. The query heads of a K/V head (up to 8) share a
/// block.
struct AttnArgs {
    int n_head = 24, n_kv = 8, d_rel = 16, extent = 1024, window = 0;
    int64_t cells = 0;   ///< the caches' cells: position p in cell p % cells
};
size_t attention_workspace_bytes(int rows, int n_head, int64_t max_keys);
/// `out_q8_1` (null: none) gets out's q8_1 blocks as quantize_q8_1_rows would make them.
void attention(const float* q, const float* r, const float* sinks, const float* gate, const float* rel_bias, const KvCaches& caches,
               const AttnArgs& a, const Members& m, const int64_t* pos, int64_t p_last, void* ws, float* out, void* out_q8_1,
               void* stream);
/// The split count of a launch of `rows` rows whose largest position is p_last: equal counts can share a captured launch.
int attention_splits(const AttnArgs& a, int64_t p_last, int rows);

/// The routed experts of `rows` rows of k picks (`ids`, [rows][k]) as native_expert_grouped's groups. Expert e's blob is at
/// `expert_ptr[e]`. One group per expert picked, in expert order; entries in row order. In a mixed-rate layer,
/// copy each picked expert's EXL3 rates to its group; pass null expert_rates for a uniform layer.
using ::basalt::kernels::Exl3Rates;
void group_experts(const int32_t* ids, int rows, int k, int n_expert, const unsigned long long* expert_ptr,
                   const Exl3Rates* expert_rates, unsigned long long* grp_ptr, Exl3Rates* grp_rates, int32_t* grp_start,
                   int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok, void* stream);

/// logits[row][i] = cap tanh(x[row][i] / cap) for i < n_vocab of rows of `ld` values.
void cap_logits(const float* x, int ld, int rows, int n_vocab, float cap, float* logits, void* stream);

}  // namespace basalt::models::inkling2x::kernels
