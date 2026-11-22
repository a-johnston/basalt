// basalt/models/inkling2x/dspark_kernels.hpp - the kernels of Inkling 2's DSpark drafter (vLLM's
// vllm/models/inkling2/nvidia/dspark.py and ops/dspark.py, Apache-2.0): the target's features kept per row, their mixes,
// the drafter's context keys and values, its block's queries and keys, its residual norms and gated MLP, and the
// recurrent Markov head. Rows are [rows][width] row-major; positions come from device arrays, so a recorded graph replays.
#pragma once

#include "basalt/models/inkling2x/kernels.hpp"

#include <cstdint>

namespace basalt::models::inkling2x::kernels {

constexpr int kMaxFeatures = 4;   ///< target features a row keeps (the drafter's target_layer_ids)

/// Where each member's rows of features go: dst[m] + (row - start[m]) * n_feat * n_embd + slot * n_embd, bf16.
struct FeatureDst {
    uint16_t* p[kMaxMembers] = {};
};
/// Feature `slot` of every row of `x` ([rows][n_embd], fp32), rounded to bf16 (round to nearest even).
void features_store(const float* x, const Members& m, const FeatureDst& dst, int n_feat, int slot, int n_embd, void* stream);

/// The mixes of `rows` rows of features ([rows][n_feat][n_embd] bf16): out[j][row] = rmsnorm(sum_f alpha[j][f] feat[row][f])
/// * w[j] (fp32 sums, eps) and their q8_1 blocks xq[j][row] (null: none), for up to kMaxFeatures mixes in one launch.
struct Mixes {
    int n = 0;
    float alpha[kMaxFeatures][kMaxFeatures] = {};
    const float* w[kMaxFeatures] = {};
};
void feature_mix(const uint16_t* feat, int rows, int n_feat, const Mixes& mixes, int n_embd, float eps, float* out, void* xq,
                 void* stream);

/// Per row and K/V head: k = rope(rmsnorm(k) * k_w) at pos[row] (neox, theta, head 128, fp32 angles as vLLM's cache) and v as
/// it is, both to fp16 cells (pos % cells) of [cells][n_kv][128]. With `q`, each Q head the same in place (q_w).
struct QkArgs {
    int n_head = 0, n_kv = 0;
    float theta = 1e6f, eps = 1e-6f;
};
void qk_norm_rope_store(float* q, const float* q_w, const float* k, const float* k_w, const float* v, const QkArgs& a,
                        const int64_t* pos, int rows, uint16_t* k_cache, uint16_t* v_cache, int64_t cells, void* stream);

/// res[row] += add[row] (add null: res as it is), then out[row] = rmsnorm(res[row]) * w (out null: not written) and its
/// q8_1 blocks xq (null: none).
void add_norm_quant(float* res, const float* add, const float* w, int rows, int n_embd, float eps, float* out, void* xq, void* stream);

/// h = silu(gate) * up ([rows][n]) as q8_1 blocks.
void swiglu_quant(const float* gate, const float* up, int rows, int n, void* xq, void* stream);

/// rows 1 .. rows-1 of x ([rows][n_embd]) = mask (row 0 as it is).
void fill_mask_rows(float* x, const float* mask, int rows, int n_embd, void* stream);

/// The Markov head's per-step context: out[k] = W[:, 2 rank:] [h[k] ; f] + b ([n][3 rank]), W [3 rank][2 rank + 2 n_embd] fp32,
/// h rows `ld_h` apart.
void markov_context(const float* W, const float* b, const float* h, int ld_h, const float* f, int n, int rank, int n_embd, float* out,
                    void* stream);

/// One step: raw = ctx + W[:, :2 rank] [state ; emb] (raw: 3 rank floats of scratch); gate, cand, out = split(raw);
/// state = sigmoid(gate) state + (1 - sigmoid(gate)) tanh(cand) in place; latent = tanh(out). `emb` is the previous token's
/// row of w1 (markov_embed, or the last markov_pick), on the card.
void markov_step(const float* ctx, const float* W, const float* emb, int rank, int n_embd, float* raw, float* state, float* latent,
                 void* stream);
/// emb = w1[*token] (w1 [n_vocab][rank] bf16, which may be mapped host memory: one row crosses the bus) as fp32.
void markov_embed(const uint16_t* w1, const int32_t* token, int rank, float* emb, void* stream);

/// The draft head's screen (as Hy4-flash's drafter screens its head): basalt/kernels/head_proxy.cuh's 2-bit copy of a Q8_0
/// head of n_rows rows of 2304, its bytes and its packing; then for up to kScreenMaxRows q8_1 rows x ([n_act][72 blocks])
/// the copy's approximate logit of every head row, out[a * ld + row].
constexpr int kScreenMaxRows = 8;
size_t head_screen_bytes(int64_t n_rows);
void head_screen_pack(const void* head_q8_0, int64_t n_rows, void* screen, void* stream);
void head_screen_scores(const void* screen, const void* x_q8_1, int n_act, int64_t n_rows, float* out, int ld, void* stream);

/// The step's draft: the capped logit cap tanh((base[v] + fold[v] . latent + fold_b[v]) / cap) over v < n_vocab (fold
/// [n_vocab][256] bf16, a Markov rank of 256, 16-byte aligned rows), with `base` the screen's approximate logits; the rows
/// within 5 of the best total are rescored with the exact base (the Q8_0 head's row against the step's q8_1 row x_q8_1, 2304
/// values), and the best exact one is the draft (the first of equals): token, and prob its probability against the
/// approximate softmax with the rescored rows' terms exact. The token also goes to *prev and its w1 row to emb (null: not), for the next step.
size_t markov_pick_scratch_floats(int n_vocab);
void markov_pick(const float* base, const void* head_q8_0, const void* x_q8_1, const uint16_t* fold, const float* fold_b, const float* latent,
                 int n_vocab, float cap, float* scratch, const uint16_t* w1, int32_t* token, float* prob, int32_t* prev, float* emb,
                 void* stream);

}  // namespace basalt::models::inkling2x::kernels
