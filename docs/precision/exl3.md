# EXL3 packs

A pack converted from an ExLlamaV3 EXL3 checkpoint (`convert.py --exl3`). The routed experts run
in their EXL3 form; the format is in
[include/basalt/kernels/cpu/exl3_cpu.hpp](../../include/basalt/kernels/cpu/exl3_cpu.hpp) and, in numpy,
[tools/convert/exl3.py](../../tools/convert/exl3.py).

## What is stored

Per expert, the checkpoint's own bits: the three mul1 trellises (tiles reordered output-major, words unchanged) and
the six fp16 `suh` / `svh` vectors. Nothing is requantized. A matrix means `W = diag(suh) H inner H diag(svh)`, `H`
the normalized 128-wide Hadamard matrix per block of 128; `inner` holds the codebook value
`fp16(fma(1024 + S, 0x1eee, 0xc931))` of each trellis state, `S` a byte sum.

`tools/convert/exl3_upstream_check.py` compares the decoder with ExLlamaV3 at commit `bd1f5191` on a checkpoint: the
inner matrix is bit-identical to the extension's `reconstruct` for every bitrate the Flash-Next quants use (1.5, 2, 3,
4 and 5 bits per weight).

## How each path computes it

| Path | Weights | Activations | Accumulation |
|---|---|---|---|
| GPU decode (`exl3_kernels.cu`) | the fp16 codebook values, exact | transformed per expert and matrix in fp32, then a power-of-two scale per 128 inputs and two fp16 values, `hi + lo` (about 22 bits) | tensor-core `mma` (m16n8k16, fp32), each 128-input block's `hi` and `lo` sums apart, scaled back per block; a fixed order per entry |
| CPU (`exl3_cpu.cpp`, `exl3_avx512.cpp`, `exl3_avx2.cpp`) | the codebook's affine form before its fp16 rounding (within 2^-10 of the fp16 value) | `H (suh * x)` as int16 with one scale per 128 inputs | exact int32 per 128-input block, then one fp32 `fma` chain |
| Prompt path (`exl3_dequant_*_f16`) | `W` in the original basis, computed in fp32 and rounded once to fp16 | the fp16 GEMM's | the fp16 GEMM's |

A token's result does not depend on the other tokens of a call on either decode path, and the CPU kernels give the
same bits on every instruction set and for any split of the rows.

## Measured error

`build/exl3_expert_parity`, random blobs at 2560 x 640, every bitrate from 1 to 8 bits per weight, against the same
expert computed in double from the fp16 codebook values. Error is the L1 ratio `sum |got - ref| / sum |ref|` of the
expert's output.

| Path | Worst over the bitrates |
|---|---|
| GPU decode | 6.9e-7 |
| CPU | 4.0e-4 |
| Prompt path's fp16 weights | within one fp16 rounding of `W` in double (worst 0.58 ulp) |

On three experts of the 2.05 bpw Flash-Next quant, the CPU path is within 9.2e-4 of ExLlamaV3's reconstructed weights
applied in double; ExLlamaV3's own kernels on fp16 inputs are within 1.2e-3 of the same.

## The other tensors

`convert.py --exl3` computes every other language-model tensor from the checkpoint once
([tools/convert/exl3_checkpoint.py](../../tools/convert/exl3_checkpoint.py)); the names, order and transforms are the
NVFP4 conversion's.

| Tensor class | In the checkpoint | In the pack |
|---|---|---|
| Dense projections, shared experts | EXL3, 4 or 5 bits per weight | `W` computed in double, stored as Q6_K where rows are whole 256-value blocks and as Q8_0 elsewhere (one more rounding); all Q8_0 with `--dense-q8-0` |
| Output head | EXL3, 4 bits per weight | `W` computed in double, stored as Q8_0 |
| Token embedding | BF16, or EXL3's row codec at 4 bits per weight (rings of 256 values, a scale each) | Q8_0 |
| Indexer q/k projections | EXL3, 2 or 5 bits per weight | `W` rounded to BF16 |
| Router, hyper-connection matrices, GDN alpha/beta, PLE projections | fp16 | BF16 (the values that are not BF16 already are rounded; `meta.exl3.dense.rounded_to_bf16` counts them) |
| Norms, GDN state tensors, conv weights | fp16 or BF16 | F32 or F16, exact |
| N-gram table | `ngram_embedding.safetensors`: rows of an fp16 scale and a ring of 160 mul1 steps at 3 or 4 bits per value, and a bias per hash head | the rows byte for byte (type `EXL3R<K>`, 2 + 20 K bytes each) and the fp16 biases; the base pack's table when the file is absent |
| Drafter, vision | EXL3's own codecs | not converted: the base pack's sections |

The row codec of the token embedding is checked against ExLlamaV3's `ngram_codec.dequant_rows` by
`exl3_upstream_check.py` (8e-8 on the 1.66 bpw quant's embedding).

Q6_K is the default for the projections because it trades little fidelity for VRAM: on the 1.66 bpw quant the 252
Q6_K tensors are 0.64 GiB smaller than as Q8_0, about 700 more expert slots on a 16 GB card, and differ from the Q8_0
values by 2.2e-2 in relative L2 (median of 21 tensors, worst 2.4e-2). Against the BF16 base that costs 0.0024 KLD
with the same top-1 ([below](#quality-against-bf16)). Q5_K was measured too: 0.35 GiB less again, for 0.008 more
KLD than Q6_K and no faster decode.

## The n-gram table

The engine decodes an n-gram row on the CPU as `fp16 codebook value * scale + bias` in fp32 (`exl3_ring_dequant_row`
in `models/qwen4exp/kernels/ngram.cpp`), without ExLlamaV3's final rounding of the row to fp16. `exl3_expert_parity` holds it to
`exl3.ngram_rows` bit for bit at every K, and `exl3_upstream_check.py` holds that to `ngram_codec.dequant_rows` (no
difference over 1024 rows of the 1.66 bpw quant's table) and the engine's read of a pack's rows to both
(`--ngram-io`: bit-identical on the 1.66 and 2.05 bpw packs).

## Quality against BF16

The [BF16 gate](quality.md#the-bf16-reference), measured on one machine: an RTX 5080 (16 GB) and 32 GB of RAM, no
peer card.

- **Base:** unsloth's BF16 GGUF through `llama-perplexity` at commit 436f6f89e, 100 chunks of 512, `-b 8192`,
  dense layers on the GPU (`-ngl 99`) and the experts on the CPU from the mapped files (`-ot exps=CPU`). One
  difference from the documented run: `-ub 4096` instead of 8192, because the 8192-token compute buffer (7.8 GiB) does
  not fit next to the dense layers on a 16 GB card. The micro-batch changes only how the batch is split. The result
  is PPL 3.2787 ± 0.043 on all rows and 3.2774 on the scored rows, against the documented 3.2749 and 3.2733.
- **Gate:** `tools/ppl_gate.py` with the harness's own settings (`--spec 5`, adaptive swaps off), plus `--t 3
  --kv int8 --resident-experts`.

| Pack | Experts | KLD mean | KLD median | KLD p99 | Same top-1 | PPL |
|---|---:|---:|---:|---:|---:|---:|
| EXL3 2.05 bpw, Q8_0 dense \* | ~28 GiB | 0.2509 | 0.0615 | 2.711 | 83.93% | 3.5536 |
| Gyro-S (TQK6/TQK7, Hadamard-folded), for comparison | 24.0 GiB | 0.3980 | 0.1063 | 4.022 | 79.53% | 3.8365 |
| EXL3 1.66 bpw, Q8_0 dense \* | 22.7 GiB | 0.4563 | 0.1385 | 4.309 | 77.41% | 4.0147 |
| EXL3 1.66 bpw, Q6_K dense | 22.7 GiB | 0.4578 | 0.1390 | 4.321 | 77.37% | 4.0172 |

\* Measured with GPU decode kernels that multiplied in fp32 instead of on the tensor cores (9.2e-7 against the
double reference instead of 6.9e-7). On the Q6_K-dense pack the two differ by 0.0008 KLD (0.4586 against 0.4578), so
the Q6_K cost above is the like-for-like 0.4586 against 0.4563.

How to read it:

- **The scale matches the official table.** On the [Hugging Face card](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt#quality),
  UD-IQ1_M (32.4 GiB of experts) has KLD 0.2560 and top-1 83.19%. The 2.05 bpw EXL3 pack reaches the same KLD with
  about 4 GiB less.
- **EXL3 and Gyro are close in quality per byte.** Gyro-S is ahead of EXL3 1.66 (0.398 against 0.456), but it is
  also 1.3 GiB larger. A line through the two EXL3 points puts EXL3 at about 0.41 at Gyro-S's size.
- **The draft head and vision do not affect these numbers.** The packs take their drafter and vision sections from the
  Gyro-S pack, and the gate scores the main model's logits only.
