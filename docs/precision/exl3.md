# EXL3 packs

Routed experts in ExLlamaV3's EXL3 format, run in that form; the format is in
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

## The n-gram table

The engine decodes an n-gram row on the CPU as `fp16 codebook value * scale + bias` in fp32 (`exl3_ring_dequant_row`
in `models/qwen4exp/kernels/ngram.cpp`), without ExLlamaV3's final rounding of the row to fp16. `exl3_expert_parity` holds it to
`exl3.ngram_rows` bit for bit at every K, and `exl3_upstream_check.py` holds that to `ngram_codec.dequant_rows` (no
difference over 1024 rows of the 1.66 bpw quant's table) and the engine's read of a pack's rows to both
(`--ngram-io`: bit-identical on the 1.66 and 2.05 bpw packs).
