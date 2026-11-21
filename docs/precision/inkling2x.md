# Inkling 2 Mini precision

Where the Inkling 2 Mini packs round and how their output compares with BF16. Method: [quality](quality.md). Packs:
[model file](../model-file.md#inkling-2-mini-test-pack). All routed experts on one GPU, fp16 KV, no drafter.

Test setup: RTX 5080 (16 GB) at 360 W.

## Against the BF16 reference

Wikitext-2, 100 chunks of 512, rows 256 to 510 of each chunk (25,500 rows), one-token decode windows (`--quality-t 1`):

| Pack | Size | PPL | ln(PPL/base) | KLD mean | KLD median | KLD p99 | KLD max | Same top-1 |
|---|---|---|---|---|---|---|---|---|
| SAGE REXL3 2.92 bpw | 10.0 GiB | 7.2173 ± 0.1112 | 0.0214 | 0.0297 | 0.0125 | 0.295 | 2.62 | 92.88% |
| InterpreticAI IQ3_XXS | 10.9 GiB | 7.4014 ± 0.1133 | 0.0465 | 0.0658 | 0.0368 | 0.480 | 9.66 | 88.79% |

Base PPL 7.0649 on the scored rows. The REXL3 pack is 0.9 GiB smaller and has less than half the mean KLD. The two packs
split their bytes differently: REXL3 has 7.9 GiB of experts and 2.1 GiB of other tensors, IQ3_XXS 10.1 GiB of experts and far
coarser attention projections ([stages](#where-each-stage-rounds)).

The [SAGE card](https://huggingface.co/vcruz305/Inkling-2-Mini-SAGE-REXL3-2.92bpw) reports a mean KLD of 0.0196 and 96.33% top-1
agreement against the official checkpoint in float64. It does not name its text, context length or scored positions, so its
figures and this gate's are not comparable. This gate's base is llama.cpp's BF16 conversion of that checkpoint.

The IQ3_XXS pack keeps the GGUF's bytes. Against llama.cpp on the same weights it scores KLD 0.0033 and 97.80% top-1, and its
greedy output matches llama.cpp's for 60 tokens.

## Where each stage rounds

| Stage | REXL3 pack | IQ3_XXS pack | Arithmetic |
|---|---|---|---|
| Routed experts | EXL3 mul1, 2 to 4 bpw per expert (gate and up share a rate) | IQ3_XXS gate/up, IQ4_NL down | EXL3: fp32 activations through the Hadamard transforms; IQ: q8_1 activations; fp32 accumulation |
| Attention projections, gate | EXL3 5 bpw, computed in double and stored once as Q8_0 | IQ2_S (Q, K), IQ3_XXS (V, R, gate), IQ3_S (output) | q8_1 activations, fp32 accumulation |
| Sinks, convolution input taps | Q8_0 of the BF16 values | IQ3_XXS | q8_1 activations |
| Output head | EXL3 6 bpw, computed in double and stored once as Q8_0 | Q5_K | q8_1 activations, fp32 logits, soft cap 30 |
| Token embedding | BF16 | IQ3_S | fp32 rows |
| Norms, router, convolutions, loop controllers, depth-mix queries | F32 | F32 | fp32 |
| K and V | fp16 cache | fp16 cache | fp32 scores, softmax and accumulation |

All activations between stages are fp32. A row's output does not depend on which rows share its launch (bitwise).

## REXL3 conversion error

In the REXL3 pack, all 478 F32 tensors and the BF16 token embedding match the corresponding tensors of the official checkpoint's
BF16 GGUF exactly. These are weight errors, not output KLD:

- **144 EXL3 projections and the head, stored as Q8_0:** full-matrix relative L2 error against BF16 from 1.94% (the head) to
  5.23% (the last layer's attention gate).
- **32 projections from BF16 straight to Q8_0** (sinks, convolution input taps): 0.51% to 0.97%.

## Kernel parity

Against a double-precision reference at model geometry, the grouped EXL3 expert call over two rate pairs with situ has a maximum
relative error of 5.94e-7 (0.0218 absolute on outputs of order 10^4). Its output equals separate uniform-rate calls bit for bit,
including each row run alone. This tests arithmetic on the same quantized weights, not quantization loss.

## Reproduce the decode gate

BF16 reference: llama.cpp `eb550aaa5`, `convert_hf_to_gguf.py --outtype bf16`. `llama-perplexity` reports 7.0729 ± 0.1090 over
all of each chunk's second half. To make the base with the experts in system RAM and score a pack:

```sh
llama-perplexity -m Inkling-2-Mini-BF16.gguf -ngl 99 -ot 'ffn_.*_exps=CPU' --no-op-offload -fa on \
    -f wiki.test.raw -c 512 --chunks 100 -b 2048 -ub 512 --kl-divergence-base inkling2x-bf16-base.kld
basalt-bench --model PACK.basalt --quality 100 --quality-t 1 --kld-base inkling2x-bf16-base.kld --max-context 1024
```

The base file is 10.2 GB. The run takes about 90 minutes on this machine; op offload needs a 24.7 GB compute buffer, hence
`--no-op-offload`.
