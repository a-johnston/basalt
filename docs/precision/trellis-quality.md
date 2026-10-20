# APR trellis pack (Gyro-L): Q8_0-reference measurements

These are the source branch's development measurements, not validation of the rebased build and not the
[BF16 quality gate](quality.md#the-bf16-reference). GPU power limit: RTX 5060 Ti at 180 W, no peer GPU, 62 GB RAM.

Two 20-chunk bases written by `llama-perplexity --kl-divergence-base` from the Q8_0 GGUF (wikitext-2 and technical
prose/code), scored with `tools/ppl_gate.py --chunks 20 --t 3 --expert-cache auto`:

| Pack | Text | Engine | PPL | KLD mean | KLD median | KLD p99 | Same top-1 |
|---|---|---|---|---|---|---|---|
| Gyro-L TQ3_0 | wikitext-2 | Basalt | 2.3646 | 0.13066 | 0.01121 | 1.857 | 91.00% |
| GSQ-RCO IQ3_S | wikitext-2 | Basalt | 2.3893 | 0.18149 | 0.01634 | 2.289 | 88.67% |
| Gyro-L TQ3_0 | wikitext-2 | llama.cpp (agention fork) | – | 0.1247 | – | – | 91.28% |
| GSQ-RCO IQ3_S | wikitext-2 | llama.cpp (agention fork) | – | 0.1808 | – | – | 88.73% |
| Gyro-L TQ3_0 | technical + code | Basalt | 2.3629 | 0.08371 | 0.00216 | 0.914 | 92.76% |
| GSQ-RCO IQ3_S | technical + code | Basalt | 2.3913 | 0.11378 | 0.00297 | 1.214 | 91.10% |
| Gyro-L TQ3_0 | technical + code | llama.cpp (agention fork) | – | 0.0763 | – | – | 92.78% |
| GSQ-RCO IQ3_S | technical + code | llama.cpp (agention fork) | – | 0.1119 | – | – | 90.94% |

Paired on the same tokens, Gyro-L's KLD is lower by 0.0508 on wikitext-2 (chunk t -6.2, lower in 19 of 20 chunks)
and by 0.0301 on technical text (chunk t -5.3, 18 of 20). IQ3_S agrees between engines to 0.002 KLD.
Gyro-L's roughly 0.006 gap is affected by the Q8_0 reference's own activation rounding: llama.cpp quantizes the
hc read's activations as the reference does, while Basalt uses fp32 activations and BF16 copies of the Q8_0 hc
weights (`--compat-bf16`, at most 2^-9 relative). These results cannot be compared directly with BF16-base results.
