# Experiment 2 — Controlled concept generalization

Experiment 2 asks whether preference fitting becomes reliable held-out control and whether that control survives changes in syntax.

> **Status: the full multi-seed run is complete** (Colab CUDA GPU, fp32, 2026-10-20). It used 20 training pairs, 36 held-out prompts, 5 genuine seeds and 3 shuffled-label seeds, and its outputs are in `results/experiment_2/full/`. The earlier quick run (5 pairs, 10 prompts, 1 seed) is kept below for comparison. The full run reused the committed screening outputs and verified the frozen split. It regenerated the base condition in the same runtime as every other condition, and those base outputs matched the CPU screening word for word on all 36 prompts.

## Design changes

### Empirical concise baselines

One hundred topic-disjoint candidate prompts were generated across at least ten domains. Base Qwen generated each prompt greedily with a 512-token allowance. Fifty-six prompts passed the preregistered screen: at most 120 words and EOS termination. This addressed Experiment 1's baseline-length confound.

A deterministic split (`seed=20260816`) froze 20 training topics and 36 held-out topics. Nested training subsets contain 1, 2, 5, 10, and 20 examples.

### Contrast-purified preferences

Rejected answers contain 50–100 words. Chosen answers contain 350–500 words and preserve the whitespace-normalized rejected answer as an exact prefix. Each pair has at least a 250-word gap. This makes “continue/elaborate” the dominant contrast rather than allowing unrelated openings, tone, or framing to vary.

Training uses one syntax throughout:

```text
<ENSURE_LONG> your response to the following request:

{concise prompt}
```

The prompt never defines the token's intended meaning.

## Controls

- **Base:** concise prompt without the new token.
- **Untrained:** training syntax with the initial random vector.
- **Trained:** training syntax with the genuine preference-trained vector.
- **Explicit English:** an ordinary instruction to expand substantially.
- **Shuffled:** same pairs, but labels independently flip when `random.Random(991).random() < 0.5`; at least one flip is forced. Quick mode flipped pairs 1 and 3 of 5.

## Full-run results

| Condition | Mean words | Δ vs base | Hit 512-token cap | Repeated-trigram fraction | Lexical diversity |
|---|---:|---:|---:|---:|---:|
| Base | 83.7 | — | 0% | 0.054 | 0.609 |
| Untrained (5 seeds) | 97.1 | +13.4 | 2% | 0.063 | 0.604 |
| Shuffled (3 seeds) | 125.6 | +41.9 | 3% | 0.100 | 0.538 |
| Explicit English | 215.6 | +131.9 | 17% | 0.147 | 0.460 |
| Trained (5 seeds) | 324.6 | +240.9 | 47% | 0.370 | 0.329 |

Per seed, as trained-minus-base words with prompt-bootstrap 95% intervals. Final DPO loss is in brackets; every run started at 0.693.

| Seed | Genuine | Shuffled |
|---:|---|---|
| 1 | +349.4 [+335.5, +362.6] (0.029) | +66.5 [+35.1, +103.6] (0.528) |
| 2 | +315.9 [+287.2, +342.6] (0.030) | +46.7 [+24.0, +72.0] (0.626) |
| 3 | +114.1 [+84.7, +147.7] (0.139) | +12.7 [+7.3, +18.4] (0.594) |
| 4 | +192.0 [+148.8, +235.3] (0.089) | — |
| 5 | +233.1 [+189.3, +276.0] (0.027) | — |

- **Seed robustness.** All 5 genuine seeds lengthened 97–100% of prompts, but effect sizes differ about 3×. The least-fitted seeds (3 and 4) are the weakest. The final vectors are nearly orthogonal (pairwise cosine 0.01–0.12), and all tokens leave token-free logits exactly unchanged (max difference 0).
- **Degeneration.** Length gains come with heavy repetition. Seed 1 hit the cap on 75% of prompts, with trigram repetition 0.53 and repeated-sentence fraction 0.33. Seed 3 hit the cap on 11%, with 0.13 and 0.03. The 1.83× explicit-effect ratio is therefore not a quality claim.
- **Learning curve** (3 seeds, mean Δ words): +11.6, +11.0, +33.5, +81.4, +190.0 and +259.8 at N = 0, 1, 2, 5, 10 and 20. It is not monotonic: N = 1 is no better than the untrained vector, and seed 2 dips at N = 1 and 2. Preference margin and behavior correlate at r = 0.95 over the 18 runs, which are dependent points.
- **Position** (5 seeds × 20 prompts; base 84.5 words on these prompts): Δ +239.7 at the prefix, +108.8 mid-verb, +35.0 in the alternate form and +25.5 as a suffix. Single-prompt compositional probes averaged 88–156 words (base 79).

### Quality-adjusted length (`scripts/analyze_quality.py`, outputs in `quality/`)

Effective words = distinct word trigrams + 2. Clean = ended with EOS and repeated-trigram fraction ≤ 0.171, the most repetitive base answer. Effective Δ vs base: untrained +6.0, shuffled +27.8, explicit English +91.4, neologism +94.9 (seeds +117.8, +112.4, +80.5, +73.2, +90.8). Clean answers: 96%, 81%, 69% and 37% respectively (per seed 14%, 14%, 75%, 47%, 36%). LoRA 1,024 gives +65.0 effective (55% clean), and LoRA r8 gives +79.5 (3% clean); the plain-prompt numbers are about the same. On effective length, the neologism matches the explicit instruction (1.04×) but with roughly half the clean-answer rate. The metric measures non-repetition, not accuracy.

### LoRA baseline (full mode, 3 seeds per arm)

| Method | Params | Δ words, token template | Δ words, plain prompt | Hit cap | Trigram repetition | Final loss |
|---|---:|---:|---:|---:|---:|---:|
| Neologism | 896 | +240.9 | 0 (by construction) | 47% | 0.37 | 0.03–0.14 |
| `lora_matched` (r1, layer-0 `v_proj`) | 1,024 | +163.2 | +155.6 | 28% | 0.28 | 0.13–0.18 |
| `lora_r8_qv` (r8, q/v, all layers) | 540,672 | +348.8 | +345.2 | 93% | 0.61 | ≈0.02 |

LoRA's effect on plain prompts is as large as with the token template, so it changes the model's default behavior. The neologism's effect is conditional on the token. Learning rates were fixed, not tuned.

### Interpretability (5 seeds × 36 prompts)

- The trained token lowers the teacher-forced log p(`<|im_end|>`) after the base answer by 11.2 nats on average, against the untrained token, and lowers it on 100% of seed × prompt pairs.
- **Activation patching** at the token position restores 88% of that effect at layer 0, 49% at layer 2, 16% at layer 5 and ≤3% from layer 8. The information leaves the token position early. The exact layer varies by seed, from ~3 to ~11.
- **Logit lens** at the stop-decision position: the trained-minus-no-token difference stays near zero until the last layers, then drops to −0.9, −3.6 and −11.3 nats at layers 21, 22 and 23. The mid-layer logit lens is noisy in a model this small.
- **Neighbours:** the nearest vocabulary rows to E_w and to E_w − E_w⁽⁰⁾ are unrelated multilingual subwords at cosine ≤ 0.16. No length-related words appear.

## Quick-run results

| Condition | Mean words | Difference from base |
|---|---:|---:|
| Base | 86.0 | — |
| Untrained | 94.9 | +8.9 |
| Shuffled | 93.1 | +7.1 |
| Trained | 159.2 | +73.2 |
| Explicit English | 160.4 | +74.4 |

The genuine token had positive trained-minus-base differences on 8/10 prompts, mean `+73.2`, median `+60.5`, standard deviation `70.68`, range `[-17, +193]`, and prompt-bootstrap 95% interval `[+35.5, +117.8]`. Three prompts exceeded +100 words and none exceeded +200.

Preference fitting was strong: genuine loss fell `0.693147 → 0.123739`, with final mean margin `21.2142`. Shuffled loss fell `0.693147 → 0.522353`, with margin `4.0289`. This separation supports structured preference information as the source of the quick run's intended effect, while remaining only a one-seed result.

### Data-size probe

| Training pairs | Mean held-out words | Mean Δ vs base | P(Δ>0) | Mean preference margin |
|---:|---:|---:|---:|---:|
| 0 | 94.9 | +8.9 | 0.8 | 0.000 |
| 1 | 99.5 | +13.5 | 0.7 | 9.943 |
| 2 | 126.3 | +40.3 | 0.9 | 11.450 |
| 5 | 159.2 | +73.2 | 0.8 | 21.214 |

This single-seed quick curve was monotonic in mean length. The margin/behavior correlation across its four points was `0.909`; four dependent points are far too few to infer a robust relationship.

### Behavior beyond length

Compared with base, trained outputs had more paragraphs (`1.7` vs `1.0`), sentences (`6.6` vs `3.6`), and heuristic elaboration indicators (`2.6` vs `1.7`). Quality did not uniformly improve: repeated-bigram fraction rose from `0.122` to `0.207`, repeated-trigram fraction from `0.065` to `0.128`, and type/token lexical diversity fell from `0.618` to `0.505`. These metrics are length-sensitive and heuristic, but they expose repetition as an important failure mode.

### Position and composition

Prefix, suffix, mid-verb, and alternate placements averaged 141.3, 72.3, 330.7, and 94.0 words respectively. Exploratory compositional probes ranged from 89 to 412 words. Some verb-like frames strongly amplified continuation, but effects were unstable and sometimes degenerate. Current evidence supports context-dependent control, not syntax-invariant lexical meaning.

## Claim assessment (full run)

- Preference fitting: **Strong** (5/5 seeds).
- Behavioral control: **Strong** in raw length. After removing repeats it matches the explicit English instruction (+94.9 vs +91.4 effective words), with fewer clean answers (37% vs 69%).
- Held-out topic transfer: **Strong** (97–100% of held-out prompts per seed).
- Seed robustness: **Partial**. The direction is robust, the size varies about 3×, and the vectors are near-orthogonal.
- Token specificity vs LoRA: **Strong**. LoRA leaks to plain prompts.
- Word-like compositional behavior: **Absent**.
- Scale, family and richer-concept robustness: **Unknown**.
