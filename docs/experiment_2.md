# Experiment 2 — Controlled concept generalization

Experiment 2 asks whether preference fitting becomes reliable held-out control and whether that control survives changes in syntax.

> **FULL MULTI-SEED EXPERIMENT: PREPARED, NOT YET EXECUTED.** The 20-pair training set and 36-prompt held-out set are frozen. Reported behavioral results below are from quick mode only: 5 training pairs, 10 held-out prompts, one genuine seed, and one shuffled seed.
>
> The full run is implemented and runs on a Colab GPU via `notebooks/colab_runner.ipynb`. It reuses the committed screening outputs, verifies the frozen split, and regenerates the base condition in the same runtime as every other condition. It has not been executed yet.

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

## Claim assessment

- Preference fitting: **Strong**.
- Behavioral control: **Moderate**.
- Held-out topic transfer: **Moderate, preliminary**.
- Word-like compositional behavior: **Absent / weak**.
- Seed, scale, family, and richer-concept robustness: **Unknown**.

