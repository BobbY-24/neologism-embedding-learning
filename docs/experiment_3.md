# Experiment 3: within-family scale replication (RQ6)

**Status: 1.5B complete (Colab CUDA GPU, float32, 2026-10-20); 3B not run.** Outputs are in `results/experiment_3/full/`. The 0.5B reference is the full Experiment 2 run.

## 1.5B results

Same frozen 20 pairs and 36 held-out prompts, the same hyperparameters (lr 0.01, β 0.1, 3 epochs) and the same 5 genuine + 3 shuffled seeds. The trainable vector has 1,536 dimensions. Peak CUDA memory during training was 10.1 GB. Base 1.5B answers are shorter (70.0 words against 83.7), and 92% of them would still pass the 0.5B screen (≤ 120 words, ended by EOS), so the reused data are close to the model's own behavior.

| | 0.5B | 1.5B |
|---|---:|---:|
| Trained Δ words vs base (mean of 5 seeds) | +240.9 | +117.7 |
| Seed range | +114 to +349 | +1.8 to +212 |
| Prompts lengthened | 99% | 85% |
| Shuffled-label Δ | +41.9 | −5.0 |
| Explicit English Δ | +131.9 | +180.4 |
| Effective Δ words, trained (repeats removed) | +94.9 | +39.4 |
| Effective Δ words, explicit English | +91.4 | +155.7 |
| Trained answers at the 512-token cap | 47% | 16% |
| Stop log-prob effect (trained − untrained) | −11.2 nats | −11.2 nats |

**Two of five seeds barely trained.** Seeds 1, 2 and 5 reached a final DPO loss of 0.007–0.039 (margin 36–51) and lengthened answers by +167, +185 and +212 words. Seeds 3 and 4 stopped at a loss of 0.105 and 0.403 (margin 23.4 and 7.4) and moved length by only +22.6 and +1.8. Behavior tracks the margin, which fits the 0.5B-tuned learning rate and epoch count being too weak for some 1.5B initializations. The size comparison is therefore confounded by under-fitting and is not a clean scale effect. The learning curve (seeds 1–3) points the same way. The effect stayed near zero up to N = 2 for all three, became large only at N = 10 for seeds 1 and 2 (+214 and +151 words), and stayed near zero even at N = 10 for seed 3 (+6.4).

**The shuffled control is cleaner.** At 1.5B, shuffled labels produce no lengthening (−5.0 words, 100% clean), unlike +41.9 at 0.5B, so the genuine effect is entirely label-dependent at this size.

**Quality.** On the working seeds, trained answers are repetitive: trigram repetition 0.36–0.49, against 0.023 for base, and 14–42% of answers are clean. After removing repeats, the token adds +39.4 effective words (working seeds +51 to +63), against +155.7 for explicit English (0.25×; at 0.5B it was 1.04×). The larger model follows the plain-English instruction much better than it uses the learned token.

**Position.** On 20 prompts (base 74.1 words), the effect is +106.0 at the prefix, +46.0 mid-verb, +67.2 in the alternate form and +20.2 as a suffix. Two working seeds kept much of their effect in the suffix and alternate placements (+141 and +97 words; seed 1: −4.9). At 0.5B the alternate form kept only 15% of the prefix effect. This is a possible sign of less position-bound behavior at 1.5B, but it rests on two seeds.

**Where the token acts.** Patching at the token position gives a different picture from 0.5B. In seeds 2, 3 and 5, restoring the token-position residual keeps 71–99% of the effect through layer 15 (seed 3 through layer 24), before collapsing. At 0.5B the effect had left the token position by about layer 8. Seed 1, the largest effect (−20.4 nats), restores ≈ 0 from layer 0 on, so in that seed the first layer's attention reads the vector directly. Pooled over seeds, restoration is 0.59 at layer 0, 0.55 at layer 15 and 0.03 by layer 18. The learned vectors are again near-orthogonal across seeds (|cosine| ≤ 0.11), and their nearest vocabulary rows are unrelated (cosine ≤ 0.18). Token-free logits are unchanged in every run (max difference 0).

**Conclusion so far.** The gated effect replicates at 1.5B for the seeds that train, the shuffled control becomes clean, and the token keeps its information at its own position much deeper into the network. The effect is smaller, less reliable across seeds and, after removing repeats, well below plain English. 3B has not been run: with the unchanged recipe, under-fitting seeds would make its result hard to interpret.

## Question

Does the Experiment 2 result hold as the frozen model grows from Qwen2.5-0.5B to 1.5B to 3B? The Experiment 2 result is a single learned input vector that lengthens held-out answers, is strongly position dependent, and does not behave like a word. The comparison covers the size of the behavioral effect, seed spread, the shuffled-label control, position sensitivity, the learning curve, repetition, and where in the network the token acts (logit lens and patching).

## What changes and what is fixed

Only the model and, for 3B, its dtype change. Everything else comes from `experiments/experiment_2/configs/default.json`: the 20 frozen pairs and 36 held-out prompts, the templates, DPO with β = 0.1, lr = 0.01, 3 epochs, 5 genuine and 3 shuffled seeds, greedy decoding up to 512 tokens, the position probes and the learning-curve sizes.

| Model | Hidden size = trainable parameters | Layers | Frozen parameters | dtype on an L4 |
|---|---:|---:|---:|---|
| Qwen2.5-0.5B-Instruct (reference: Experiment 2) | 896 | 24 | 494,032,768 | float32 |
| Qwen2.5-1.5B-Instruct | 1,536 | 28 | 1,543,714,304 | float32 |
| Qwen2.5-3B-Instruct | 2,048 | 36 | 3,085,938,688 | bfloat16 |

The hidden sizes and parameter counts come from the checkpoints' `config.json` and from loading each checkpoint once locally. The trainable vector always has the model's hidden size, so the parameter budget grows with the model. All three models share one tokenizer, so `<ENSURE_LONG>` gets the same id (151,665) at every size, and the random initialization uses each model's own embedding-table mean and standard deviation.

## Dataset carry-over: reuse the frozen 0.5B data (recommended)

There are two options.

1. **Reuse the frozen 0.5B pairs and prompts** (implemented). The data are held fixed, so model size is the only variable, and the 0.5B split check still applies. The cost is that the data are off-policy for the larger models. The rejected answers are 0.5B's concise outputs, and the chosen answers are 0.5B continuations of them, not each model's own. The held-out prompts were selected because 0.5B answered them in 120 words or fewer, and a larger model may already answer them at greater length, which leaves less room below the 512-token cap.
2. **Re-screen per model.** Each model screens the 100 candidates and writes its own pairs. This is on-policy and matches the original construction exactly. But the split changes with the model (different prompts pass the screen), so data and scale are confounded, and the screen plus pair construction cost several hundred extra generations per model.

The recommendation is option 1. RQ6 asks whether the same training signal produces the same kind of vector at a larger size, and only fixed data answers that without a second moving part. The off-policy cost is measured rather than assumed. `scale_comparison_<mode>.csv` reports `base_passes_screen`, the fraction of held-out prompts on which each model's own base answer would still pass the 0.5B screen (≤ 120 words, ended by EOS), together with `mean_base_words`. The main effect is still trained minus base, with base regenerated by the same model in the same runtime. A per-model re-screen is a possible follow-up. It is refused for now (`NEOLOGISM_RESCREEN=1` raises under `NEOLOGISM_MODEL`), because the screen writes into the committed `experiments/experiment_2/data/`.

One more consequence: the interpretability stop metric teacher-forces the 0.5B concise answer at every size. The text is identical across models, but it is not each model's own answer.

## Precision and memory

All of these are estimates, not measurements.

- **1.5B in float32.** The weights take about 6.2 GB. The backward pass to the input vector stores activations for every layer, because gradient checkpointing is inactive in eval mode. For a pair of roughly 750 tokens that is an estimated 3–4 GB, plus about 1.5 GB for the logits. Peak memory should be about 11 GB, comfortable on an L4 (24 GB), and the numerics match the 0.5B run.
- **3B in bfloat16.** Float32 would be about 12.3 GB of weights plus an estimated 5–7 GB of activations and logits, around 19 GB of the roughly 22 GB an L4 can use. That is too close to rely on. bfloat16 halves the weights. The trainable vector keeps a float32 master copy (`FrozenLMWithNeologism(trainable_dtype=torch.float32)`) and is cast down only where it enters the model, so AdamW updates are not rounded to bfloat16. This avoids the failure seen on MPS, where an fp16 vector diverged to inf.
- Smoke mode trains on a full-length pair, and scale runs record `peak_cuda_memory_gb` in `training_summary.json`. So the Colab smoke cell reports a measured peak before any long run starts.

bfloat16 at 3B is a confound. Two ways to remove or size it: run 3B in float32 on an A100 (`NEOLOGISM_DTYPE=float32`), or run the 0.5B model in bfloat16 as a cheap control (`NEOLOGISM_MODEL=qwen2.5-0.5b NEOLOGISM_DTYPE=bfloat16`). Greedy decoding can also flip under a different dtype, so a 0.5B bfloat16 run will not reproduce Experiment 2 exactly.

## Running

```bash
# One model, any machine with a checkpoint; outputs go to $NEOLOGISM_OUTPUT_ROOT/experiment_3/qwen2.5-1.5b-float32/
export NEOLOGISM_MODEL=qwen2.5-1.5b QWEN_MODEL_PATH=models/Qwen2.5-1.5B-Instruct NEOLOGISM_OUTPUT_ROOT=outputs
python scripts/run_experiment2.py --mode smoke --stage all
python scripts/run_experiment2.py --mode full --stage all
python scripts/run_interp.py --mode full
# Compare every finished size with the 0.5B reference in outputs/experiment_2 (no NEOLOGISM_MODEL needed)
python scripts/compare_scales.py --mode full
```

The registry is `experiments/experiment_3/configs/scale_models.json`. `NEOLOGISM_MODEL` unset runs the original Experiment 2 exactly as before: same config, same output tree and same cache signatures. Each model and dtype pair writes to its own output tree. The runtime refuses a checkpoint whose hidden size does not match the registry entry.

## Lessons from the earlier local attempt

An earlier attempt outside this repository targeted quick mode on an 8 GB MacBook M3. Both 1.5B and 3B loaded, but the first full-length DPO preference step did not finish within a 25-minute cutoff, with paging observed. Both were classed impractical locally, and no scientific result was produced. That attempt also fixed the data, kept the hyperparameters, used cache signatures keyed by model, and checked the hidden size, and those choices carry over here. This version runs the full multi-seed protocol on a Colab GPU instead of quick mode on a laptop.

## Known limitations

- The learning rate (0.01) was chosen at 0.5B and is not retuned per size. A weaker effect at 3B could reflect the optimizer setting rather than scale.
- The parameter budget grows with the hidden size (896 → 1,536 → 2,048). The comparison is "one vector per model", not "equal parameters".
- The LoRA baseline is not part of Experiment 3. Its code is size agnostic, but its matched arm was sized for 0.5B.
- There are as many generations per size as in the 0.5B full run (about 1,700), at batch size 1. The 1.5B float32 run fit in 10.1 GB peak on the Colab GPU.
- The 1.5B result is confounded by two under-fit seeds (see above); a per-size learning-rate or epoch check is needed before 3B.
