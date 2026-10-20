# Limitations

## Small-model testbed

All completed experiments use Qwen2.5-0.5B-Instruct. A 0.5B model may have a weaker representation space, weaker instruction following, more repetition, poorer syntax robustness, and unusually high sensitivity to embedding perturbations. Both directions are plausible: this setup could underestimate semantic richness available at larger scales, or overestimate steerability because smaller models are more perturbation-sensitive. Current effects must not be extrapolated to larger or newer language models.

## Single model family

Only Qwen has been tested. Architecture, tokenizer, chat template, pretraining distribution, and instruction tuning may all influence whether a new embedding can acquire reusable control behavior.

## Seed coverage

The full run has five genuine and three shuffled initializations. Effect sizes differ about 3× across seeds, so a single seed's prompt-level bootstrap interval understates the real uncertainty. Five seeds remain a small sample of initializations. Length is measured under a 512-token cap that truncates the largest effects, and word count overstates useful elaboration because long outputs are often repetitive.

## One development concept

Length/elaboration is an unusually measurable and potentially low-level behavior. It does not establish that a single vector can encode richer concepts, nuanced styles, factual abstractions, or latent machine-native concepts.

## Weak compositionality

Position and phrasing radically alter the effect. This is evidence against treating `<ENSURE_LONG>` as a stable word-like representation at this stage.

## Objective and decoding limitations

DPO preference fitting is not equivalent to intended behavior or semantics. Sequence-level summed log probabilities can reward pathologies; chosen and rejected likelihoods may move in unintuitive ways. Deterministic greedy decoding measures one behavioral realization and does not characterize the full output distribution.

## Prompt-set uncertainty

Held-out prompts are deliberately diverse but are not IID samples from a formally defined population. Bootstrap intervals summarize variation within the fixed prompt set only. The quick run has ten held-out prompts.

## Quality trade-offs

Longer outputs were often more repetitive and less lexically diverse. Word count alone is not evidence of better elaboration. Deterministic structural metrics help expose this issue but do not replace human evaluation.

## Nearest-neighbor interpretation

Cosine-nearest vocabulary embeddings are representational diagnostics, not definitions of the new token. Similar generation effects need not correspond to geometrically similar final vectors.

