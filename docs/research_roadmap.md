# Research roadmap

The program distinguishes increasingly demanding claims and treats external validity separately.

## RQ1 — Preference fitting

Can one embedding encode preference information while the model is frozen?

**Current evidence:** Strong within Qwen2.5-0.5B.

## RQ2 — Behavioral control

Does that fitted information change generation in the intended direction?

**Current evidence:** Moderate within-model evidence from one quick seed.

## RQ3 — Held-out topic transfer

Does the effect transfer to unseen prompts and topics in the same model?

**Current evidence:** Preliminary moderate evidence on ten quick-run prompts.

## RQ4 — Compositionality

Does behavior survive position and syntax changes without retraining?

**Current evidence:** Absent / weak; effects are highly context-dependent.

## RQ5 — Seed robustness

Do independent initializations fit the objective and induce comparable behavior? Do final embeddings converge geometrically?

**Current evidence:** Unknown. Next step: execute the prepared five-seed genuine and at least three-seed shuffled controls.

## RQ6 — Scale generalization

Does the phenomenon survive a controlled within-family progression?

```text
Qwen2.5-0.5B → Qwen ~1.5B → Qwen ~3B → larger models if resources permit
```

**Current evidence:** Unknown. Compare behavior strength, repetition, position sensitivity, optimization stability, and memory/runtime at every scale.

## RQ7 — Cross-family replication

Does the effect reproduce in Gemma, Llama-family, or other legally accessible checkpoints with matched protocols?

**Current evidence:** Unknown. Tokenization and chat-template differences must be treated as experimental variables, not implementation noise.

## RQ8 — Control vector versus word

What properties distinguish a soft behavioral control from a word-like lexical representation?

Candidate criteria include held-out topic transfer, syntax and position invariance, compatibility with grammatical operators, stable effects under paraphrase, nondegenerate quality, seed robustness, and reproducibility across model scales and families.

## Planned sequence

1. Complete Experiment 2 multi-seed runs on the 0.5B testbed.
2. Replicate within the Qwen family at approximately 1.5B and 3B.
3. Replicate across model families under matched data and decoding.
4. Move beyond length to richer concepts with purpose-built behavioral metrics.
5. Test whether learned controls meet explicit operational criteria for lexical composition.

