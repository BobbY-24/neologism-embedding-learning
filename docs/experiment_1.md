# Experiment 1 — Minimal proof of concept

Experiment 1 was a three-pair pilot designed to validate the mechanics of optimizing one token embedding while freezing an entire pretrained language model.

## Setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- New token: `<ENSURE_LONG>` (tokenizer size 151,665 → 151,666; ID 151,665)
- Frozen parameters: 494,032,768
- Trainable parameters: one 896-dimensional vector
- Training pairs: 3
- Objective: manual response-only DPO, `β=0.1`
- Optimizer: AdamW, learning rate `0.01`, 3 epochs
- Generation: deterministic greedy decoding, up to 512 new tokens

The wrapper leaves the pretrained input embedding table untouched. It maps the out-of-table new token ID to a safe frozen row, computes ordinary embeddings, and replaces only masked positions with `neo_embedding`. The optimizer receives only that standalone parameter.

## Preference fitting worked

Mean DPO loss fell from `0.693147` to `0.255095`; the final mean preference margin was `12.5185`. The learned vector moved `1.0044` in L2 distance from initialization and retained cosine similarity `0.4721` with its initial value.

All pretrained parameter checksums were identical after training. A prompt without the new token had maximum absolute before/after logit difference `0.0`.

## Behavior was weak and mixed

| Held-out prompt | Base words | Untrained words | Trained words |
|---|---:|---:|---:|
| Sky blue | 308 | 107 | 253 |
| Airplane lift | 329 | 399 | 334 |
| Exercise | 356 | 270 | 377 |

The trained token increased length over base on two of three prompts, but only modestly on those two; it was also shorter than the untrained token for airplane lift. Position changes produced 364 words at prefix, 253 at suffix, and 350 in the middle.

## What the pilot revealed

The chosen responses were about 270–300 words, yet base Qwen already generated roughly 300+ words on these prompts. That eliminated much of the intended short-versus-long contrast. Chosen and rejected answers were also independently written, so length was confounded with wording, tone, and content.

Experiment 1 therefore validated the implementation and revealed design confounds. It motivated Experiment 2's empirical concise-baseline screening and exact-prefix preference construction. It did **not** establish a robust lexical concept.

