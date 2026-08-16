# Real Qwen neologism experiment

- Checkpoint: `Qwen/Qwen2.5-0.5B-Instruct`
- Model weights: external dependency (`Qwen/Qwen2.5-0.5B-Instruct`), not included
- Device: `cpu`
- MPS fallback: none
- Model / frozen parameters: 494032768 / 494032768
- Hidden size / trainable parameters: 896 / 896
- Tokenizer vocabulary: 151665 → 151666; `<ENSURE_LONG>` id 151665
- Hyperparameters: lr=0.01, epochs=3, beta=0.1, seed=42
- Decoding: greedy (`do_sample=False`), max_new_tokens=512
- Mean DPO loss: 0.693147182 → 0.255094637
- Final mean preference margin: 12.518519
- Embedding L2 movement / cosine with initial: 1.004399 / 0.472116
- Frozen parameter checksums unchanged: yes
- No-token maximum absolute logit difference: 0
- Classification: **B. Weak/mixed effect**

## Held-out generation

| Prompt | Condition | Words | Tokens | Δ trained-base | Δ trained-untrained |
|---|---|---:|---:|---:|---:|
| Sky blue | base | 308 | 376 |  |  |
| Sky blue | untrained | 107 | 128 |  |  |
| Airplane lift | base | 329 | 374 |  |  |
| Airplane lift | untrained | 399 | 459 |  |  |
| Exercise | base | 356 | 443 |  |  |
| Exercise | untrained | 270 | 343 |  |  |
| Sky blue | trained | 253 | 292 | -55 | 146 |
| Airplane lift | trained | 334 | 397 | 5 | -65 |
| Exercise | trained | 377 | 460 | 21 | 107 |

## Position generalization

- prefix: 364 words, 451 tokens
- suffix: 253 words, 292 tokens
- middle: 350 words, 421 tokens

## Learning curve

- N=0: 258.67 mean words
- N=1: 244.33 mean words
- N=2: 329.67 mean words
- N=3: 321.33 mean words

## Interpretation and limitations

This is a three-example proof-of-concept. It cannot establish statistical significance, robust semantic learning, or generalization beyond these prompts. Nearest tokens are geometric diagnostics only. Sequence length, initialization, decoding, and the instruction-tuned chat prior may dominate the result.

## Embedding diagnostics

- L2 movement: 1.004399
- Initial/final cosine similarity: 0.472116
- Five closest tokens after training: 'Ġavid', 'Ġ[*', 'ĠEvangel', 'Ġhym', 'Ġinhib'

Nearest neighbors are exploratory diagnostics and should not be interpreted as the token's literal meaning.
