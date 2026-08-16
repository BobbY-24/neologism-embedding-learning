# Experiment 2: concept generalization (quick mode)

## Configuration

- Model: Qwen/Qwen2.5-0.5B-Instruct
- Trainable parameters: one 896-dimensional embedding; all 494,032,768 Qwen parameters frozen
- Primary syntax: `<ENSURE_LONG> your response to the following request:

{prompt}`
- Training examples: 5; held-out prompts: 10
- Genuine seeds: [1]; shuffled seeds: [1]
- Hyperparameters: lr=0.01, beta=0.1, epochs=3

## Claim hierarchy

- Preference fitting: **Strong**
- Behavioral control: **Moderate**
- Held-out generalization: **Moderate**
- Word-like compositional behavior: **Absent**

## Main effect

- Mean trained-minus-base words across seeds: 73.20
- Mean proportion of prompts with Δ>0: 0.800
- Mean trained-minus-untrained words: 64.30
- Mean shuffled-minus-base words: 7.10
- Mean explicit-effect reproduction ratio: 0.984
- Preference-margin/behavior correlation over learning-curve runs: 0.909
- Learning curve monotonic: True

## Position/compositional means

- alternate: 94.00 words
- compositional_adjective: 123.00 words
- compositional_imperative_pronoun: 89.00 words
- compositional_verb_answer: 412.00 words
- compositional_verb_direct: 375.00 words
- mid_verb: 330.67 words
- prefix: 141.33 words
- suffix: 72.33 words

## Caution

Prompt observations are not IID draws from a formally defined population. Bootstrap intervals quantify prompt-set variability only. Successful DPO optimization is not by itself evidence of semantic or lexical acquisition.

## Execution scope

Quick mode was completed end-to-end. Full screening and 20-pair dataset
construction were completed, but full multi-seed training/evaluation was not
run because measured CPU throughput implies multiple hours. No full-mode
behavioral metrics are claimed by this summary.
