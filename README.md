# Learning Behavioral Concepts as Single Token Embeddings in Frozen Language Models

[![tests](https://github.com/BobbY-24/neologism-embedding-learning/actions/workflows/tests.yml/badge.svg)](https://github.com/BobbY-24/neologism-embedding-learning/actions/workflows/tests.yml) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)


> Instead of fine-tuning an entire language model, can we add one new token and optimize only its embedding from preference pairs—and when does that embedding begin to behave like a reusable word rather than a soft control vector?

> **Status / Scope: work in progress.** Experiment 1 and Experiment 2 **quick mode** are complete on `Qwen/Qwen2.5-0.5B-Instruct`, which serves as a development and mechanistic testbed. Full multi-seed Experiment 2, the LoRA baseline and the interpretability analyses are **implemented and CPU-tested but not yet run**. They run on a Colab GPU through [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb). A within-family scale replication (Qwen2.5-1.5B and 3B) is also implemented and CPU-tested but not run. Cross-family replication has not started. Current results establish within-model behavior only and should not be generalized to language models broadly.

## Done vs pending

| Component | Code | Results |
|---|---|---|
| Experiment 1: 3-pair pilot | ✅ | ✅ `results/experiment_1/` |
| Experiment 2 quick mode (1 seed, 5 pairs, 10 prompts) | ✅ | ✅ `results/experiment_2/quick/` |
| Experiment 2 full mode (5 genuine + 3 shuffled seeds, 20 pairs, 36 prompts, learning curves) | ✅ | ⏳ not run; pending a Colab run |
| LoRA baseline (matched 1,024-param arm + rank-8 arm, 3 seeds each) | ✅ | ⏳ not run |
| Interpretability: vocabulary neighbours, logit lens, activation patching | ✅ | ⏳ not run |
| Colab driver notebook | ✅ | n/a |
| CPU tests (tiny random Qwen2; no download) | ✅ in CI | n/a |
| Experiment 3: within-family scale (1.5B in fp32, 3B in bf16; frozen 0.5B data; full protocol) | ✅ | ⏳ not run; needs the 0.5B full run first |
| Cross-family replication | ❌ | ❌ |

Everything below the "Current result" heading comes from completed runs. Sections marked *pending* describe methods only and report no numbers.

This repository is a controlled framework for studying the progression

```text
preference fitting → behavioral control → held-out generalization → word-like composition
```

while treating seed robustness, model scale, model family, and concept/task transfer as separate external-validity questions.

## The parameter bottleneck

The experiment adds one token, `<ENSURE_LONG>`, to a Qwen tokenizer and substitutes one independently trainable vector wherever that token occurs. The base embedding table and every transformer parameter stay frozen.

| Parameter group | Count |
|---|---:|
| Frozen Qwen parameters | 494,032,768 |
| **Trainable parameters** | **896** |

```text
Preference pairs
      ↓
DPO-style objective
      ↓
single new embedding E_w
      ↓
frozen Qwen testbed
      ↓
behavior on held-out prompts
      ↓
tests of syntax / compositionality / scale
```

The manually implemented objective is

\[
L=-\log \sigma\left(\beta\left[(\log p_{E_w}(y_c|x)-\log p_{E_w^{(0)}}(y_c|x))-(\log p_{E_w}(y_r|x)-\log p_{E_w^{(0)}}(y_r|x))\right]\right),
\]

with response-token log probabilities only. Optimization updates `E_w`; model parameters `θ_Qwen` are checksum-verified as unchanged.

## Current result: measurable control, not a learned word

Experiment 2 first screened 100 prompts with base Qwen, retained 56 concise/EOS-terminated prompts, and froze a disjoint 20-train/36-held-out split. Its completed **quick run** used 5 exact-prefix preference pairs, 10 held-out prompts, one genuine initialization seed, and one shuffled-label seed.

| Condition | Mean held-out words |
|---|---:|
| Base | 86.0 |
| Untrained token | 94.9 |
| Shuffled token | 93.1 |
| **Trained token** | **159.2** |
| Explicit English | 160.4 |

The genuinely trained embedding increased output length by an average of **+73.2 words relative to base** and **+64.3 relative to the untrained token**. Eight of ten held-out prompts had a positive trained-minus-base difference; the prompt-bootstrap 95% interval was `[+35.5, +117.8]` words. Its mean effect was 0.984 of the explicit-English instruction effect, but that ratio is based on one small run and is not a general equivalence claim.

![Condition word counts](results/experiment_2/quick/figures/condition_word_counts.png)

The controls matter. Genuine preferences reduced DPO loss from `0.693` to `0.124`; shuffled preferences reduced it only to `0.522`. The shuffled token averaged just **+7.1 words over base**. Optimization success and intended held-out behavioral meaning are distinct phenomena.

## Main negative result: syntax dominates

The same trained vector behaved very differently when moved without retraining:

| Placement | Mean words |
|---|---:|
| Prefix (training syntax) | 141.3 |
| Suffix | 72.3 |
| Mid-verb | 330.7 |
| Alternate form | 94.0 |

![Position generalization](results/experiment_2/quick/figures/position_generalization.png)

Some long generations were also repetitive: trained outputs had repeated-bigram fraction `0.207` versus `0.122` for base and `0.144` for explicit English; lexical diversity fell from `0.618` (base) to `0.505` (trained). The vector appears to encode a context-dependent tendency to continue or add material more strongly than a stable, high-quality lexical concept.

## Evidence ladder

| Claim | Current evidence | Scope |
|---|---|---|
| Preference fitting | **Strong** | Qwen2.5-0.5B testbed |
| Behavioral control | **Moderate** | One quick genuine seed |
| Held-out topic transfer | **Moderate, preliminary** | 10 prompts in the same model |
| Word-like compositional behavior | **Absent / weak** | Strong position dependence |
| Seed robustness | **Unknown** | Full multi-seed run implemented, not yet executed |
| Advantage over a LoRA of similar size | **Unknown** | LoRA baseline implemented, not yet executed |
| Where in the network the token acts | **Unknown** | Logit lens / patching implemented, not yet executed |
| Scale generalization | **Unknown** | 1.5B / 3B replication implemented, not yet executed |
| Cross-family generalization | **Unknown** | Qwen only |
| Richer concepts | **Unknown** | Length/elaboration only |

## Pending analyses (implemented, not yet run)

**LoRA baseline** (`src/neologism_embedding/lora_baseline.py`). This arm uses the same 20 pairs, DPO objective, prompt template and 36 held-out prompts. The token stays at its untrained random vector, and only low-rank adapters learn. The LoRA helpers are adapted from [qwen-lora-finetuning](https://github.com/BobbY-24/qwen-lora-finetuning).

| Arm | Adapted modules | Trainable parameters |
|---|---|---:|
| Neologism | one input vector | 896 |
| `lora_matched` | rank 1, `v_proj` of layer 0 | 1,024 (closest standard LoRA) |
| `lora_r8_qv` | rank 8, `q_proj` + `v_proj`, all 24 layers | 540,672 (reported, not matched) |

Each arm is evaluated with the token template and on plain prompts. LoRA is always on, so the plain-prompt condition measures leakage onto ordinary requests. The neologism cannot leak by construction. Learning rates are fixed in `experiments/experiment_2/configs/lora_baseline.json` and were not tuned on held-out data.

**Interpretability of E_w** (`src/neologism_embedding/interp.py`). For each genuine seed:

1. *Vocabulary neighbours.* The nearest input-embedding (and, if untied, unembedding) rows to E_w, to its initialization, and to the update E_w − E_w⁽⁰⁾. The update matters because E_w starts as a random vector.
2. *Logit lens.* Each layer's residual stream is decoded through the final norm and unembedding at the token position. It is also decoded at the position that decides whether to stop after the base model's own concise answer, which gives the layer-wise trajectory of log p(`<|im_end|>`) for the trained, untrained and no-token conditions.
3. *Activation patching at the token position.* Starting from the untrained run, one layer's residual output at the token position is restored from the trained run, and the fraction of the trained effect on the stop log-prob that returns is measured. Patching the input vector restores everything by construction. Patching late layers at that position can restore nothing, because those layers only feed that position's own logits. The curve between the two shows when the information leaves the token position.

Figures are written to `OUTPUT_ROOT/experiment_2/figures/` (`interp_logit_lens_stop.png`, `interp_patching.png`, `interp_residual_change.png`, `lora_vs_neologism.png`) alongside Experiment 2's own figures.

Read [Experiment 1](docs/experiment_1.md), [Experiment 2](docs/experiment_2.md), [Experiment 3 (scale)](docs/experiment_3.md), [limitations](docs/limitations.md), and the [research roadmap](docs/research_roadmap.md) for the complete progression.

## Reproduce

Requirements: Python 3.10+, enough disk/RAM for the external checkpoint, and Git LFS is **not** required.

```bash
git clone https://github.com/BobbY-24/neologism-embedding-learning.git
cd neologism-embedding-learning

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

bash scripts/download_model.sh
bash scripts/reproduce_quick.sh
```

The download script obtains the official `Qwen/Qwen2.5-0.5B-Instruct` checkpoint from Hugging Face into the ignored `models/` directory. Model weights are not part of this repository and remain governed by the upstream model provider's license. To use an existing local Transformers checkpoint instead:

```bash
QWEN_MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct bash scripts/reproduce_quick.sh
```

Experiment 2 is resumable and supports `screen`, `dataset`, `train`, `evaluate`, `positions`, `learning-curve`, `analyze`, and `all` stages:

```bash
python -m neologism_embedding.experiment2 --mode quick --stage analyze
python -m neologism_embedding.experiment2 --mode full --stage screen
```

### Full runs on Google Colab

The full multi-seed Experiment 2, the LoRA baseline and the interpretability analyses run on a Colab GPU:

1. Open [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb) in Colab (File → Open notebook → GitHub → `BobbY-24/neologism-embedding-learning`) and choose a GPU runtime (L4 or A100).
2. Optional: add `HF_TOKEN` under 🔑 Secrets with notebook access. The model is public, and the token is never printed.
3. Run the cells top to bottom. Results go to `MyDrive/neologism-embedding-learning/outputs`, and each stage resumes after a disconnect.
4. The Phase 4 cell writes `results_full.zip` (CSV/JSON/PNG/summary only), which can be copied into `results/experiment_2/full/`.
5. Phase 5 is the scale replication. Set `SCALE_MODEL` to `qwen2.5-1.5b`, run the section, then repeat with `qwen2.5-3b`. It reuses the frozen 0.5B pairs and prompts, writes to `outputs/experiment_3/<model>-<dtype>/`, compares every finished size with the 0.5B run, and bundles `results_scale.zip`. The dataset and dtype choices are explained in [docs/experiment_3.md](docs/experiment_3.md).

The same scripts run anywhere with a checkpoint:

```bash
export QWEN_MODEL_PATH=models/Qwen2.5-0.5B-Instruct NEOLOGISM_OUTPUT_ROOT=outputs
python scripts/run_experiment2.py --mode smoke --stage all   # minutes; checks the pipeline, numbers meaningless
python scripts/run_experiment2.py --mode full --stage all
python scripts/run_interp.py --mode full
python scripts/run_lora_baseline.py --mode full
python scripts/make_figures.py --mode full

# Experiment 3: the same stages with NEOLOGISM_MODEL=qwen2.5-1.5b or qwen2.5-3b, then
python scripts/compare_scales.py --mode full
```

Full mode reuses the committed screening outputs and aborts if the train/held-out split it rebuilds differs from the frozen manifest. All conditions, base included, are generated in the same runtime, so they share one device and dtype.

Completed runs used a MacBook Pro M3 with 8 GB RAM on CPU, Python 3.13.3, PyTorch 2.13.0, and Transformers 5.15.0. Full-mode multi-seed training can take hours on this hardware.

## Repository map

```text
src/neologism_embedding/       frozen-model wrapper and experiment pipelines
  lora_baseline.py             LoRA arm with the same pairs and DPO objective
  interp.py                    neighbours, logit lens, activation patching
  figures.py                   interpretability and LoRA figures
  scale.py                     Experiment 3 cross-size comparison
notebooks/colab_runner.ipynb   Colab GPU driver for the full runs
experiments/experiment_1/      3-pair pilot data and configuration
experiments/experiment_2/      100 candidates, frozen split, 20 prefix-purified pairs
experiments/experiment_3/      scale-replication model registry (0.5B, 1.5B, 3B)
results/experiment_1/          pilot metrics and diagnostics
results/experiment_2/quick/    completed quick-run metrics, figures, and examples
results/experiment_2/prepared_full_dataset/
                               prepared full split; no full behavioral results
docs/                          experiment narratives, limitations, and roadmap
tests/                         freezing, gradient, data, split, cache, LoRA, patching,
                               scale, figure and notebook checks (CPU, tiny random models)
```

## Origin and citation

This work is inspired by and experimentally extends the neologism proposal in John Hewitt, Robert Geirhos, and Been Kim, *We Can't Understand AI Using our Existing Vocabulary* (ICML 2025 position paper; [arXiv:2502.07586](https://arxiv.org/abs/2502.07586)). This repository is independent work and does not imply affiliation with the paper's authors.

To cite this work-in-progress research software, use [CITATION.cff](CITATION.cff).

## License

Repository code is available under the [MIT License](LICENSE). This license does not relicense Qwen or any externally downloaded model weights.

