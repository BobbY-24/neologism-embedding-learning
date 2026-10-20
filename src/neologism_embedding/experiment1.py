"""Run the complete real-Qwen three-example neologism experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import (
    NEW_TOKEN,
    FrozenLMWithNeologism,
    add_exactly_one_token,
    assert_checksums_equal,
    chat_prompt_ids,
    dpo_loss,
    greedy_generate,
    json_dump,
    load_preferences,
    logits_for_plain_prompt,
    parameter_checksums,
    random_initial_embedding,
    render_chat_prompt,
    resolve_device,
    response_log_probability,
    seed_everything,
    tokenize_pair,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "experiment_1"
MODEL_PATH = Path(os.environ.get("QWEN_MODEL_PATH", REPO_ROOT / "models" / "Qwen2.5-0.5B-Instruct"))
DATA_PATH = EXPERIMENT_DIR / "data" / "preferences.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "experiment_1"

LEARNING_RATE = 1e-2
EPOCHS = 3
BETA = 0.1
SEED = 42
DEVICE = "auto"
MAX_NEW_TOKENS = 512

HELDOUT = [
    ("Sky blue", "Explain why the sky appears blue."),
    ("Airplane lift", "Explain how airplanes generate lift."),
    ("Exercise", "Explain why exercise can improve cardiovascular health."),
]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_model(path: Path, requested_device: str, dtype: torch.dtype | None = None):
    """dtype=None keeps the original rule: float16 on MPS, float32 elsewhere."""
    device = resolve_device(requested_device)
    if dtype is None:
        dtype = torch.float16 if device.type == "mps" else torch.float32
    fallback_reason = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
    except RuntimeError as error:
        if device.type != "mps":
            raise
        fallback_reason = f"MPS load failed: {error}"
        device = torch.device("cpu")
        model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=True
        ).to(device)
    model.gradient_checkpointing_enable()
    return model, device, fallback_reason


def make_tokenized(tokenizer, records, device, neo_id):
    pairs = []
    for index, record in enumerate(records):
        prompt_ids = chat_prompt_ids(tokenizer, record["prompt"])
        count = prompt_ids.count(neo_id)
        if count != 1:
            raise AssertionError(f"Example {index} contains {count} templated neologism ids, expected 1")
        pairs.append({
            "chosen": tokenize_pair(tokenizer, record["prompt"], record["chosen"], device),
            "rejected": tokenize_pair(tokenizer, record["prompt"], record["rejected"], device),
        })
    return pairs


@torch.no_grad()
def reference_scores(wrapper, tokenized, initial):
    return [
        {
            "chosen": response_log_probability(wrapper, *pair["chosen"], vector=initial).detach(),
            "rejected": response_log_probability(wrapper, *pair["rejected"], vector=initial).detach(),
        }
        for pair in tokenized
    ]


@torch.no_grad()
def dataset_state(wrapper, tokenized, references, beta, limit):
    rows = []
    for index, (pair, ref) in enumerate(zip(tokenized[:limit], references[:limit])):
        chosen = response_log_probability(wrapper, *pair["chosen"])
        rejected = response_log_probability(wrapper, *pair["rejected"])
        loss, margin = dpo_loss(chosen, rejected, ref["chosen"], ref["rejected"], beta)
        rows.append({
            "example_index": index,
            "loss": float(loss.item()),
            "chosen_log_probability": float(chosen.item()),
            "rejected_log_probability": float(rejected.item()),
            "r_chosen": float((chosen - ref["chosen"]).item()),
            "r_rejected": float((rejected - ref["rejected"]).item()),
            "preference_margin": float(margin.item()),
        })
    return rows


def train_vector(wrapper, tokenized, references, initial_cpu, n_examples, epochs, lr, beta, label):
    with torch.no_grad():
        wrapper.neo_embedding.copy_(initial_cpu.to(wrapper.neo_embedding))
    optimizer = torch.optim.AdamW([wrapper.neo_embedding], lr=lr, weight_decay=0.0)
    metrics = []
    step = 0
    started = time.time()
    for epoch in range(epochs):
        for example_index in range(n_examples):
            step += 1
            pair, ref = tokenized[example_index], references[example_index]
            with torch.no_grad():
                chosen_value = response_log_probability(wrapper, *pair["chosen"])
                rejected_value = response_log_probability(wrapper, *pair["rejected"])
                loss, margin = dpo_loss(
                    chosen_value, rejected_value, ref["chosen"], ref["rejected"], beta
                )
                r_chosen = chosen_value - ref["chosen"]
                r_rejected = rejected_value - ref["rejected"]
                dloss_dmargin = beta * (torch.sigmoid(beta * margin) - 1.0)

            optimizer.zero_grad(set_to_none=True)
            chosen_graph = response_log_probability(wrapper, *pair["chosen"])
            chosen_graph.backward(dloss_dmargin)
            rejected_graph = response_log_probability(wrapper, *pair["rejected"])
            rejected_graph.backward(-dloss_dmargin)
            if wrapper.neo_embedding.grad is None:
                raise AssertionError("No gradient reached neo_embedding")
            gradient_norm = float(wrapper.neo_embedding.grad.detach().float().norm().item())
            optimizer.step()

            current_cpu = wrapper.neo_embedding.detach().float().cpu()
            row = {
                "condition": label,
                "step": step,
                "epoch": epoch + 1,
                "example_index": example_index,
                "loss": float(loss.item()),
                "chosen_log_probability": float(chosen_value.item()),
                "rejected_log_probability": float(rejected_value.item()),
                "r_chosen": float(r_chosen.item()),
                "r_rejected": float(r_rejected.item()),
                "preference_margin": float(margin.item()),
                "gradient_norm": gradient_norm,
                "embedding_norm": float(current_cpu.norm().item()),
                "distance_from_initial": float((current_cpu - initial_cpu).norm().item()),
                "cosine_with_initial": float(
                    F.cosine_similarity(current_cpu.view(1, -1), initial_cpu.view(1, -1)).item()
                ),
                "elapsed_seconds": time.time() - started,
            }
            metrics.append(row)
            print(
                f"[{label}] epoch={row['epoch']} example={example_index} "
                f"loss={row['loss']:.6f} rc={row['r_chosen']:.4f} rr={row['r_rejected']:.4f} "
                f"margin={row['preference_margin']:.4f} grad={gradient_norm:.6f} "
                f"move={row['distance_from_initial']:.6f} cos={row['cosine_with_initial']:.6f}",
                flush=True,
            )
    return wrapper.neo_embedding.detach().float().cpu().clone(), metrics


@torch.no_grad()
def generate_one(wrapper, tokenizer, prompt, vector_cpu, max_new_tokens, device):
    vector = vector_cpu.to(device=device, dtype=wrapper.neo_embedding.dtype)
    text, token_ids, termination = greedy_generate(
        wrapper, tokenizer, prompt, vector, max_new_tokens, device
    )
    return {
        "words": len(text.split()),
        "tokens": len(token_ids),
        "termination_reason": termination,
        "output_text": text,
    }


def baseline_evaluation(wrapper, tokenizer, initial, max_new_tokens, device):
    rows = []
    for label, prompt in HELDOUT:
        for condition, actual in [
            ("base", prompt),
            ("untrained", f"{prompt} {NEW_TOKEN}"),
        ]:
            result = generate_one(wrapper, tokenizer, actual, initial, max_new_tokens, device)
            row = {"prompt_label": label, "condition": condition, "prompt": actual, **result}
            rows.append(row)
            print(f"baseline {label} {condition}: {result['words']} words, {result['tokens']} tokens", flush=True)
    return rows


def add_trained_evaluation(wrapper, tokenizer, trained, baseline_rows, max_new_tokens, device):
    rows = list(baseline_rows)
    for label, prompt in HELDOUT:
        result = generate_one(
            wrapper, tokenizer, f"{prompt} {NEW_TOKEN}", trained, max_new_tokens, device
        )
        rows.append({"prompt_label": label, "condition": "trained", "prompt": f"{prompt} {NEW_TOKEN}", **result})
        print(f"post-training {label}: {result['words']} words, {result['tokens']} tokens", flush=True)
    for row in rows:
        if row["condition"] == "trained":
            same = [r for r in rows if r["prompt_label"] == row["prompt_label"]]
            base = next(r for r in same if r["condition"] == "base")
            untrained = next(r for r in same if r["condition"] == "untrained")
            row["delta_trained_base_words"] = row["words"] - base["words"]
            row["delta_trained_untrained_words"] = row["words"] - untrained["words"]
        else:
            row["delta_trained_base_words"] = ""
            row["delta_trained_untrained_words"] = ""
    return rows


def nearest(weight, vector, tokenizer, old_vocab, k=20):
    searchable = weight[:old_vocab].detach().float().cpu()
    scores = F.cosine_similarity(searchable, vector.view(1, -1), dim=1)
    values, ids = scores.topk(k)
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token": tokenizer.convert_ids_to_tokens(int(token_id)),
            "cosine_similarity": float(score),
        }
        for rank, (score, token_id) in enumerate(zip(values, ids), start=1)
    ]


def plot_counts(rows, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [label for label, _ in HELDOUT]
    conditions = ["base", "untrained", "trained"]
    x = list(range(len(labels)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, condition in enumerate(conditions):
        values = [next(r["words"] for r in rows if r["prompt_label"] == label and r["condition"] == condition) for label in labels]
        ax.bar([value + (offset - 1) * width for value in x], values, width, label=condition.title())
    ax.set_xticks(x, labels)
    ax.set_ylabel("Generated words")
    ax.set_title("Held-out response length (3-example proof-of-concept)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default=DEVICE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--skip-learning-curve", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    old_vocab, new_vocab, neo_id = add_exactly_one_token(tokenizer)
    encoded = tokenizer.encode(NEW_TOKEN, add_special_tokens=False)
    assert encoded == [neo_id]

    model, device, fallback_reason = load_model(args.model_path, args.device)
    wrapper = FrozenLMWithNeologism(
        model, neo_id, random_initial_embedding(model.get_input_embeddings().weight, args.seed)
    )
    hidden_size = int(model.config.hidden_size)
    embedding_width = int(model.get_input_embeddings().weight.shape[1])
    assert hidden_size == embedding_width
    assert wrapper.neo_embedding.numel() == hidden_size
    model_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    frozen_parameters = sum(p.numel() for p in model.parameters())
    assert trainable_parameters == hidden_size
    assert all(not p.requires_grad for p in model.parameters())

    print(f"Model: Qwen/Qwen2.5-0.5B-Instruct ({args.model_path.resolve()})")
    print(f"Selected device: {device}")
    print(f"Model parameters: {model_parameters}")
    print(f"Hidden size: {hidden_size}")
    print(f"Vocabulary: {old_vocab} -> {new_vocab}")
    print(f"{NEW_TOKEN} token id: {neo_id}")
    print(f"Trainable parameters: {trainable_parameters}; frozen parameters: {frozen_parameters}")

    records = load_preferences(DATA_PATH)
    for index, record in enumerate(records):
        chosen_words = len(record["chosen"].split())
        rejected_words = len(record["rejected"].split())
        assert 250 <= chosen_words <= 350 and 50 <= rejected_words <= 80
        print(f"Example {index}: chosen={chosen_words} words rejected={rejected_words} words")
    print("\nFully rendered training prompt example:\n" + render_chat_prompt(tokenizer, records[0]["prompt"]))

    tokenized = make_tokenized(tokenizer, records, device, neo_id)
    initial = wrapper.neo_embedding.detach().float().cpu().clone()
    torch.save(initial, OUTPUT_DIR / "initial_embedding.pt")
    initial_device = initial.to(device=device, dtype=wrapper.neo_embedding.dtype)
    references = reference_scores(wrapper, tokenized, initial_device)
    initial_state = dataset_state(wrapper, tokenized, references, args.beta, 3)
    initial_loss = sum(row["loss"] for row in initial_state) / 3
    if abs(initial_loss - math.log(2.0)) > 1e-4:
        raise AssertionError(f"Initial DPO loss {initial_loss} is not approximately ln(2)")
    print(f"Initial mean DPO loss: {initial_loss:.9f}")

    pretrained_before = parameter_checksums(model)
    plain_before = logits_for_plain_prompt(wrapper, tokenizer, HELDOUT[0][1], device)

    # This intentionally happens before the first optimizer step.
    baseline_rows = baseline_evaluation(
        wrapper, tokenizer, initial, args.max_new_tokens, device
    )
    json_dump(OUTPUT_DIR / "pretraining_baseline.json", baseline_rows)

    trained, metrics = train_vector(
        wrapper, tokenized, references, initial, 3, args.epochs,
        args.learning_rate, args.beta, "N=3"
    )
    with torch.no_grad():
        wrapper.neo_embedding.copy_(trained.to(wrapper.neo_embedding))
    final_state = dataset_state(wrapper, tokenized, references, args.beta, 3)
    final_loss = sum(row["loss"] for row in final_state) / 3
    final_margin = sum(row["preference_margin"] for row in final_state) / 3

    pretrained_after = parameter_checksums(model)
    assert_checksums_equal(pretrained_before, pretrained_after)
    plain_after = logits_for_plain_prompt(wrapper, tokenizer, HELDOUT[0][1], device)
    max_logit_difference = float((plain_before - plain_after).abs().max().item())

    evaluation_rows = add_trained_evaluation(
        wrapper, tokenizer, trained, baseline_rows, args.max_new_tokens, device
    )
    json_dump(OUTPUT_DIR / "evaluation_results.json", evaluation_rows)
    write_csv(OUTPUT_DIR / "evaluation_results.csv", evaluation_rows)

    position_rows = []
    for position, prompt in [
        ("prefix", f"{NEW_TOKEN} Explain why the sky appears blue."),
        ("suffix", f"Explain why the sky appears blue. {NEW_TOKEN}"),
        ("middle", f"Explain {NEW_TOKEN} why the sky appears blue."),
    ]:
        result = generate_one(wrapper, tokenizer, prompt, trained, args.max_new_tokens, device)
        position_rows.append({"position": position, "prompt": prompt, **result})
    json_dump(OUTPUT_DIR / "position_generalization.json", position_rows)

    before_neighbors = nearest(model.get_input_embeddings().weight, initial, tokenizer, old_vocab)
    after_neighbors = nearest(model.get_input_embeddings().weight, trained, tokenizer, old_vocab)
    write_csv(OUTPUT_DIR / "embedding_neighbors_before.csv", before_neighbors)
    write_csv(OUTPUT_DIR / "embedding_neighbors_after.csv", after_neighbors)

    movement = float((trained - initial).norm().item())
    embedding_cosine = float(F.cosine_similarity(trained.view(1, -1), initial.view(1, -1)).item())
    artifact = {
        "format_version": 2,
        "checkpoint": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_path": str(args.model_path.resolve()),
        "new_token": NEW_TOKEN,
        "new_token_id": neo_id,
        "old_tokenizer_vocab_size": old_vocab,
        "new_tokenizer_vocab_size": new_vocab,
        "embedding_dimension": hidden_size,
        "initial_embedding": initial,
        "learned_embedding": trained,
    }
    torch.save(artifact, OUTPUT_DIR / "learned_embedding.pt")
    write_csv(OUTPUT_DIR / "training_metrics.csv", metrics)
    json_dump(OUTPUT_DIR / "training_metrics.json", {
        "config": vars(args) | {"model_path": str(args.model_path)},
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_mean_preference_margin": final_margin,
        "initial_per_example": initial_state,
        "final_per_example": final_state,
        "steps": metrics,
    })

    # N=0 reuses the genuine pre-training untrained-token baseline; N=3 reuses main training.
    curve_rows = [{
        "training_examples": 0,
        "mean_heldout_words": sum(r["words"] for r in baseline_rows if r["condition"] == "untrained") / 3,
        "same_initial_embedding": True,
    }]
    if not args.skip_learning_curve:
        for n_examples in (1, 2):
            curve_vector, _ = train_vector(
                wrapper, tokenized, references, initial, n_examples, args.epochs,
                args.learning_rate, args.beta, f"N={n_examples}"
            )
            counts = []
            for _, prompt in HELDOUT:
                result = generate_one(
                    wrapper, tokenizer, f"{prompt} {NEW_TOKEN}", curve_vector,
                    args.max_new_tokens, device
                )
                counts.append(result["words"])
            curve_rows.append({
                "training_examples": n_examples,
                "mean_heldout_words": sum(counts) / 3,
                "same_initial_embedding": True,
            })
    curve_rows.append({
        "training_examples": 3,
        "mean_heldout_words": sum(r["words"] for r in evaluation_rows if r["condition"] == "trained") / 3,
        "same_initial_embedding": True,
    })
    write_csv(OUTPUT_DIR / "learning_curve.csv", curve_rows)

    plot_counts(evaluation_rows, OUTPUT_DIR / "heldout_word_counts.png")
    if len(curve_rows) == 4:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot([r["training_examples"] for r in curve_rows], [r["mean_heldout_words"] for r in curve_rows], marker="o")
        ax.set_xlabel("Preference-training examples")
        ax.set_ylabel("Mean held-out words")
        ax.set_title("Tiny learning curve (proof-of-concept)")
        ax.set_xticks([0, 1, 2, 3])
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "tiny_learning_curve.png", dpi=150)
        plt.close(fig)

    trained_rows = [r for r in evaluation_rows if r["condition"] == "trained"]
    clear = all(
        r["delta_trained_base_words"] > 0 and r["delta_trained_untrained_words"] > 0
        for r in trained_rows
    )
    if final_loss >= initial_loss or final_margin <= 0:
        classification = "D. Optimization failure"
    elif clear:
        classification = "A. Clear proof-of-concept effect"
    elif any(r["delta_trained_base_words"] > 0 or r["delta_trained_untrained_words"] > 0 for r in trained_rows):
        classification = "B. Weak/mixed effect"
    else:
        classification = "C. No detectable effect"

    table_lines = [
        "| Prompt | Condition | Words | Tokens | Δ trained-base | Δ trained-untrained |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in evaluation_rows:
        table_lines.append(
            f"| {row['prompt_label']} | {row['condition']} | {row['words']} | {row['tokens']} | "
            f"{row['delta_trained_base_words']} | {row['delta_trained_untrained_words']} |"
        )
    summary = f"""# Real Qwen neologism experiment

- Checkpoint: `Qwen/Qwen2.5-0.5B-Instruct`
- Local path: `{args.model_path.resolve()}`
- Device: `{device}`
- MPS fallback: {fallback_reason or 'none'}
- Model / frozen parameters: {model_parameters} / {frozen_parameters}
- Hidden size / trainable parameters: {hidden_size} / {trainable_parameters}
- Tokenizer vocabulary: {old_vocab} → {new_vocab}; `{NEW_TOKEN}` id {neo_id}
- Hyperparameters: lr={args.learning_rate}, epochs={args.epochs}, beta={args.beta}, seed={args.seed}
- Decoding: greedy (`do_sample=False`), max_new_tokens={args.max_new_tokens}
- Mean DPO loss: {initial_loss:.9f} → {final_loss:.9f}
- Final mean preference margin: {final_margin:.6f}
- Embedding L2 movement / cosine with initial: {movement:.6f} / {embedding_cosine:.6f}
- Frozen parameter checksums unchanged: yes
- No-token maximum absolute logit difference: {max_logit_difference:.9g}
- Classification: **{classification}**

## Held-out generation

{chr(10).join(table_lines)}

## Position generalization

""" + "\n".join(f"- {r['position']}: {r['words']} words, {r['tokens']} tokens" for r in position_rows) + f"""

## Learning curve

""" + "\n".join(f"- N={r['training_examples']}: {r['mean_heldout_words']:.2f} mean words" for r in curve_rows) + """

## Interpretation and limitations

This is a three-example proof-of-concept. It cannot establish statistical significance, robust semantic learning, or generalization beyond these prompts. Nearest tokens are geometric diagnostics only. Sequence length, initialization, decoding, and the instruction-tuned chat prior may dominate the result.
"""
    (OUTPUT_DIR / "experiment_summary.md").write_text(summary)
    print(f"FINAL loss={final_loss:.9f} margin={final_margin:.6f} movement={movement:.6f} classification={classification}")


if __name__ == "__main__":
    main()
