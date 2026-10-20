"""LoRA baseline arm for Experiment 2.

Same preference pairs, DPO objective, templated prompts and held-out prompts as
the neologism, but the learned parameters are low-rank adapters inside the
frozen model. The <ENSURE_LONG> token keeps its untrained random initial vector,
so any behavioral change comes from the adapters alone.

Two evaluations per held-out prompt:
- ``lora_token``: the training template (compare with the neologism's ``trained``);
- ``lora_plain``: the bare prompt. Adapters are always on, so this measures
  leakage onto ordinary prompts, which the neologism has by construction none of.

``target_modules``, ``attach_lora`` and ``accounting`` are adapted from
BobbY-24/qwen-lora-finetuning (src/community_qwen/model/lora.py); ``attach_lora``
gains layer selection so the matched-budget arm can adapt a single layer.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model

from .core import (
    dpo_loss,
    json_dump,
    parameter_checksums,
    random_initial_embedding,
    response_log_probability,
    seed_everything,
)
from .experiment1 import write_csv
from .experiment2 import (
    EXP_DIR,
    OUTPUT_ROOT,
    bootstrap_ci,
    generate,
    load_compatible_cache,
    load_config,
    neologism_parameters,
    runtime,
    save_cache,
    stage_dataset,
    tokenized_preferences,
)

ARM_CONFIG_PATH = EXP_DIR / "configs" / "lora_baseline.json"
LAYER_INDEX = re.compile(r"\.layers\.(\d+)\.")


def target_modules(model, requested="auto_inspect"):
    available = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    targets = [name for name in ["q_proj", "k_proj", "v_proj", "o_proj"] if name in available] if requested == "auto_inspect" else requested
    if not isinstance(targets, list) or not targets or not set(targets) <= available:
        raise ValueError(f"Invalid LoRA targets: {targets}; available linear modules: {sorted(available)}")
    return targets


def attach_lora(model, arm: dict[str, Any]):
    targets = target_modules(model, arm.get("target_modules", "auto_inspect"))
    model = get_peft_model(model, LoraConfig(
        r=arm["rank"], lora_alpha=arm["alpha"], lora_dropout=arm["dropout"], target_modules=targets,
        layers_to_transform=arm.get("layers"), bias="none", task_type="CAUSAL_LM",
    ))
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    return model


def accounting(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(total_parameters=total, trainable_parameters=trainable, frozen_parameters=total - trainable,
                trainable_percent=100 * trainable / total)


def expected_lora_parameters(model, arm: dict[str, Any]) -> int:
    """rank * (in + out) for every targeted Linear in the selected layers, before attaching."""
    targets = set(target_modules(model, arm.get("target_modules", "auto_inspect")))
    layers = arm.get("layers")
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or name.rsplit(".", 1)[-1] not in targets:
            continue
        match = LAYER_INDEX.search(f".{name}.")
        if layers is not None and (match is None or int(match.group(1)) not in layers):
            continue
        count += arm["rank"] * (module.in_features + module.out_features)
    return count


def base_checksums(model) -> dict[str, str]:
    """Checksums of the pretrained weights, keyed by their pre-LoRA names."""
    sums = parameter_checksums(model)
    return {name.replace(".base_layer", ""): value for name, value in sums.items() if "lora_" not in name}


def lora_state(model) -> dict[str, torch.Tensor]:
    return {name: p.detach().float().cpu().clone() for name, p in model.named_parameters() if "lora_" in name}


@torch.no_grad()
def adapter_free_references(peft_model, wrapper, pairs, vector):
    with peft_model.disable_adapter():
        return [
            {
                "chosen": response_log_probability(wrapper, *pair["chosen"], vector=vector).detach(),
                "rejected": response_log_probability(wrapper, *pair["rejected"], vector=vector).detach(),
            }
            for pair in pairs
        ]


@torch.no_grad()
def mean_dpo_state(wrapper, pairs, references, vector, beta):
    losses, margins = [], []
    for pair, ref in zip(pairs, references):
        chosen = response_log_probability(wrapper, *pair["chosen"], vector=vector)
        rejected = response_log_probability(wrapper, *pair["rejected"], vector=vector)
        loss, margin = dpo_loss(chosen, rejected, ref["chosen"], ref["rejected"], beta)
        losses.append(float(loss)); margins.append(float(margin))
    return statistics.mean(losses), statistics.mean(margins)


def train_lora(peft_model, wrapper, pairs, references, vector, epochs, lr, beta, label):
    """DPO on adapter weights; mirrors experiment1.train_vector's two-backward step."""
    params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    metrics, step, started = [], 0, time.time()
    for epoch in range(epochs):
        for index, (pair, ref) in enumerate(zip(pairs, references)):
            step += 1
            with torch.no_grad():
                chosen_value = response_log_probability(wrapper, *pair["chosen"], vector=vector)
                rejected_value = response_log_probability(wrapper, *pair["rejected"], vector=vector)
                loss, margin = dpo_loss(chosen_value, rejected_value, ref["chosen"], ref["rejected"], beta)
                dloss_dmargin = beta * (torch.sigmoid(beta * margin) - 1.0)
            optimizer.zero_grad(set_to_none=True)
            response_log_probability(wrapper, *pair["chosen"], vector=vector).backward(dloss_dmargin)
            response_log_probability(wrapper, *pair["rejected"], vector=vector).backward(-dloss_dmargin)
            if any(p.grad is None for p in params):
                raise AssertionError("A LoRA parameter received no gradient")
            gradient_norm = float(torch.sqrt(sum(p.grad.detach().float().pow(2).sum() for p in params)))
            optimizer.step()
            metrics.append({
                "condition": label, "step": step, "epoch": epoch + 1, "example_index": index,
                "loss": float(loss), "preference_margin": float(margin), "gradient_norm": gradient_norm,
                "lora_norm": float(torch.sqrt(sum(p.detach().float().pow(2).sum() for p in params))),
                "elapsed_seconds": time.time() - started,
            })
            print(f"[{label}] epoch={epoch + 1} example={index} loss={float(loss):.6f} margin={float(margin):.4f} grad={gradient_norm:.6f}", flush=True)
    return metrics


def load_arms(selected=None):
    spec = json.loads(ARM_CONFIG_PATH.read_text())
    arms = spec["arms"] if not selected else {name: spec["arms"][name] for name in selected}
    return spec, arms


def lora_dir(config, arm_name, seed):
    return OUTPUT_ROOT / "lora_baseline" / config["mode"] / arm_name / f"seed_{seed}"


def run_arm_seed(config, arm_name, arm, seed, preferences, heldout, rows, cache_path, signature):
    out = lora_dir(config, arm_name, seed)
    tokenizer, model, wrapper, device, neo_id, _ = runtime(config)
    wrapper.neo_embedding.requires_grad_(False)
    initial = random_initial_embedding(model.get_input_embeddings().weight, seed).float()
    vector = initial.to(device=device, dtype=wrapper.neo_embedding.dtype)
    expected = expected_lora_parameters(model, arm)
    before = base_checksums(model)
    seed_everything(seed)
    peft_model = attach_lora(model, arm)
    counts = accounting(peft_model)
    if counts["trainable_parameters"] != expected:
        raise AssertionError(f"{arm_name}: {counts['trainable_parameters']} trainable != expected {expected}")

    state_path = out / "lora_state.pt"
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=True)
        named = dict(peft_model.named_parameters())
        with torch.no_grad():
            for name, value in saved.items():
                named[name].copy_(value.to(named[name]))
    else:
        n_examples = config["mode_config"]["train_count"]
        pairs, _ = tokenized_preferences(tokenizer, preferences[:n_examples], device, neo_id)
        references = adapter_free_references(peft_model, wrapper, pairs, vector)
        initial_loss, _ = mean_dpo_state(wrapper, pairs, references, vector, config["beta"])
        if abs(initial_loss - math.log(2)) > 1e-4:
            raise AssertionError(f"Initial LoRA loss {initial_loss} is not ln 2")
        metrics = train_lora(peft_model, wrapper, pairs, references, vector, config["epochs"],
                             arm["learning_rate"], config["beta"], f"{arm_name}-seed{seed}")
        final_loss, final_margin = mean_dpo_state(wrapper, pairs, references, vector, config["beta"])
        if base_checksums(model) != before:
            raise AssertionError("Pretrained weights changed during LoRA training")
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / "training_metrics.csv", metrics)
        json_dump(out / "training_summary.json", {
            "arm": arm_name, "seed": seed, "n_examples": n_examples, "arm_config": arm,
            "trainable_parameters": counts["trainable_parameters"],
            "expected_trainable_parameters": expected,
            "ratio_to_neologism": counts["trainable_parameters"] / neologism_parameters(config),
            "initial_loss": initial_loss, "final_loss": final_loss, "final_mean_preference_margin": final_margin,
            "pretrained_checksums_unchanged": True,
        })
        torch.save(lora_state(peft_model), state_path)

    done = {(r["arm"], r["seed"], r["condition"], r["prompt_id"]) for r in rows}
    for item in heldout:
        for condition, prompt in (
            ("lora_token", config["primary_template"].format(prompt=item["prompt"])),
            ("lora_plain", item["prompt"]),
        ):
            if (arm_name, seed, condition, item["id"]) in done:
                continue
            rows.append({
                "arm": arm_name, "seed": seed, "condition": condition, "prompt_id": item["id"], "prompt": prompt,
                **generate(wrapper, tokenizer, prompt, vector.float().cpu(), config["max_new_tokens"], device),
            })
            save_cache(cache_path, signature, rows)
            print(f"lora {arm_name} seed={seed} {condition} {item['id']}: {rows[-1]['words']} words", flush=True)
    del peft_model, wrapper, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize(config, arms, rows, behavior):
    """One row per method and seed: Δ words vs base with the token template and on plain prompts."""
    base = {r["prompt_id"]: r["words"] for r in behavior if r["condition"] == "base"}
    samples, boot_seed = config["mode_config"]["bootstrap_samples"], config["bootstrap_seed"]
    table = []

    def entry(method, seed, params, token_deltas, plain_deltas, plain_source, summary):
        low, high = bootstrap_ci(token_deltas, samples, boot_seed + seed)
        table.append({
            "method": method, "seed": seed, "trainable_parameters": params,
            "n_prompts": len(token_deltas),
            "mean_delta_base_token": statistics.mean(token_deltas),
            "bootstrap_95_low": low, "bootstrap_95_high": high,
            "proportion_delta_positive": sum(d > 0 for d in token_deltas) / len(token_deltas),
            "mean_delta_base_plain": statistics.mean(plain_deltas) if plain_deltas else 0.0,
            "plain_prompt_source": plain_source,
            "final_train_loss": summary.get("final_loss"),
            "final_mean_preference_margin": summary.get("final_mean_preference_margin"),
        })

    from .experiment2 import seed_output_dir
    for seed in sorted({r["seed"] for r in behavior if r["condition"] == "trained"}):
        deltas = [r["words"] - base[r["prompt_id"]] for r in behavior if r["condition"] == "trained" and r["seed"] == seed]
        summary_path = seed_output_dir(config, seed, "genuine") / "training_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        # The plain prompt never contains the token, so its output is the base output.
        entry("neologism", seed, neologism_parameters(config), deltas, [], "identical_to_base_by_construction", summary)
    for arm_name in arms:
        for seed in sorted({r["seed"] for r in rows if r["arm"] == arm_name}):
            selected = [r for r in rows if r["arm"] == arm_name and r["seed"] == seed]
            token = [r["words"] - base[r["prompt_id"]] for r in selected if r["condition"] == "lora_token"]
            plain = [r["words"] - base[r["prompt_id"]] for r in selected if r["condition"] == "lora_plain"]
            summary_path = lora_dir(config, arm_name, seed) / "training_summary.json"
            summary = json.loads(summary_path.read_text())
            entry(arm_name, seed, summary["trainable_parameters"], token, plain, "generated", summary)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["quick", "full", "smoke"], default="full")
    parser.add_argument("--arms", nargs="*", help="subset of arms in configs/lora_baseline.json")
    args = parser.parse_args()
    config = load_config(args.mode)
    spec, arms = load_arms(args.arms)
    seeds = spec["smoke_seeds"] if args.mode == "smoke" else spec["seeds"]
    behavior_path = OUTPUT_ROOT / "controls" / f"behavior_results_{args.mode}.json"
    if not behavior_path.exists():
        raise SystemExit(f"Run Experiment 2 --mode {args.mode} --stage evaluate first (needs {behavior_path})")
    behavior = json.loads(behavior_path.read_text())
    preferences, heldout = stage_dataset(config)
    cache_path = OUTPUT_ROOT / "lora_baseline" / f"generation_cache_{args.mode}.json"
    signature = {
        "stage": "lora_baseline", "mode": args.mode, "model_id": config["model_id"],
        "heldout_ids": [item["id"] for item in heldout], "max_new_tokens": config["max_new_tokens"],
        "do_sample": False, "arms": spec["arms"],
    }
    rows = load_compatible_cache(cache_path, signature) or []
    for arm_name, arm in arms.items():
        for seed in seeds:
            run_arm_seed(config, arm_name, arm, seed, preferences, heldout, rows, cache_path, signature)
    out = OUTPUT_ROOT / "lora_baseline"
    write_csv(out / f"lora_generations_{args.mode}.csv", rows)
    table = summarize(config, arms, rows, behavior)
    write_csv(out / f"lora_vs_neologism_{args.mode}.csv", table)
    print(f"Wrote {out / f'lora_vs_neologism_{args.mode}.csv'}")


if __name__ == "__main__":
    main()
