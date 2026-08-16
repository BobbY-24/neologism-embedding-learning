"""Resumable Experiment 2 pipeline built on Experiment 1's tested primitives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = PROJECT_ROOT / "experiments" / "experiment_2"

from .core import (  # noqa: E402
    NEW_TOKEN,
    FrozenLMWithNeologism,
    add_exactly_one_token,
    assert_checksums_equal,
    greedy_generate,
    json_dump,
    logits_for_plain_prompt,
    parameter_checksums,
    random_initial_embedding,
    response_log_probability,
    seed_everything,
    tokenize_pair,
)
from .experiment1 import (  # noqa: E402
    dataset_state,
    load_model,
    nearest,
    reference_scores,
    train_vector,
    write_csv,
)

CONFIG_PATH = EXP_DIR / "configs" / "default.json"
DATA_DIR = EXP_DIR / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiment_2"
FIGURE_DIR = OUTPUT_ROOT / "figures"


def load_config(mode: str) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text())
    config["mode"] = mode
    config["mode_config"] = config[mode]
    model_override = os.environ.get("QWEN_MODEL_PATH")
    model_path = Path(model_override or config["model_path"])
    config["resolved_model_path"] = str(model_path.expanduser().resolve() if model_override else (EXP_DIR / model_path).resolve())
    return config


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_cache(path: Path, signature: dict[str, Any], payload: Any) -> None:
    json_dump(path, {"signature": signature, "signature_hash": fingerprint(signature), "payload": payload})


def load_compatible_cache(path: Path, signature: dict[str, Any]) -> Any | None:
    if not path.exists():
        return None
    cached = json.loads(path.read_text())
    expected = fingerprint(signature)
    if cached.get("signature_hash") != expected or cached.get("signature") != signature:
        raise RuntimeError(f"Incompatible cache at {path}; refusing silent reuse")
    return cached["payload"]


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def response_metrics(text: str, token_count: int) -> dict[str, Any]:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    word_count = len(words)
    paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    def repeated_ngram_fraction(n: int) -> float:
        grams = [tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))]
        if not grams:
            return 0.0
        counts = Counter(grams)
        repeated_occurrences = sum(count - 1 for count in counts.values() if count > 1)
        return repeated_occurrences / len(grams)

    sentence_norm = [normalize_whitespace(s).lower() for s in sentences]
    repeated_sentences = max(0, len(sentence_norm) - len(set(sentence_norm)))
    example_patterns = re.findall(r"\b(?:for example|for instance|such as|consider)\b", text.lower())
    transition_patterns = re.findall(
        r"\b(?:because|therefore|however|moreover|furthermore|in addition|as a result|consequently|specifically|in contrast)\b",
        text.lower(),
    )
    return {
        "words": word_count,
        "tokens": token_count,
        "paragraphs": len(paragraphs),
        "sentences": len(sentences),
        "lexical_diversity": len(set(words)) / word_count if word_count else 0.0,
        "repeated_bigram_fraction": repeated_ngram_fraction(2),
        "repeated_trigram_fraction": repeated_ngram_fraction(3),
        "repeated_sentence_fraction": repeated_sentences / len(sentences) if sentences else 0.0,
        "example_indicators": len(example_patterns),
        "explanatory_transitions": len(transition_patterns),
        "elaboration_score": len(paragraphs) + len(example_patterns) + len(transition_patterns),
    }


def runtime(config: dict[str, Any], seed: int = 1):
    model_path = Path(config["resolved_model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    old_vocab, new_vocab, neo_id = add_exactly_one_token(tokenizer)
    assert tokenizer.encode(NEW_TOKEN, add_special_tokens=False) == [neo_id]
    model, device, fallback = load_model(model_path, "auto")
    initial = random_initial_embedding(model.get_input_embeddings().weight, seed)
    wrapper = FrozenLMWithNeologism(model, neo_id, initial)
    hidden = int(model.config.hidden_size)
    assert hidden == 896
    assert wrapper.neo_embedding.numel() == 896
    assert sum(p.numel() for p in wrapper.parameters() if p.requires_grad) == 896
    assert all(not p.requires_grad for p in model.parameters())
    write_reproducibility(config, tokenizer, model, device, old_vocab, new_vocab, fallback)
    return tokenizer, model, wrapper, device, neo_id, old_vocab


def write_reproducibility(config, tokenizer, model, device, old_vocab, new_vocab, fallback):
    out = OUTPUT_ROOT / "reproducibility"
    mode_out = out / config["mode"]
    out.mkdir(parents=True, exist_ok=True)
    mode_out.mkdir(parents=True, exist_ok=True)

    def metadata(name, value):
        json_dump(out / name, value)
        json_dump(mode_out / name, value)

    metadata("config.json", config)
    metadata("environment.json", {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "selected_device": str(device),
        "mps_fallback": fallback,
    })
    metadata("random_seeds.json", {
        "initialization_seeds": config["mode_config"]["seeds"],
        "shuffled_seeds": config["mode_config"]["shuffled_seeds"],
        "split_seed": config["split_seed"],
        "shuffle_seed": config["shuffle_seed"],
        "bootstrap_seed": config["bootstrap_seed"],
    })
    metadata("model_info.json", {
        "model_id": config["model_id"],
        "model_path": config["resolved_model_path"],
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "hidden_size": int(model.config.hidden_size),
        "old_tokenizer_vocabulary": old_vocab,
        "new_tokenizer_vocabulary": new_vocab,
        "new_token_id": tokenizer.convert_tokens_to_ids(NEW_TOKEN),
        "model_config": model.config.to_dict(),
    })
    try:
        diff = subprocess.run(
            ["git", "diff"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).stdout
    except Exception as error:
        diff = f"git diff unavailable: {error}\n"
    (out / "git_diff.patch").write_text(diff)
    (mode_out / "git_diff.patch").write_text(diff)


@torch.no_grad()
def generate(wrapper, tokenizer, prompt, vector, max_new_tokens, device):
    text, ids, termination = greedy_generate(
        wrapper,
        tokenizer,
        prompt,
        vector.to(device=device, dtype=wrapper.neo_embedding.dtype),
        max_new_tokens,
        device,
    )
    return {
        "output_text": text,
        "termination_reason": termination,
        **response_metrics(text, len(ids)),
    }


def screen_signature(config, candidates):
    return {
        "stage": "screen",
        "model_id": config["model_id"],
        "prompt_hash": fingerprint(candidates),
        "decoding": {"do_sample": False, "max_new_tokens": config["max_new_tokens"]},
        "chat_template": True,
    }


def stage_screen(config):
    candidates = json.loads((DATA_DIR / "candidate_prompts.json").read_text())
    if len(candidates) != 100:
        raise AssertionError(f"Expected 100 candidate prompts, found {len(candidates)}")
    signature = screen_signature(config, candidates)
    path = OUTPUT_ROOT / "baseline" / "candidate_outputs.json"
    cached = load_compatible_cache(path, signature)
    results = cached or []
    done = {row["id"] for row in results}
    if len(done) < len(candidates):
        tokenizer, model, wrapper, device, _, _ = runtime(config)
        initial = wrapper.neo_embedding.detach().float().cpu()
        for index, item in enumerate(candidates, start=1):
            if item["id"] in done:
                continue
            row = {**item, **generate(wrapper, tokenizer, item["prompt"], initial, config["max_new_tokens"], device)}
            row["passes_limit"] = row["words"] <= config["screen_word_limit"] and row["termination_reason"] == "eos_token"
            row["preferred_range"] = config["preferred_screen_min_words"] <= row["words"] <= config["preferred_screen_max_words"]
            results.append(row)
            save_cache(path, signature, results)
            print(f"screen {index:03d}/100 {item['id']}: {row['words']} words {row['termination_reason']}", flush=True)
    results.sort(key=lambda row: row["id"])
    save_cache(path, signature, results)
    write_csv(DATA_DIR / "prompt_selection_results.csv", results)
    selected = [row for row in results if row["passes_limit"]]
    print(f"Screened {len(results)} prompts; {len(selected)} passed <= {config['screen_word_limit']} words with EOS")
    return results


def frozen_split(config, screening):
    eligible_train = [
        row for row in screening
        if row["passes_limit"] and config["rejected_min_words"] <= row["words"] <= config["rejected_max_words"]
    ]
    eligible_heldout = [row for row in screening if row["passes_limit"]]
    rng = random.Random(config["split_seed"])
    rng.shuffle(eligible_train)
    train = eligible_train[: config["full"]["train_count"]]
    train_topics = {row["topic"] for row in train}
    remaining = [row for row in eligible_heldout if row["topic"] not in train_topics]
    rng.shuffle(remaining)
    heldout = remaining[: config["full"]["heldout_count"]]
    if len(train) < 20 or len(heldout) < 30:
        raise RuntimeError(f"Insufficient screened pool: train={len(train)}, heldout={len(heldout)}")
    return train, heldout


def extend_rejected(wrapper, tokenizer, item, config, device):
    rejected = normalize_whitespace(item["output_text"])
    chosen = rejected
    attempts = []
    while len(chosen.split()) < config["chosen_min_words"] and len(attempts) < 3:
        remaining = max(120, config["chosen_min_words"] - len(chosen.split()) + 60)
        prompt = (
            "Continue the existing answer below with additional accurate explanatory material. "
            f"Write about {remaining} more words. Add mechanisms, examples, implications, and useful qualifications. "
            "Do not repeat the existing answer, do not add a heading, and output only the continuation.\n\n"
            f"Request: {item['prompt']}\n\nExisting answer:\n{chosen}"
        )
        continuation = generate(
            wrapper,
            tokenizer,
            prompt,
            wrapper.neo_embedding.detach().float().cpu(),
            config["max_new_tokens"],
            device,
        )
        attempts.append(continuation)
        chosen = normalize_whitespace(chosen + " " + continuation["output_text"])
    # If iterative continuation stalls or starts echoing itself, restart the
    # extension while retaining the exact rejected prefix. This is recorded as
    # a construction attempt, never hidden as a successful first pass.
    if (
        len(chosen.split()) < config["chosen_min_words"]
        or len(chosen.split()) - len(rejected.split()) < config["minimum_length_gap"]
        or response_metrics(chosen, 0)["repeated_trigram_fraction"] > 0.35
    ):
        fallback_prompt = (
            "Write 320–380 words of additional, accurate explanatory material for the request below. "
            "Cover mechanisms, a concrete example, useful qualifications, and practical implications. "
            "Do not restate or paraphrase the existing concise answer, do not add a heading, and output only new continuation paragraphs.\n\n"
            f"Request: {item['prompt']}\n\nSentences that must not be repeated:\n{rejected}"
        )
        fallback = generate(
            wrapper, tokenizer, fallback_prompt, wrapper.neo_embedding.detach().float().cpu(),
            config["max_new_tokens"], device,
        )
        attempts.append(fallback)
        chosen = normalize_whitespace(rejected + " " + fallback["output_text"])
        focus_areas = [
            "the underlying physical or causal mechanism",
            "a concrete worked example and comparison",
            "limitations, edge cases, misconceptions, and practical implications",
        ]
        for focus in focus_areas:
            if len(chosen.split()) >= config["chosen_min_words"]:
                break
            section_prompt = (
                f"Write 130–180 words of new explanatory detail focused only on {focus}. "
                "Do not add a heading, do not summarize, and do not repeat the concise answer quoted below. "
                "Output only the additional paragraph.\n\n"
                f"Request: {item['prompt']}\n\nConcise answer to avoid repeating:\n{rejected}"
            )
            section = generate(
                wrapper, tokenizer, section_prompt, wrapper.neo_embedding.detach().float().cpu(),
                config["max_new_tokens"], device,
            )
            attempts.append(section)
            chosen = normalize_whitespace(chosen + " " + section["output_text"])
    words = chosen.split()
    if len(words) > config["chosen_max_words"]:
        chosen = " ".join(words[: config["chosen_max_words"]])
    normalized_rejected = normalize_whitespace(rejected)
    normalized_chosen = normalize_whitespace(chosen)
    prefix_ok = normalized_chosen.startswith(normalized_rejected)
    rejected_words = len(normalized_rejected.split())
    chosen_words = len(normalized_chosen.split())
    valid = (
        prefix_ok
        and config["rejected_min_words"] <= rejected_words <= config["rejected_max_words"]
        and config["chosen_min_words"] <= chosen_words <= config["chosen_max_words"]
        and chosen_words - rejected_words >= config["minimum_length_gap"]
    )
    return {
        "id": item["id"],
        "domain": item["domain"],
        "topic": item["topic"],
        "prompt": item["prompt"],
        "training_prompt": config["primary_template"].format(prompt=item["prompt"]),
        "rejected": normalized_rejected,
        "chosen": normalized_chosen,
        "rejected_words": rejected_words,
        "chosen_words": chosen_words,
        "length_gap": chosen_words - rejected_words,
        "prefix_valid": prefix_ok,
        "valid": valid,
        "construction_attempts": len(attempts),
    }


def stage_dataset(config):
    screening = stage_screen(config)
    train_pool, heldout = frozen_split(config, screening)
    screening_by_id = {row["id"]: row for row in screening}
    overrides_path = DATA_DIR / "train_pool_overrides.json"
    overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    for invalid_id, replacement_id in overrides.items():
        train_pool = [screening_by_id[replacement_id] if row["id"] == invalid_id else row for row in train_pool]
        # Keep the established held-out order stable; exchange only membership.
        heldout = [row for row in heldout if row["id"] != replacement_id]
        if invalid_id not in {row["id"] for row in heldout}:
            heldout.append(screening_by_id[invalid_id])
    json_dump(DATA_DIR / "heldout_prompts.json", [
        {key: row[key] for key in ("id", "domain", "topic", "prompt", "words")}
        for row in heldout
    ])
    required = config["mode_config"]["train_count"]
    preferences_path = DATA_DIR / "train_preferences.json"
    existing = json.loads(preferences_path.read_text()) if preferences_path.exists() else []
    by_id = {row["id"]: row for row in existing}
    missing = [row for row in train_pool[:required] if row["id"] not in by_id]
    if missing:
        tokenizer, model, wrapper, device, _, _ = runtime(config)
        for index, item in enumerate(missing, start=1):
            record = extend_rejected(wrapper, tokenizer, item, config, device)
            if not record["valid"]:
                invalid_path = DATA_DIR / "invalid_preference_examples.json"
                invalid_records = json.loads(invalid_path.read_text()) if invalid_path.exists() else []
                invalid_records.append(record)
                json_dump(invalid_path, invalid_records)
                eligible = [
                    row for row in screening
                    if row["passes_limit"]
                    and config["rejected_min_words"] <= row["words"] <= config["rejected_max_words"]
                ]
                random.Random(config["split_seed"]).shuffle(eligible)
                used = {row["id"] for row in train_pool} | set(overrides.values())
                protected_quick_heldout = {row["id"] for row in heldout[: config["quick"]["heldout_count"]]}
                replacement = next(
                    row for row in eligible
                    if row["id"] not in used and row["id"] not in protected_quick_heldout
                )
                overrides[item["id"]] = replacement["id"]
                json_dump(overrides_path, overrides)
                raise RuntimeError(
                    f"Flagged invalid pair {item['id']}; deterministically assigned replacement "
                    f"{replacement['id']}. Rerun dataset stage to continue."
                )
            by_id[item["id"]] = record
            ordered = [by_id[row["id"]] for row in train_pool if row["id"] in by_id]
            json_dump(preferences_path, ordered)
            print(f"dataset {index}/{len(missing)} {item['id']}: {record['rejected_words']} -> {record['chosen_words']} words", flush=True)
    ordered = [by_id[row["id"]] for row in train_pool if row["id"] in by_id]
    json_dump(preferences_path, ordered)
    manifest = {
        "split_seed": config["split_seed"],
        "train_ids_full": [row["id"] for row in train_pool],
        "heldout_ids_full": [row["id"] for row in heldout],
        "available_preference_ids": [row["id"] for row in ordered],
        "nested_subsets": {
            str(size): [row["id"] for row in train_pool[:size]]
            for size in config["full"]["learning_curve_sizes"] if size > 0
        },
        "primary_template": config["primary_template"],
        "screening_hash": fingerprint(screening),
    }
    json_dump(OUTPUT_ROOT / "dataset" / "dataset_manifest.json", manifest)
    json_dump(OUTPUT_ROOT / "reproducibility" / "dataset_manifest.json", manifest)
    destination = OUTPUT_ROOT / "dataset"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATA_DIR / "heldout_prompts.json", destination / "heldout_prompts.json")
    shutil.copy2(preferences_path, destination / "train_preferences.json")
    shutil.copy2(DATA_DIR / "prompt_selection_results.csv", destination / "prompt_selection_results.csv")
    describe_lengths(ordered[:required], OUTPUT_ROOT / "dataset" / "length_statistics.json")
    return ordered[:required], heldout[: config["mode_config"]["heldout_count"]]


def describe_lengths(records, path):
    result = {}
    for field in ("rejected_words", "chosen_words", "length_gap"):
        values = [row[field] for row in records]
        result[field] = {
            "minimum": min(values), "maximum": max(values),
            "mean": statistics.mean(values), "median": statistics.median(values),
        }
    json_dump(path, result)


def tokenized_preferences(tokenizer, preferences, device, neo_id, shuffled=False, shuffle_seed=991):
    flips = [False] * len(preferences)
    if shuffled:
        rng = random.Random(shuffle_seed)
        flips = [rng.random() < 0.5 for _ in preferences]
        if not any(flips):
            flips[0] = True
    pairs = []
    for record, flip in zip(preferences, flips):
        chosen, rejected = (record["rejected"], record["chosen"]) if flip else (record["chosen"], record["rejected"])
        prompt = record["training_prompt"]
        ids = tokenizer.encode(NEW_TOKEN, add_special_tokens=False)
        if ids != [neo_id]:
            raise AssertionError("Neologism stopped being exactly one token")
        pairs.append({
            "chosen": tokenize_pair(tokenizer, prompt, chosen, device),
            "rejected": tokenize_pair(tokenizer, prompt, rejected, device),
        })
    return pairs, flips


def seed_output_dir(config, seed, kind):
    return OUTPUT_ROOT / "seeds" / config["mode"] / f"seed_{seed}" / kind


def validate_training_artifact(path, seed, kind, n_examples):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    observed = (artifact.get("seed"), artifact.get("kind"), artifact.get("n_examples"))
    expected = (seed, kind, n_examples)
    if observed != expected:
        raise RuntimeError(f"Incompatible training artifact at {path}: {observed} != {expected}")
    return artifact


def save_training_run(config, model, wrapper, tokenizer, device, neo_id, old_vocab, preferences, seed, kind, n_examples, out_dir):
    seed_everything(seed)
    initial = random_initial_embedding(model.get_input_embeddings().weight, seed).float().cpu()
    with torch.no_grad():
        wrapper.neo_embedding.copy_(initial.to(wrapper.neo_embedding))
    shuffled = kind == "shuffled"
    pairs, flips = tokenized_preferences(
        tokenizer, preferences[:n_examples], device, neo_id, shuffled, config["shuffle_seed"]
    )
    initial_device = initial.to(device=device, dtype=wrapper.neo_embedding.dtype)
    references = reference_scores(wrapper, pairs, initial_device)
    initial_state = dataset_state(wrapper, pairs, references, config["beta"], n_examples)
    initial_loss = statistics.mean(row["loss"] for row in initial_state)
    if abs(initial_loss - math.log(2)) > 1e-4:
        raise AssertionError(f"Initial loss {initial_loss} is not ln 2")
    before_hash = parameter_checksums(model)
    plain_before = logits_for_plain_prompt(wrapper, tokenizer, "Briefly explain why the sky appears blue.", device)
    final, metrics = train_vector(
        wrapper, pairs, references, initial, n_examples, config["epochs"],
        config["learning_rate"], config["beta"], f"{kind}-seed{seed}-N{n_examples}"
    )
    with torch.no_grad():
        wrapper.neo_embedding.copy_(final.to(wrapper.neo_embedding))
    final_state = dataset_state(wrapper, pairs, references, config["beta"], n_examples)
    final_loss = statistics.mean(row["loss"] for row in final_state)
    final_margin = statistics.mean(row["preference_margin"] for row in final_state)
    assert_checksums_equal(before_hash, parameter_checksums(model))
    plain_after = logits_for_plain_prompt(wrapper, tokenizer, "Briefly explain why the sky appears blue.", device)
    max_logit_difference = float((plain_before - plain_after).abs().max().item())
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "seed": seed, "kind": kind, "n_examples": n_examples,
        "initial_embedding": initial, "learned_embedding": final,
        "model_path": config["resolved_model_path"], "new_token_id": neo_id,
    }, out_dir / "embedding.pt")
    write_csv(out_dir / "training_metrics.csv", metrics)
    json_dump(out_dir / "training_summary.json", {
        "seed": seed, "kind": kind, "n_examples": n_examples,
        "initial_loss": initial_loss, "final_loss": final_loss,
        "final_mean_preference_margin": final_margin,
        "embedding_movement": float((final - initial).norm().item()),
        "embedding_initial_final_cosine": float(F.cosine_similarity(initial[None], final[None]).item()),
        "pretrained_checksums_unchanged": True,
        "max_no_token_logit_difference": max_logit_difference,
        "shuffle_flips": flips,
        "initial_state": initial_state, "final_state": final_state,
    })
    write_csv(out_dir / "nearest_initial.csv", nearest(model.get_input_embeddings().weight, initial, tokenizer, old_vocab))
    write_csv(out_dir / "nearest_final.csv", nearest(model.get_input_embeddings().weight, final, tokenizer, old_vocab))
    return final


def stage_train(config):
    preferences, heldout = stage_dataset(config)
    tokenizer, model, wrapper, device, neo_id, old_vocab = runtime(config)
    n_examples = config["mode_config"]["train_count"]
    for seed in config["mode_config"]["seeds"]:
        out = seed_output_dir(config, seed, "genuine")
        if not (out / "embedding.pt").exists():
            save_training_run(config, model, wrapper, tokenizer, device, neo_id, old_vocab, preferences, seed, "genuine", n_examples, out)
        else:
            validate_training_artifact(out / "embedding.pt", seed, "genuine", n_examples)
    flips_reference = None
    for seed in config["mode_config"]["shuffled_seeds"]:
        out = seed_output_dir(config, seed, "shuffled")
        if not (out / "embedding.pt").exists():
            save_training_run(config, model, wrapper, tokenizer, device, neo_id, old_vocab, preferences, seed, "shuffled", n_examples, out)
        else:
            validate_training_artifact(out / "embedding.pt", seed, "shuffled", n_examples)
        summary = json.loads((out / "training_summary.json").read_text())
        flips_reference = summary["shuffle_flips"]
    json_dump(OUTPUT_ROOT / "controls" / "shuffle_definition.json", {
        "method": "Each example label is independently flipped when random.Random(991).random() < 0.5; at least one flip is forced.",
        "shuffle_seed": config["shuffle_seed"],
        "flips": flips_reference,
    })


def generation_cache_signature(config, heldout):
    return {
        "stage": "evaluate",
        "mode": config["mode"],
        "model_id": config["model_id"],
        "heldout_ids": [row["id"] for row in heldout],
        "max_new_tokens": config["max_new_tokens"],
        "do_sample": False,
        "primary_template": config["primary_template"],
        "explicit_template": config["explicit_template"],
    }


def stage_evaluate(config):
    stage_train(config)
    _, heldout = stage_dataset(config)
    screening = stage_screen(config)
    screen_by_id = {row["id"]: row for row in screening}
    tokenizer, model, wrapper, device, neo_id, old_vocab = runtime(config)
    signature = generation_cache_signature(config, heldout)
    path = OUTPUT_ROOT / "controls" / f"generation_cache_{config['mode']}.json"
    cached = load_compatible_cache(path, signature)
    rows = cached or []
    keys = {(r["seed"], r["condition"], r["prompt_id"]) for r in rows}

    def append(seed, condition, item, prompt, vector, artifact_hash="fixed"):
        key = (seed, condition, item["id"])
        if key in keys:
            return
        if condition == "base":
            source = screen_by_id[item["id"]]
            result = {k: source[k] for k in (
                "output_text", "termination_reason", "words", "tokens", "paragraphs", "sentences",
                "lexical_diversity", "repeated_bigram_fraction", "repeated_trigram_fraction",
                "repeated_sentence_fraction", "example_indicators", "explanatory_transitions", "elaboration_score"
            )}
        else:
            result = generate(wrapper, tokenizer, prompt, vector, config["max_new_tokens"], device)
        row = {
            "seed": seed, "condition": condition, "prompt_id": item["id"],
            "domain": item["domain"], "topic": item["topic"], "prompt": prompt,
            "artifact_hash": artifact_hash, **result,
        }
        rows.append(row)
        keys.add(key)
        save_cache(path, signature, rows)
        print(f"evaluate seed={seed} {condition} {item['id']}: {row['words']} words", flush=True)

    placeholder = wrapper.neo_embedding.detach().float().cpu()
    for item in heldout:
        append(0, "base", item, item["prompt"], placeholder)
        append(0, "explicit", item, config["explicit_template"].format(prompt=item["prompt"]), placeholder)
    for seed in config["mode_config"]["seeds"]:
        artifact_path = seed_output_dir(config, seed, "genuine") / "embedding.pt"
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        digest = file_sha256(artifact_path)
        for item in heldout:
            prompt = config["primary_template"].format(prompt=item["prompt"])
            append(seed, "untrained", item, prompt, artifact["initial_embedding"], digest)
            append(seed, "trained", item, prompt, artifact["learned_embedding"], digest)
    for seed in config["mode_config"]["shuffled_seeds"]:
        artifact_path = seed_output_dir(config, seed, "shuffled") / "embedding.pt"
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        digest = file_sha256(artifact_path)
        for item in heldout:
            prompt = config["primary_template"].format(prompt=item["prompt"])
            append(seed, "shuffled", item, prompt, artifact["learned_embedding"], digest)
    save_cache(path, signature, rows)
    json_dump(OUTPUT_ROOT / "controls" / f"behavior_results_{config['mode']}.json", rows)
    write_csv(OUTPUT_ROOT / "controls" / f"behavior_results_{config['mode']}.csv", rows)


def stage_positions(config):
    result_path = OUTPUT_ROOT / "position_tests" / f"position_results_{config['mode']}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    stage_evaluate(config)
    _, heldout = stage_dataset(config)
    subset = heldout[: config["mode_config"]["position_prompt_count"]]
    tokenizer, model, wrapper, device, _, _ = runtime(config)
    rows = []
    templates = {
        "prefix": "<ENSURE_LONG> your response to the following request:\n\n{prompt}",
        "suffix": "{prompt} <ENSURE_LONG>",
        "mid_verb": "Please <ENSURE_LONG> your response to this request:\n\n{prompt}",
        "alternate": "I want you to <ENSURE_LONG> the answer.\n\n{prompt}",
    }
    for seed in config["mode_config"]["seeds"]:
        artifact = torch.load(
            seed_output_dir(config, seed, "genuine") / "embedding.pt",
            map_location="cpu", weights_only=True,
        )
        for item in subset:
            for position, template in templates.items():
                prompt = template.format(prompt=item["prompt"])
                rows.append({
                    "seed": seed, "position": position, "prompt_id": item["id"], "prompt": prompt,
                    **generate(wrapper, tokenizer, prompt, artifact["learned_embedding"], config["max_new_tokens"], device),
                })
        probe_task = heldout[0]["prompt"]
        probes = {
            "verb_direct": f"Please {NEW_TOKEN} this explanation.\n\n{probe_task}",
            "verb_answer": f"Could you {NEW_TOKEN} your answer?\n\n{probe_task}",
            "adjective": f"I want the explanation to be {NEW_TOKEN}.\n\n{probe_task}",
            "imperative_pronoun": f"The previous answer was too short. {NEW_TOKEN} it.\n\n{probe_task}",
        }
        for name, prompt in probes.items():
            rows.append({
                "seed": seed, "position": f"compositional_{name}", "prompt_id": heldout[0]["id"], "prompt": prompt,
                **generate(wrapper, tokenizer, prompt, artifact["learned_embedding"], config["max_new_tokens"], device),
            })
    json_dump(result_path, rows)
    write_csv(OUTPUT_ROOT / "position_tests" / f"position_results_{config['mode']}.csv", rows)
    return rows


def stage_learning_curve(config):
    result_path = OUTPUT_ROOT / "learning_curve" / f"learning_curve_{config['mode']}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    stage_evaluate(config)
    preferences, heldout = stage_dataset(config)
    tokenizer, model, wrapper, device, neo_id, old_vocab = runtime(config)
    rows = []
    for seed in config["mode_config"]["learning_curve_seeds"]:
        main_artifact = torch.load(
            seed_output_dir(config, seed, "genuine") / "embedding.pt",
            map_location="cpu", weights_only=True,
        )
        initial = main_artifact["initial_embedding"]
        for size in config["mode_config"]["learning_curve_sizes"]:
            if size == 0:
                vector = initial
                margin = 0.0
            elif size == config["mode_config"]["train_count"]:
                vector = main_artifact["learned_embedding"]
                summary = json.loads((seed_output_dir(config, seed, "genuine") / "training_summary.json").read_text())
                margin = summary["final_mean_preference_margin"]
            else:
                out = OUTPUT_ROOT / "learning_curve" / f"seed_{seed}" / f"N_{size}"
                if not (out / "embedding.pt").exists():
                    save_training_run(config, model, wrapper, tokenizer, device, neo_id, old_vocab, preferences, seed, "genuine", size, out)
                artifact = torch.load(out / "embedding.pt", map_location="cpu", weights_only=True)
                vector = artifact["learned_embedding"]
                margin = json.loads((out / "training_summary.json").read_text())["final_mean_preference_margin"]
            effects, counts = [], []
            behavior = json.loads((OUTPUT_ROOT / "controls" / f"behavior_results_{config['mode']}.json").read_text())
            base_by_id = {r["prompt_id"]: r["words"] for r in behavior if r["condition"] == "base"}
            for item in heldout:
                result = generate(
                    wrapper, tokenizer, config["primary_template"].format(prompt=item["prompt"]),
                    vector, config["max_new_tokens"], device,
                )
                counts.append(result["words"])
                effects.append(result["words"] - base_by_id[item["id"]])
            rows.append({
                "seed": seed, "training_examples": size,
                "mean_heldout_words": statistics.mean(counts),
                "mean_delta_words": statistics.mean(effects),
                "proportion_delta_positive": sum(value > 0 for value in effects) / len(effects),
                "final_mean_preference_margin": margin,
                "same_initial_embedding_hash": hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
            })
            print(f"curve seed={seed} N={size}: mean_delta={statistics.mean(effects):.2f}", flush=True)
    write_csv(OUTPUT_ROOT / "learning_curve" / f"learning_curve_{config['mode']}.csv", rows)
    json_dump(result_path, rows)
    return rows


def bootstrap_ci(values, samples, seed):
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples)]
    means.sort()
    return means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]


def summarize_effects(config, behavior):
    base = {r["prompt_id"]: r for r in behavior if r["condition"] == "base"}
    untrained = {(r["seed"], r["prompt_id"]): r for r in behavior if r["condition"] == "untrained"}
    explicit = {r["prompt_id"]: r for r in behavior if r["condition"] == "explicit"}
    rows = []
    for condition in ("trained", "shuffled"):
        seeds = sorted({r["seed"] for r in behavior if r["condition"] == condition})
        for seed in seeds:
            selected = [r for r in behavior if r["condition"] == condition and r["seed"] == seed]
            deltas = [r["words"] - base[r["prompt_id"]]["words"] for r in selected]
            delta_untrained = [r["words"] - untrained[(seed, r["prompt_id"])]["words"] for r in selected]
            low, high = bootstrap_ci(deltas, config["mode_config"]["bootstrap_samples"], config["bootstrap_seed"] + seed)
            base_mean = statistics.mean(base[r["prompt_id"]]["words"] for r in selected)
            explicit_mean = statistics.mean(explicit[r["prompt_id"]]["words"] for r in selected)
            trained_mean = statistics.mean(r["words"] for r in selected)
            denominator = explicit_mean - base_mean
            rows.append({
                "condition": condition, "seed": seed, "n_prompts": len(selected),
                "mean_words": trained_mean,
                "mean_delta_base": statistics.mean(deltas),
                "median_delta_base": statistics.median(deltas),
                "std_delta_base": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "min_delta_base": min(deltas), "max_delta_base": max(deltas),
                "proportion_delta_positive": sum(d > 0 for d in deltas) / len(deltas),
                "proportion_delta_gt_100": sum(d > 100 for d in deltas) / len(deltas),
                "proportion_delta_gt_200": sum(d > 200 for d in deltas) / len(deltas),
                "mean_delta_untrained": statistics.mean(delta_untrained),
                "bootstrap_95_low": low, "bootstrap_95_high": high,
                "explicit_effect_ratio": (trained_mean - base_mean) / denominator if abs(denominator) > 1e-9 else None,
            })
    return rows


def qualitative_examples(behavior, seed):
    by = {}
    for row in behavior:
        if row["condition"] in {"base", "explicit"} or row["seed"] == seed:
            by[(row["prompt_id"], row["condition"])] = row
    ids = sorted({key[0] for key in by if (key[0], "trained") in by})
    combined = []
    for prompt_id in ids:
        required = ["base", "untrained", "trained", "explicit"]
        if not all((prompt_id, condition) in by for condition in required):
            continue
        rows = {condition: by[(prompt_id, condition)] for condition in required}
        delta = rows["trained"]["words"] - rows["base"]["words"]
        oddity = rows["trained"]["repeated_bigram_fraction"] + rows["trained"]["repeated_trigram_fraction"] + rows["trained"]["repeated_sentence_fraction"]
        combined.append({
            "prompt_id": prompt_id, "prompt": rows["base"]["prompt"],
            "delta_words": delta, "oddity_score": oddity,
            **{f"{condition}_output": rows[condition]["output_text"] for condition in required},
            **{f"{condition}_metrics": {k: v for k, v in rows[condition].items() if k not in {"output_text"}} for condition in required},
        })
    count = min(5, len(combined))
    success = sorted(combined, key=lambda r: (-r["delta_words"], r["prompt_id"]))[:count]
    failure = sorted(combined, key=lambda r: (r["delta_words"], r["prompt_id"]))[:count]
    unusual_pool = [r for r in combined if r["delta_words"] > 0]
    unusual = sorted(unusual_pool, key=lambda r: (-r["oddity_score"], r["prompt_id"]))[:count]
    return {"selection_rules": {
        "success": "largest trained-minus-base word delta",
        "failure": "smallest trained-minus-base word delta",
        "unusual": "largest deterministic repetition score among positive-delta cases",
    }, "success": success, "failure": failure, "unusual": unusual}


def stage_analyze(config):
    stage_positions(config)
    stage_learning_curve(config)
    behavior = json.loads((OUTPUT_ROOT / "controls" / f"behavior_results_{config['mode']}.json").read_text())
    aggregates = summarize_effects(config, behavior)
    write_csv(OUTPUT_ROOT / "aggregate_results.csv", aggregates)
    json_dump(OUTPUT_ROOT / "qualitative_examples" / f"examples_{config['mode']}.json", qualitative_examples(behavior, config["mode_config"]["seeds"][0]))

    (OUTPUT_ROOT / "embedding_analysis").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "qualitative_examples").mkdir(parents=True, exist_ok=True)
    training_summaries = []
    finals = []
    for seed in config["mode_config"]["seeds"]:
        root = seed_output_dir(config, seed, "genuine")
        summary = json.loads((root / "training_summary.json").read_text())
        training_summaries.append(summary)
        artifact = torch.load(root / "embedding.pt", map_location="cpu", weights_only=True)
        finals.append((seed, artifact["learned_embedding"].float()))
    cosine_rows = []
    for seed_a, vector_a in finals:
        row = {"seed": seed_a}
        for seed_b, vector_b in finals:
            row[f"seed_{seed_b}"] = float(F.cosine_similarity(vector_a[None], vector_b[None]).item())
        cosine_rows.append(row)
    write_csv(OUTPUT_ROOT / "embedding_analysis" / "final_embedding_seed_cosines.csv", cosine_rows)

    tokenizer, model, wrapper, device, _, _ = runtime(config)
    ordinary = []
    for seed, vector in finals:
        for word in ("long", "detailed", "elaborate", "expand", "thorough", "continue"):
            ids = tokenizer.encode(word, add_special_tokens=False)
            for token_id in ids:
                base_vector = model.get_input_embeddings().weight[token_id].detach().float().cpu()
                ordinary.append({
                    "seed": seed, "word": word, "token_id": token_id,
                    "token": tokenizer.convert_ids_to_tokens(token_id),
                    "cosine_similarity": float(F.cosine_similarity(vector[None], base_vector[None]).item()),
                })
    write_csv(OUTPUT_ROOT / "embedding_analysis" / "ordinary_word_comparisons.csv", ordinary)
    create_figures(config, behavior, aggregates, cosine_rows)
    write_summary(config, behavior, aggregates, training_summaries, cosine_rows)


def create_figures(config, behavior, aggregates, cosine_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    subtitle = "Qwen2.5-0.5B-Instruct · one 896-D embedding · frozen model"
    conditions = ["base", "untrained", "trained", "explicit", "shuffled"]
    means = []
    for condition in conditions:
        selected = [r["words"] for r in behavior if r["condition"] == condition]
        means.append(statistics.mean(selected) if selected else 0)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.bar(conditions, means); ax.set_ylabel("Mean words"); ax.set_title("Condition response lengths\n" + subtitle); fig.tight_layout(); fig.savefig(FIGURE_DIR / "condition_word_counts.png", dpi=150); plt.close(fig)

    base = {r["prompt_id"]: r["words"] for r in behavior if r["condition"] == "base"}
    deltas = [r["words"] - base[r["prompt_id"]] for r in behavior if r["condition"] == "trained"]
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.hist(deltas, bins=min(15, max(5, len(deltas)))); ax.axvline(0, color="black", linewidth=1); ax.set_xlabel("Trained − base words"); ax.set_ylabel("Prompt-seed observations"); ax.set_title("Behavioral effect distribution\n" + subtitle); fig.tight_layout(); fig.savefig(FIGURE_DIR / "delta_distribution.png", dpi=150); plt.close(fig)

    curve = list(csv.DictReader((OUTPUT_ROOT / "learning_curve" / f"learning_curve_{config['mode']}.csv").open()))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seed in sorted({int(r["seed"]) for r in curve}):
        rows = sorted((r for r in curve if int(r["seed"]) == seed), key=lambda r: int(r["training_examples"]))
        ax.plot([int(r["training_examples"]) for r in rows], [float(r["mean_delta_words"]) for r in rows], marker="o", label=f"seed {seed}")
    ax.axhline(0, color="black", linewidth=1); ax.set_xlabel("Training examples"); ax.set_ylabel("Mean Δ words"); ax.set_title("Data-size learning curve\n" + subtitle); ax.legend(); fig.tight_layout(); fig.savefig(FIGURE_DIR / "data_size_learning_curve.png", dpi=150); plt.close(fig)

    trained = [r for r in aggregates if r["condition"] == "trained"]
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.bar([str(r["seed"]) for r in trained], [r["mean_delta_base"] for r in trained]); ax.set_xlabel("Seed"); ax.set_ylabel("Mean Δ words"); ax.set_title("Seed effects\n" + subtitle); fig.tight_layout(); fig.savefig(FIGURE_DIR / "seed_effects.png", dpi=150); plt.close(fig)

    margins = []
    effects = []
    for row in curve:
        margins.append(float(row["final_mean_preference_margin"])); effects.append(float(row["mean_delta_words"]))
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.scatter(margins, effects); ax.set_xlabel("Final mean preference margin"); ax.set_ylabel("Mean held-out Δ words"); ax.set_title("Optimization vs behavior\n" + subtitle); fig.tight_layout(); fig.savefig(FIGURE_DIR / "preference_margin_vs_behavior.png", dpi=150); plt.close(fig)

    positions = json.loads((OUTPUT_ROOT / "position_tests" / f"position_results_{config['mode']}.json").read_text())
    position_names = sorted({r["position"] for r in positions if not r["position"].startswith("compositional_")})
    position_means = [statistics.mean(r["words"] for r in positions if r["position"] == name) for name in position_names]
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.bar(position_names, position_means); ax.set_ylabel("Mean words"); ax.set_title("Position generalization\n" + subtitle); fig.tight_layout(); fig.savefig(FIGURE_DIR / "position_generalization.png", dpi=150); plt.close(fig)

    matrix = np.array([[float(row[f"seed_{other['seed']}"]) for other in cosine_rows] for row in cosine_rows])
    fig, ax = plt.subplots(figsize=(5.5, 4.5)); image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm"); ax.set_xticks(range(len(cosine_rows)), [r["seed"] for r in cosine_rows]); ax.set_yticks(range(len(cosine_rows)), [r["seed"] for r in cosine_rows]); ax.set_xlabel("Seed"); ax.set_ylabel("Seed"); ax.set_title("Final embedding cosine matrix\n" + subtitle); fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(FIGURE_DIR / "embedding_seed_cosine_matrix.png", dpi=150); plt.close(fig)


def evidence_label(preference_ok, mean_delta, positive_rate):
    if preference_ok is not None:
        return "Strong" if preference_ok else "Absent"
    if positive_rate >= 0.8 and mean_delta > 100: return "Strong"
    if positive_rate >= 0.65 and mean_delta > 50: return "Moderate"
    if positive_rate > 0.4 or mean_delta > 0: return "Weak/Mixed"
    return "Absent"


def write_summary(config, behavior, aggregates, training_summaries, cosine_rows):
    trained = [r for r in aggregates if r["condition"] == "trained"]
    shuffled = [r for r in aggregates if r["condition"] == "shuffled"]
    mean_delta = statistics.mean(r["mean_delta_base"] for r in trained)
    positive = statistics.mean(r["proportion_delta_positive"] for r in trained)
    preference_ok = all(r["final_loss"] < math.log(2) and r["final_mean_preference_margin"] > 0 for r in training_summaries)
    positions = json.loads((OUTPUT_ROOT / "position_tests" / f"position_results_{config['mode']}.json").read_text())
    position_means = {
        name: statistics.mean(r["words"] for r in positions if r["position"] == name)
        for name in sorted({r["position"] for r in positions})
    }
    curve = list(csv.DictReader((OUTPUT_ROOT / "learning_curve" / f"learning_curve_{config['mode']}.csv").open()))
    curve_points = sorted((int(r["training_examples"]), float(r["mean_delta_words"])) for r in curve)
    monotonic = all(curve_points[i][1] <= curve_points[i + 1][1] for i in range(len(curve_points) - 1))
    correlation = 0.0
    if len(curve) > 1:
        import numpy as np
        correlation = float(np.corrcoef(
            [float(r["final_mean_preference_margin"]) for r in curve],
            [float(r["mean_delta_words"]) for r in curve],
        )[0, 1])
    claims = {
        "preference_fitting": evidence_label(preference_ok, 0, 0),
        "behavioral_control": evidence_label(None, mean_delta, positive),
        "generalization": evidence_label(None, mean_delta, positive),
        "word_like_behavior": "Weak/Mixed" if max(position_means.values()) - min(position_means.values()) < 100 else "Absent",
    }
    json_dump(OUTPUT_ROOT / "claim_classifications.json", claims)
    summary = f"""# Experiment 2: concept generalization ({config['mode']} mode)

## Configuration

- Model: Qwen/Qwen2.5-0.5B-Instruct
- Trainable parameters: one 896-dimensional embedding; all 494,032,768 Qwen parameters frozen
- Primary syntax: `{config['primary_template']}`
- Training examples: {config['mode_config']['train_count']}; held-out prompts: {config['mode_config']['heldout_count']}
- Genuine seeds: {config['mode_config']['seeds']}; shuffled seeds: {config['mode_config']['shuffled_seeds']}
- Hyperparameters: lr={config['learning_rate']}, beta={config['beta']}, epochs={config['epochs']}

## Claim hierarchy

- Preference fitting: **{claims['preference_fitting']}**
- Behavioral control: **{claims['behavioral_control']}**
- Held-out generalization: **{claims['generalization']}**
- Word-like compositional behavior: **{claims['word_like_behavior']}**

## Main effect

- Mean trained-minus-base words across seeds: {mean_delta:.2f}
- Mean proportion of prompts with Δ>0: {positive:.3f}
- Mean trained-minus-untrained words: {statistics.mean(r['mean_delta_untrained'] for r in trained):.2f}
- Mean shuffled-minus-base words: {statistics.mean(r['mean_delta_base'] for r in shuffled) if shuffled else float('nan'):.2f}
- Mean explicit-effect reproduction ratio: {statistics.mean(r['explicit_effect_ratio'] for r in trained if r['explicit_effect_ratio'] is not None):.3f}
- Preference-margin/behavior correlation over learning-curve runs: {correlation:.3f}
- Learning curve monotonic: {monotonic}

## Position/compositional means

""" + "\n".join(f"- {name}: {value:.2f} words" for name, value in position_means.items()) + """

## Caution

Prompt observations are not IID draws from a formally defined population. Bootstrap intervals quantify prompt-set variability only. Successful DPO optimization is not by itself evidence of semantic or lexical acquisition.

## Execution scope

Quick mode was completed end-to-end. Full screening and 20-pair dataset
construction were completed, but full multi-seed training/evaluation was not
run because measured CPU throughput implies multiple hours. No full-mode
behavioral metrics are claimed by this summary.
"""
    (OUTPUT_ROOT / "experiment_summary.md").write_text(summary)


STAGES = {
    "screen": stage_screen,
    "dataset": stage_dataset,
    "train": stage_train,
    "evaluate": stage_evaluate,
    "positions": stage_positions,
    "learning-curve": stage_learning_curve,
    "analyze": stage_analyze,
}


def main(default_stage: str | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=[*STAGES, "all"], default=default_stage or "all")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    args = parser.parse_args()
    config = load_config(args.mode)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        stage_analyze(config)
    else:
        STAGES[args.stage](config)


if __name__ == "__main__":
    main()
