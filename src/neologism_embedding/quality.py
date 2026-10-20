"""Quality-adjusted re-analysis of saved generations; reads output files only, loads no model.

Raw word count rewards degenerate looping. Two stricter measures:

- effective length: the number of distinct word trigrams + 2, i.e. the words an
  output would keep if every repeated three-word span were deleted. For text with
  no repeated trigram it equals the word count.
- clean output: ends with EOS (not cut off by the token cap) and its repeated-
  trigram fraction is no higher than the most repetitive base output. The
  threshold is taken from the base condition, not tuned on trained outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

from .experiment1 import write_csv
from .experiment2 import bootstrap_ci

WORD = re.compile(r"[A-Za-z0-9']+")  # same tokenization as experiment2.response_metrics


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def effective_length(text: str) -> int:
    tokens = words(text)
    if len(tokens) < 3:
        return len(tokens)
    return len({tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)}) + 2


def repeated_trigram_fraction(text: str) -> float:
    tokens = words(text)
    grams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def is_clean(row: dict, threshold: float) -> bool:
    return row["termination_reason"] == "eos_token" and repeated_trigram_fraction(row["output_text"]) <= threshold


def annotate(rows, threshold):
    for row in rows:
        row["effective_words"] = effective_length(row["output_text"])
        row["raw_words"] = len(words(row["output_text"]))
        row["clean"] = is_clean(row, threshold)
        row["hit_cap"] = row["termination_reason"] == "max_new_tokens"
    return rows


def summarize_group(label, seed, rows, base_effective, base_raw, samples, boot_seed):
    deltas = [r["effective_words"] - base_effective[r["prompt_id"]] for r in rows]
    raw_deltas = [r["raw_words"] - base_raw[r["prompt_id"]] for r in rows]
    low, high = bootstrap_ci(deltas, samples, boot_seed)
    clean = [r for r in rows if r["clean"]]
    return {
        "condition": label, "seed": seed, "n": len(rows),
        "mean_raw_words": statistics.mean(r["raw_words"] for r in rows),
        "mean_effective_words": statistics.mean(r["effective_words"] for r in rows),
        "mean_delta_raw": statistics.mean(raw_deltas),
        "mean_delta_effective": statistics.mean(deltas),
        "effective_bootstrap_95_low": low, "effective_bootstrap_95_high": high,
        "retained_fraction": statistics.mean(r["effective_words"] for r in rows) / statistics.mean(r["raw_words"] for r in rows),
        "proportion_hit_cap": statistics.mean(r["hit_cap"] for r in rows),
        "proportion_clean": len(clean) / len(rows),
        "proportion_clean_and_longer": sum(r["clean"] and r["raw_words"] > base_raw[r["prompt_id"]] for r in rows) / len(rows),
        "mean_delta_raw_clean_only": statistics.mean(r["raw_words"] - base_raw[r["prompt_id"]] for r in clean) if clean else None,
    }


def analyze(results_dir: Path, mode: str, samples: int = 5000, boot_seed: int = 31415):
    behavior = json.loads((results_dir / "controls" / f"behavior_results_{mode}.json").read_text())
    base_rows = [r for r in behavior if r["condition"] == "base"]
    threshold = max(repeated_trigram_fraction(r["output_text"]) for r in base_rows)
    annotate(behavior, threshold)
    base_effective = {r["prompt_id"]: r["effective_words"] for r in base_rows}
    base_raw = {r["prompt_id"]: r["raw_words"] for r in base_rows}

    groups = []  # (label, seed, rows)
    for condition in ("base", "untrained", "shuffled", "explicit", "trained"):
        selected = [r for r in behavior if r["condition"] == condition]
        groups.append((condition, "all", selected))
        if condition in ("trained", "shuffled"):
            for seed in sorted({r["seed"] for r in selected}):
                groups.append((condition, seed, [r for r in selected if r["seed"] == seed]))

    lora_path = results_dir / "lora_baseline" / f"lora_generations_{mode}.csv"
    if lora_path.exists():
        with lora_path.open(newline="") as handle:
            lora = list(csv.DictReader(handle))
        annotate(lora, threshold)
        for arm in sorted({r["arm"] for r in lora}):
            for condition in ("lora_token", "lora_plain"):
                selected = [r for r in lora if r["arm"] == arm and r["condition"] == condition]
                groups.append((f"{arm}:{condition}", "all", selected))
                for seed in sorted({r["seed"] for r in selected}, key=int):
                    groups.append((f"{arm}:{condition}", int(seed), [r for r in selected if r["seed"] == seed]))

    summary = [
        summarize_group(label, seed, rows, base_effective, base_raw, samples, boot_seed + (seed if isinstance(seed, int) else 0))
        for label, seed, rows in groups
    ]
    explicit = next(s for s in summary if s["condition"] == "explicit")
    for row in summary:
        row["explicit_ratio_raw"] = row["mean_delta_raw"] / explicit["mean_delta_raw"]
        row["explicit_ratio_effective"] = row["mean_delta_effective"] / explicit["mean_delta_effective"]

    out = results_dir / "quality"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / f"quality_summary_{mode}.csv", summary)
    per_output = [
        {k: r.get(k) for k in ("condition", "arm", "seed", "prompt_id", "termination_reason", "raw_words", "effective_words", "clean")}
        for r in behavior + (lora if lora_path.exists() else [])
    ]
    write_csv(out / f"quality_per_output_{mode}.csv", per_output)
    json_out = {
        "clean_rule": "EOS-terminated and repeated-trigram fraction <= max over base outputs",
        "repeated_trigram_threshold": threshold,
        "effective_length": "distinct word trigrams + 2",
    }
    (out / f"quality_definitions_{mode}.json").write_text(json.dumps(json_out, indent=2) + "\n")
    return summary, threshold


def quality_figure(summary, figure_dir: Path):
    from .figures import BLUE, INK, MUTED, ORANGE, _plt, _title
    import numpy as np

    order = [("base", "Base"), ("untrained", "Untrained"), ("shuffled", "Shuffled"), ("explicit", "Explicit\nEnglish"),
             ("trained", "Neologism"), ("lora_matched:lora_token", "LoRA 1,024\n(token)"),
             ("lora_matched:lora_plain", "LoRA 1,024\n(plain)"), ("lora_r8_qv:lora_token", "LoRA r8\n(token)"),
             ("lora_r8_qv:lora_plain", "LoRA r8\n(plain)")]
    rows = {s["condition"]: s for s in summary if s["seed"] == "all"}
    present = [(key, label) for key, label in order if key in rows]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(present)); width = 0.38
    ax.bar(x - width / 2 - 0.01, [rows[k]["mean_raw_words"] for k, _ in present], width, color=ORANGE, label="raw words")
    ax.bar(x + width / 2 + 0.01, [rows[k]["mean_effective_words"] for k, _ in present], width, color=BLUE, label="effective words (repeats removed)")
    for i, (key, _) in enumerate(present):
        ax.text(x[i], rows[key]["mean_raw_words"] + 8, f"{rows[key]['proportion_clean']:.0%} clean", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(x, [label for _, label in present], fontsize=8.5)
    ax.set_ylabel("Mean words per held-out answer")
    ax.legend(frameon=False, loc="upper left")
    _title(ax, "Raw vs quality-adjusted length", "clean = ended with EOS and no more repetitive than any base answer")
    fig.tight_layout()
    path = figure_dir / "quality_adjusted_length.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True, help="e.g. results/experiment_2/full")
    parser.add_argument("--mode", default="full")
    args = parser.parse_args()
    summary, threshold = analyze(args.results, args.mode)
    figure = quality_figure(summary, args.results / "figures")
    print(f"clean threshold (max base repeated-trigram fraction): {threshold:.3f}")
    for row in summary:
        if row["seed"] == "all":
            print(f"{row['condition']:<26} raw {row['mean_raw_words']:6.1f}  effective {row['mean_effective_words']:6.1f}  "
                  f"Δeff {row['mean_delta_effective']:+7.1f}  clean {row['proportion_clean']:.0%}")
    print(figure)


if __name__ == "__main__":
    main()
