"""Experiment 3 (RQ6): compare the neologism across Qwen2.5 sizes.

Each model is trained and evaluated by the Experiment 2 pipeline with
NEOLOGISM_MODEL=<key>, which writes to <output base>/experiment_3/<key>-<dtype>.
The 0.5B reference is read from <output base>/experiment_2. This module only
reads those outputs; it trains and generates nothing.

The data are the frozen 0.5B pairs and prompts for every size. The column
base_passes_screen measures how far that drifts from each model's own
behavior: the fraction of held-out prompts on which the model's own base answer
would still pass the 0.5B screen (<= screen_word_limit words, ended by EOS).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .experiment1 import write_csv
from .experiment2 import CONFIG_PATH, SCALE_CONFIG_PATH, json_dump

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def model_dirs(output_base: Path, mode: str) -> list[tuple[str, Path]]:
    """(label, directory) for every model with evaluate-stage outputs in this mode."""
    found = []
    if (output_base / "experiment_2" / "controls" / f"behavior_results_{mode}.json").exists():
        found.append(("experiment_2", output_base / "experiment_2"))
    scale_root = output_base / "experiment_3"
    if scale_root.exists():
        for directory in sorted(scale_root.iterdir()):
            if (directory / "controls" / f"behavior_results_{mode}.json").exists():
                found.append((directory.name, directory))
    return found


def summarize_model(label: str, root: Path, mode: str, screen_word_limit: int) -> dict[str, Any]:
    behavior = _json(root / "controls" / f"behavior_results_{mode}.json")
    info = _json(root / "reproducibility" / mode / "model_info.json") or {}
    config = _json(root / "reproducibility" / mode / "config.json") or {}
    environment = _json(root / "reproducibility" / mode / "environment.json") or {}
    base = {r["prompt_id"]: r for r in behavior if r["condition"] == "base"}
    explicit = {r["prompt_id"]: r for r in behavior if r["condition"] == "explicit"}

    def per_seed(condition):
        seeds = sorted({r["seed"] for r in behavior if r["condition"] == condition})
        return {
            seed: statistics.mean(
                r["words"] - base[r["prompt_id"]]["words"]
                for r in behavior if r["condition"] == condition and r["seed"] == seed
            )
            for seed in seeds
        }

    trained, shuffled = per_seed("trained"), per_seed("shuffled")
    trained_rows = [r for r in behavior if r["condition"] == "trained"]
    explicit_delta = statistics.mean(explicit[p]["words"] - base[p]["words"] for p in base)
    summaries = [
        _json(path) for path in sorted((root / "seeds" / mode).glob("seed_*/genuine/training_summary.json"))
    ]
    interp = _json(root / "interp" / mode / "interp_summary.json") or {}
    mean_trained = statistics.mean(trained.values()) if trained else None
    # Experiment 2 never sets a dtype: it loads float16 on MPS and float32 elsewhere.
    device = environment.get("selected_device", "")
    dtype = config.get("dtype") or ("float16" if device.startswith("mps") else "float32")
    return {
        "run": label,
        "model_id": info.get("model_id", config.get("model_id")),
        "dtype": dtype,
        "device": device,
        "hidden_size": info.get("hidden_size"),
        "model_parameters": info.get("model_parameters"),
        "n_prompts": len(base),
        "n_seeds": len(trained),
        "mean_base_words": statistics.mean(r["words"] for r in base.values()),
        "base_passes_screen": sum(
            r["words"] <= screen_word_limit and r["termination_reason"] == "eos_token" for r in base.values()
        ) / len(base),
        "mean_delta_base": mean_trained,
        "seed_min_delta_base": min(trained.values()) if trained else None,
        "seed_max_delta_base": max(trained.values()) if trained else None,
        "proportion_delta_positive": (
            sum(r["words"] > base[r["prompt_id"]]["words"] for r in trained_rows) / len(trained_rows)
            if trained_rows else None
        ),
        "mean_shuffled_delta_base": statistics.mean(shuffled.values()) if shuffled else None,
        "explicit_delta_base": explicit_delta,
        "explicit_effect_ratio": mean_trained / explicit_delta if mean_trained is not None and abs(explicit_delta) > 1e-9 else None,
        "mean_final_loss": statistics.mean(s["final_loss"] for s in summaries) if summaries else None,
        "mean_final_preference_margin": statistics.mean(s["final_mean_preference_margin"] for s in summaries) if summaries else None,
        "interp_stop_logprob_effect": interp.get("mean_stop_logprob_effect_trained_minus_untrained"),
        "per_seed_delta_base": json.dumps(trained),
    }


def compare(output_base: Path, mode: str) -> tuple[list[dict[str, Any]], Path | None]:
    found = model_dirs(output_base, mode)
    if not found:
        raise SystemExit(f"No evaluate-stage outputs for mode {mode!r} under {output_base}")
    limit = json.loads(CONFIG_PATH.read_text())["screen_word_limit"]
    rows = [summarize_model(label, root, mode, limit) for label, root in found]
    rows.sort(key=lambda r: (r["model_parameters"] or 0, r["run"]))
    out = output_base / "experiment_3"
    write_csv(out / f"scale_comparison_{mode}.csv", rows)
    json_dump(out / f"scale_comparison_{mode}.json", {
        "mode": mode,
        "dataset_policy": json.loads(SCALE_CONFIG_PATH.read_text())["dataset_policy"],
        "rows": rows,
    })
    return rows, scale_figure(rows, out / f"scale_comparison_{mode}.png")


def scale_figure(rows, path: Path):
    rows = [r for r in rows if r["mean_delta_base"] is not None]
    if not rows:
        return None
    from .figures import _plt
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [f"{r['model_id'].split('/')[-1].replace('-Instruct', '')}\n{r['dtype']}" for r in rows]
    for x, row in enumerate(rows):
        seeds = json.loads(row["per_seed_delta_base"])
        ax.scatter([x] * len(seeds), list(seeds.values()), color=BLUE, alpha=0.5, s=18, label="seed" if x == 0 else None)
        ax.scatter([x], [row["mean_delta_base"]], color=BLUE, s=60, marker="D", label="mean over seeds" if x == 0 else None)
        ax.scatter([x], [row["explicit_delta_base"]], facecolors="none", edgecolors=ORANGE, s=60, label="explicit instruction" if x == 0 else None)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(range(len(rows)), labels)
    ax.set_ylabel("Mean held-out Δ words vs base")
    ax.legend(frameon=False)
    ax.set_title("Neologism effect across Qwen2.5 sizes\nsame frozen 0.5B pairs and prompts · one hidden-size vector per model",
                 loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def main():
    from .experiment2 import OUTPUT_BASE
    parser = argparse.ArgumentParser(description="Compare Experiment 2 outputs across Qwen2.5 model sizes")
    parser.add_argument("--mode", choices=["quick", "full", "smoke"], default="full")
    args = parser.parse_args()
    rows, figure = compare(OUTPUT_BASE, args.mode)
    for row in rows:
        print(f"{row['run']}: Δ={row['mean_delta_base']} words over {row['n_seeds']} seeds, base passes screen {row['base_passes_screen']:.2f}")
    print(OUTPUT_BASE / "experiment_3" / f"scale_comparison_{args.mode}.csv")
    if figure:
        print(figure)


if __name__ == "__main__":
    main()
