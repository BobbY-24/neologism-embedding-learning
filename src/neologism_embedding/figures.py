"""Figures for the interpretability analyses and the LoRA baseline.

Each function reads the CSVs written by interp.py / lora_baseline.py and skips
quietly when its inputs are absent, so this can run after any subset of stages.
Experiment 2's own figures are written by `experiment2 --stage analyze`.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

# Categorical slots 1-3 of a colorblind-validated palette, plus neutral ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, MUTED = "#8a8984", "#0b0b0b", "#52514e"
SUBTITLE = "Qwen2.5-0.5B-Instruct · frozen model"


def _read(path: Path):
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": "#e4e3df", "grid.linewidth": 0.8, "lines.linewidth": 2, "font.size": 10,
    })
    return plt


def _mean_by(rows, key, value, where=lambda r: True):
    groups = defaultdict(list)
    for row in rows:
        if where(row):
            groups[int(row[key])].append(float(row[value]))
    keys = sorted(groups)
    return keys, [statistics.mean(groups[k]) for k in keys]


def _title(ax, title, n_note, subtitle=SUBTITLE):
    ax.set_title(f"{title}\n{subtitle} · {n_note}", loc="left", fontsize=10, color=INK)


def logit_lens_stop_figure(interp_dir: Path, figure_dir: Path, subtitle: str = SUBTITLE):
    rows = _read(interp_dir / "logit_lens_stop.csv")
    if not rows:
        return None
    plt = _plt()
    # Raw log-probs span ~150 nats across layers, so plot each condition relative to the no-token run.
    base = {(r["seed"], r["prompt_id"], r["layer"]): float(r["stop_log_probability"]) for r in rows if r["condition"] == "base_no_token"}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for condition, color, label in (("trained", BLUE, "trained E_w"), ("untrained", ORANGE, "untrained E_w⁽⁰⁾")):
        diffs = [
            {"layer": r["layer"], "delta": float(r["stop_log_probability"]) - base[(r["seed"], r["prompt_id"], r["layer"])]}
            for r in rows if r["condition"] == condition
        ]
        layers, means = _mean_by(diffs, "layer", "delta")
        ax.plot(layers, means, color=color, marker="o", markersize=4, label=label)
    ax.axhline(0, color=MUTED, linewidth=1)
    n = len({(r["seed"], r["prompt_id"]) for r in rows})
    ax.set_xlabel("Layer (−1 = embeddings)")
    ax.set_ylabel("Δ logit-lens log p(<|im_end|>) vs no token")
    ax.legend(frameon=False)
    _title(ax, "Stop probability after the concise answer, decoded at each layer", f"{n} seed×prompt pairs; <0 = less likely to stop", subtitle)
    fig.tight_layout()
    path = figure_dir / "interp_logit_lens_stop.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def patching_figure(interp_dir: Path, figure_dir: Path, subtitle: str = SUBTITLE):
    rows = _read(interp_dir / "patching.csv")
    if not rows:
        return None
    from .interp import pooled_restoration
    for row in rows:
        for key in ("layer",):
            row[key] = int(row[key])
        for key in ("metric_clean", "metric_corrupt", "metric_patched"):
            row[key] = float(row[key])
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for seed in sorted({r["seed"] for r in rows}):
        curve = pooled_restoration([r for r in rows if r["seed"] == seed])
        ax.plot(list(curve), list(curve.values()), color=GRAY, linewidth=1, alpha=0.6)
    pooled = pooled_restoration(rows)
    ax.plot(list(pooled), list(pooled.values()), color=BLUE, marker="o", markersize=4, label="pooled over seeds")
    ax.plot([], [], color=GRAY, linewidth=1, label="individual seeds")
    ax.axhline(0, color=MUTED, linewidth=1); ax.axhline(1, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xlabel("Patched layer output at the token position (−1 = input vector)")
    ax.set_ylabel("Fraction of trained effect restored")
    ax.legend(frameon=False)
    n = len({r["prompt_id"] for r in rows})
    _title(ax, "Activation patching: trained → untrained run, token position only", f"{n} held-out prompts", subtitle)
    fig.tight_layout()
    path = figure_dir / "interp_patching.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def residual_change_figure(interp_dir: Path, figure_dir: Path, subtitle: str = SUBTITLE):
    rows = _read(interp_dir / "residual_change.csv")
    if not rows:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for position, color, label in (("token", BLUE, "token position"), ("last_prompt", ORANGE, "last prompt position"), ("stop_decision", AQUA, "stop-decision position")):
        layers, means = _mean_by(rows, "layer", "relative_residual_change", lambda r, p=position: r["position"] == p)
        ax.plot(layers, means, color=color, marker="o", markersize=4, label=label)
    ax.set_xlabel("Layer (−1 = embeddings)")
    ax.set_ylabel("‖h_trained − h_untrained‖ / ‖h_untrained‖")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    _title(ax, "Where the learned vector changes the residual stream", "mean over seeds and prompts", subtitle)
    fig.tight_layout()
    path = figure_dir / "interp_residual_change.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def lora_comparison_figure(table_path: Path, figure_dir: Path, subtitle: str = SUBTITLE):
    rows = _read(table_path)
    if not rows:
        return None
    plt = _plt()
    methods = list(dict.fromkeys(r["method"] for r in rows))
    labels, token_means, plain_means, token_spread = [], [], [], []
    for method in methods:
        selected = [r for r in rows if r["method"] == method]
        token = [float(r["mean_delta_base_token"]) for r in selected]
        plain = [float(r["mean_delta_base_plain"]) for r in selected]
        labels.append(f"{method}\n{int(selected[0]['trainable_parameters']):,} params · {len(selected)} seed{'s' if len(selected) != 1 else ''}")
        token_means.append(statistics.mean(token)); plain_means.append(statistics.mean(plain))
        token_spread.append((statistics.mean(token) - min(token), max(token) - statistics.mean(token)))
    import numpy as np
    x = np.arange(len(methods)); width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2 - 0.01, token_means, width, color=BLUE, label="with token template",
           yerr=np.array(token_spread).T, error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
    ax.bar(x + width / 2 + 0.01, plain_means, width, color=ORANGE, label="plain prompt (no token)")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean Δ words vs base")
    ax.legend(frameon=False)
    _title(ax, "Neologism vs LoRA: held-out length effect and leakage", "error bars: seed min–max", subtitle)
    fig.tight_layout()
    path = figure_dir / "lora_vs_neologism.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def make_all(output_root: Path, mode: str, subtitle: str = SUBTITLE):
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    interp_dir = output_root / "interp" / mode
    made = [
        logit_lens_stop_figure(interp_dir, figure_dir, subtitle),
        patching_figure(interp_dir, figure_dir, subtitle),
        residual_change_figure(interp_dir, figure_dir, subtitle),
        lora_comparison_figure(output_root / "lora_baseline" / f"lora_vs_neologism_{mode}.csv", figure_dir, subtitle),
    ]
    return [path for path in made if path]


def main():
    from .experiment2 import OUTPUT_ROOT, load_config
    parser = argparse.ArgumentParser(description="Interpretability and LoRA-baseline figures")
    parser.add_argument("--mode", choices=["quick", "full", "smoke"], default="full")
    args = parser.parse_args()
    subtitle = f"{load_config(args.mode)['model_id'].split('/')[-1]} · frozen model"
    for path in make_all(OUTPUT_ROOT, args.mode, subtitle):
        print(path)


if __name__ == "__main__":
    main()
