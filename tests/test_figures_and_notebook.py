import csv
import json
import re
from pathlib import Path

from neologism_embedding.figures import make_all

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_runner.ipynb"


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_figures_render_from_synthetic_outputs(tmp_path):
    interp = tmp_path / "interp" / "smoke"
    layers = range(-1, 3)
    write(interp / "logit_lens_stop.csv", [
        {"seed": 1, "prompt_id": "p", "condition": c, "layer": l, "stop_log_probability": -l - 1.0}
        for c in ("trained", "untrained", "base_no_token") for l in layers
    ])
    write(interp / "patching.csv", [
        {"seed": s, "prompt_id": "p", "layer": l, "token_position": 1, "metric_clean": -3.0,
         "metric_corrupt": -1.0, "metric_patched": -3.0 + l, "metric_base_no_token": -1.0}
        for s in (1, 2) for l in layers
    ])
    write(interp / "residual_change.csv", [
        {"seed": 1, "prompt_id": "p", "layer": l, "position": p, "relative_residual_change": 0.1 * (l + 2),
         "cosine_trained_untrained": 0.9}
        for p in ("token", "last_prompt", "stop_decision") for l in layers
    ])
    write(tmp_path / "lora_baseline" / "lora_vs_neologism_smoke.csv", [
        {"method": m, "seed": s, "trainable_parameters": n, "mean_delta_base_token": 10 * s,
         "mean_delta_base_plain": p}
        for m, n, p in (("neologism", 896, 0), ("lora_matched", 1024, 3)) for s in (1, 2)
    ])
    made = make_all(tmp_path, "smoke")
    assert {p.name for p in made} == {
        "interp_logit_lens_stop.png", "interp_patching.png", "interp_residual_change.png", "lora_vs_neologism.png",
    }
    assert all(p.stat().st_size > 0 for p in made)


def test_figures_skip_missing_inputs(tmp_path):
    assert make_all(tmp_path, "full") == []


def test_colab_notebook_is_valid_and_calls_existing_scripts():
    notebook = json.loads(NOTEBOOK.read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    for script in set(re.findall(r"scripts/[\w./]+\.py", code)):
        assert (ROOT / script).exists(), script
    assert all(not c.get("outputs") for c in notebook["cells"] if c["cell_type"] == "code")


def test_colab_notebook_never_prints_or_embeds_secrets():
    notebook = json.loads(NOTEBOOK.read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "userdata.get(key)" in code and 'secret("HF_TOKEN")' in code
    # The secret is held in `hf` only long enough to export it; it is never formatted or printed.
    assert not re.search(r"print\((hf|os\.environ)", code)
    assert "{hf}" not in code and "{os.environ" not in code and "del hf" in code
    assert "$HF_TOKEN" not in code and "{HF_TOKEN}" not in code
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", code)
