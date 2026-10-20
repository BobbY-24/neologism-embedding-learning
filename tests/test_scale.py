import json
import math
import re
from pathlib import Path

import pytest
import torch

from neologism_embedding import experiment2 as e
from neologism_embedding.core import FrozenLMWithNeologism, parameter_checksums
from neologism_embedding.experiment1 import dataset_state, load_model, reference_scores, train_vector
from neologism_embedding.scale import compare

ROOT = Path(__file__).resolve().parents[1]
SCALE_KEYS = {"scale_model", "model_id", "hidden_size", "dtype", "screening_model_id", "experiment_name", "model_path", "resolved_model_path"}


def _pairs(neo_id):
    prompt = [1, 2, neo_id, 3]
    chosen = torch.tensor([prompt + [5, 6, 7, 8, 9]])
    rejected = torch.tensor([prompt + [5, 6, 10]])
    return [{"chosen": (chosen, len(prompt)), "rejected": (rejected, len(prompt))}]


def test_float32_master_vector_trains_through_a_bf16_model(tiny_qwen):
    model = tiny_qwen.to(torch.bfloat16)
    neo_id = model.config.vocab_size
    initial = torch.randn(model.config.hidden_size) * 0.02
    wrapper = FrozenLMWithNeologism(model, neo_id, initial.clone(), trainable_dtype=torch.float32)
    assert wrapper.neo_embedding.dtype == torch.float32
    assert wrapper.embed(torch.tensor([[1, neo_id]])).dtype == torch.bfloat16
    pairs = _pairs(neo_id)
    references = reference_scores(wrapper, pairs, initial)
    # Policy and reference are the same bf16 forward pass, so the DPO loss starts at exactly ln 2.
    assert dataset_state(wrapper, pairs, references, 0.1, 1)[0]["loss"] == pytest.approx(math.log(2), abs=1e-6)
    before = parameter_checksums(model)
    final, metrics = train_vector(wrapper, pairs, references, initial, 1, 2, 0.01, 0.1, "bf16")
    assert parameter_checksums(model) == before
    assert final.dtype == torch.float32 and torch.isfinite(final).all()
    assert metrics[-1]["gradient_norm"] > 0 and (final - initial).norm() > 0


def test_default_wrapper_keeps_the_model_dtype(tiny_qwen):
    wrapper = FrozenLMWithNeologism(tiny_qwen, 64, torch.randn(32))
    assert wrapper.neo_embedding.dtype == tiny_qwen.get_input_embeddings().weight.dtype == torch.float32


def test_load_model_dtype_override(tmp_path, tiny_qwen):
    tiny_qwen.save_pretrained(tmp_path)
    assert next(load_model(tmp_path, "cpu")[0].parameters()).dtype == torch.float32
    assert next(load_model(tmp_path, "cpu", torch.bfloat16)[0].parameters()).dtype == torch.bfloat16


def test_scale_spec_and_output_root():
    assert e.scale_spec(None) is None and e.scale_spec("") is None
    spec = e.scale_spec("qwen2.5-3b")
    assert (spec["model_id"], spec["hidden_size"], spec["dtype"]) == ("Qwen/Qwen2.5-3B-Instruct", 2048, "bfloat16")
    assert e.scale_spec("qwen2.5-3b", "float32")["dtype"] == "float32"
    assert e.output_root(Path("o"), None) == Path("o/experiment_2")
    assert e.output_root(Path("o"), spec) == Path("o/experiment_3/qwen2.5-3b-bfloat16")
    with pytest.raises(ValueError, match="Unknown NEOLOGISM_MODEL"):
        e.scale_spec("qwen2.5-7b")
    with pytest.raises(ValueError, match="Unknown NEOLOGISM_DTYPE"):
        e.scale_spec("qwen2.5-3b", "int8")


def test_registry_entries_are_consistent():
    registry = json.loads(e.SCALE_CONFIG_PATH.read_text())
    reference = registry["models"][registry["reference"]]
    assert reference["hidden_size"] == e.DEFAULT_HIDDEN_SIZE
    assert reference["model_id"] == registry["screening_model_id"] == e.load_config("full")["model_id"]
    for key, entry in registry["models"].items():
        assert entry["dtype"] in e.DTYPES
        assert re.fullmatch(r"Qwen/Qwen2\.5-[\d.]+B-Instruct", entry["model_id"])
        assert key == entry["model_id"].split("/")[1].removesuffix("-Instruct").lower()


@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_scale_config_swaps_only_the_model(mode, monkeypatch):
    monkeypatch.delenv("QWEN_MODEL_PATH", raising=False)
    original = e.load_config(mode)
    # The 0.5B path must not gain any scale keys, so its outputs stay byte-identical.
    assert not {"dtype", "hidden_size", "scale_model", "screening_model_id"} & set(original)
    scaled = e.load_config(mode, e.scale_spec("qwen2.5-1.5b"))
    changed = {key for key in scaled.keys() | original.keys() if scaled.get(key) != original.get(key)}
    assert changed <= SCALE_KEYS
    assert scaled["resolved_model_path"].endswith("models/Qwen2.5-1.5B-Instruct")
    assert e.neologism_parameters(original) == 896 and e.neologism_parameters(scaled) == 1536


def test_scale_runs_refuse_rescreening(monkeypatch):
    monkeypatch.setenv("NEOLOGISM_RESCREEN", "1")
    with pytest.raises(RuntimeError, match="re-screening"):
        e.load_config("full", e.scale_spec("qwen2.5-1.5b"))


def test_screen_cache_names_the_model_that_produced_the_screen():
    scaled = e.load_config("full", e.scale_spec("qwen2.5-3b"))
    assert e.screen_signature(scaled, [])["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert e.screen_signature(e.load_config("full"), [])["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_hidden_size_check_rejects_the_wrong_checkpoint(tiny_qwen):
    config = {"model_id": "tiny", "resolved_model_path": "tiny", "hidden_size": 32}
    assert e.check_hidden_size(tiny_qwen, config) == 32
    with pytest.raises(AssertionError, match="expected 896"):
        e.check_hidden_size(tiny_qwen, {"model_id": "tiny", "resolved_model_path": "tiny"})


def test_figure_subtitle_is_unchanged_for_0_5b():
    assert e.figure_subtitle(e.load_config("full")) == "Qwen2.5-0.5B-Instruct · one 896-D embedding · frozen model"
    scaled = e.load_config("full", e.scale_spec("qwen2.5-3b"))
    assert e.figure_subtitle(scaled) == "Qwen2.5-3B-Instruct · one 2048-D embedding · frozen model"


def _write_run(root, model_id, hidden, parameters, base_words, trained_words, dtype=None):
    behavior = []
    for i, (b, t) in enumerate(zip(base_words, trained_words)):
        behavior += [
            {"seed": 0, "condition": "base", "prompt_id": f"p{i}", "words": b, "termination_reason": "eos_token"},
            {"seed": 0, "condition": "explicit", "prompt_id": f"p{i}", "words": b + 100, "termination_reason": "eos_token"},
            {"seed": 1, "condition": "trained", "prompt_id": f"p{i}", "words": t, "termination_reason": "eos_token"},
            {"seed": 2, "condition": "trained", "prompt_id": f"p{i}", "words": t + 10, "termination_reason": "eos_token"},
            {"seed": 1, "condition": "shuffled", "prompt_id": f"p{i}", "words": b, "termination_reason": "eos_token"},
        ]
    e.json_dump(root / "controls" / "behavior_results_smoke.json", behavior)
    e.json_dump(root / "reproducibility" / "smoke" / "model_info.json",
                {"model_id": model_id, "hidden_size": hidden, "model_parameters": parameters})
    e.json_dump(root / "reproducibility" / "smoke" / "config.json", {"model_id": model_id, **({"dtype": dtype} if dtype else {})})
    e.json_dump(root / "reproducibility" / "smoke" / "environment.json", {"selected_device": "cuda"})
    for seed in (1, 2):
        e.json_dump(root / "seeds" / "smoke" / f"seed_{seed}" / "genuine" / "training_summary.json",
                    {"final_loss": 0.5, "final_mean_preference_margin": 2.0})


def test_compare_reads_reference_and_scale_runs(tmp_path):
    _write_run(tmp_path / "experiment_3" / "qwen2.5-3b-bfloat16", "Qwen/Qwen2.5-3B-Instruct", 2048, 3_085_938_688,
               [200, 300], [250, 350], dtype="bfloat16")
    _write_run(tmp_path / "experiment_2", "Qwen/Qwen2.5-0.5B-Instruct", 896, 494_032_768, [60, 100], [160, 300])
    rows, figure = compare(tmp_path, "smoke")
    assert [r["run"] for r in rows] == ["experiment_2", "qwen2.5-3b-bfloat16"]
    small, large = rows
    assert (small["dtype"], large["dtype"]) == ("float32", "bfloat16")
    assert small["mean_delta_base"] == pytest.approx(155.0)  # seeds: 150 and 160
    assert large["mean_delta_base"] == pytest.approx(55.0)
    assert small["base_passes_screen"] == 1.0 and large["base_passes_screen"] == 0.0
    assert large["explicit_effect_ratio"] == pytest.approx(0.55)
    assert (tmp_path / "experiment_3" / "scale_comparison_smoke.csv").exists()
    assert figure is not None and figure.stat().st_size > 0


def test_compare_without_outputs_exits(tmp_path):
    with pytest.raises(SystemExit):
        compare(tmp_path, "full")


def test_notebook_scale_section_uses_registered_models():
    notebook = json.loads((ROOT / "notebooks" / "colab_runner.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    registry = json.loads(e.SCALE_CONFIG_PATH.read_text())["models"]
    named = set(re.findall(r'SCALE_MODEL = "([^"]+)"', code)) | set(re.findall(r'with "(qwen[^"]+)"', code))
    assert named and named <= set(registry)
    # Every scale-stage command carries the per-command model prefix.
    scale_lines = [line for line in code.splitlines() if "--stage" in line and "{SCALE}" in line]
    assert len(scale_lines) == 6
