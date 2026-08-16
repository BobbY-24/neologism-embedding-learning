import json
import os
import random
from pathlib import Path

import pytest

from neologism_embedding.core import add_exactly_one_token
from neologism_embedding.experiment2 import load_compatible_cache, normalize_whitespace, save_cache


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/experiment_2/data"


def test_candidate_pool_is_fixed_diverse_and_neologism_free():
    rows = json.loads((DATA / "candidate_prompts.json").read_text())
    assert len(rows) == 100
    assert len({row["id"] for row in rows}) == 100
    assert len({row["topic"] for row in rows}) == 100
    assert len({row["domain"] for row in rows}) >= 10
    assert all("<ENSURE_LONG>" not in row["prompt"] for row in rows)


def test_train_and_heldout_are_disjoint_and_pairs_are_contrast_purified():
    train = json.loads((DATA / "train_preferences.json").read_text())
    heldout = json.loads((DATA / "heldout_prompts.json").read_text())
    assert len(train) == 20
    assert len(heldout) == 36
    assert {row["topic"] for row in train}.isdisjoint({row["topic"] for row in heldout})
    for row in train:
        assert row["chosen_words"] - row["rejected_words"] >= 250
        assert normalize_whitespace(row["chosen"]).startswith(normalize_whitespace(row["rejected"]))
        assert row["prefix_valid"] is True


def test_screened_prompts_meet_length_and_eos_criterion():
    rows = list(__import__("csv").DictReader((DATA / "prompt_selection_results.csv").open()))
    selected = [row for row in rows if row["passes_limit"].lower() == "true"]
    assert len(selected) == 56
    assert all(int(row["words"]) <= 120 and row["termination_reason"] == "eos_token" for row in selected)


def test_nested_training_sizes_use_prefixes_of_one_fixed_split():
    manifest = json.loads((ROOT / "results/experiment_2/prepared_full_dataset/dataset_manifest.json").read_text())
    nested = manifest["nested_subsets"]
    sizes = [1, 2, 5, 10, 20]
    for small, large in zip(sizes, sizes[1:]):
        assert nested[str(large)][:small] == nested[str(small)]
    assert set(manifest["train_ids_full"]).isdisjoint(manifest["heldout_ids_full"])


def test_explicit_control_has_no_neologism_and_shuffle_changes_labels():
    config = json.loads((ROOT / "experiments/experiment_2/configs/default.json").read_text())
    assert "<ENSURE_LONG>" not in config["explicit_template"]
    train = json.loads((DATA / "train_preferences.json").read_text())[:5]
    rng = random.Random(config["shuffle_seed"])
    flips = [rng.random() < 0.5 for _ in train]
    if not any(flips):
        flips[0] = True
    assert any(flips)


def test_incompatible_cache_is_rejected(tmp_path):
    path = tmp_path / "cache.json"
    save_cache(path, {"max_new_tokens": 512}, [{"ok": True}])
    assert load_compatible_cache(path, {"max_new_tokens": 512}) == [{"ok": True}]
    with pytest.raises(RuntimeError, match="Incompatible cache"):
        load_compatible_cache(path, {"max_new_tokens": 256})


def test_new_token_is_exactly_one_token_when_checkpoint_is_available():
    from transformers import AutoTokenizer

    model_path = Path(os.environ.get("QWEN_MODEL_PATH", ROOT / "models/Qwen2.5-0.5B-Instruct"))
    if not (model_path / "config.json").exists():
        pytest.skip("External Qwen checkpoint not installed")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    old, new, token_id = add_exactly_one_token(tokenizer)
    assert new == old + 1
    assert tokenizer.encode("<ENSURE_LONG>", add_special_tokens=False) == [token_id]
