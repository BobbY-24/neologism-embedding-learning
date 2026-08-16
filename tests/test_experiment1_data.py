import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_1_has_exactly_three_preference_pairs():
    records = json.loads((ROOT / "experiments/experiment_1/data/preferences.json").read_text())
    assert len(records) == 3
    for record in records:
        assert set(record) == {"prompt", "chosen", "rejected"}
        assert record["prompt"].count("<ENSURE_LONG>") == 1
        assert 250 <= len(record["chosen"].split()) <= 350
        assert 50 <= len(record["rejected"].split()) <= 80
