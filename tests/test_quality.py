import json

from neologism_embedding.quality import analyze, effective_length, is_clean, repeated_trigram_fraction


def test_effective_length_equals_word_count_without_repeats_and_drops_loops():
    text = "Light scatters off small molecules in the air."
    assert effective_length(text) == 8
    loop = "the cache was fast. " * 10
    assert effective_length(loop) < 10 < len(loop.split())
    assert effective_length("") == 0 and effective_length("two words") == 2


def test_clean_requires_eos_and_low_repetition():
    row = {"termination_reason": "eos_token", "output_text": "a b c d e f"}
    assert repeated_trigram_fraction(row["output_text"]) == 0.0
    assert is_clean(row, 0.1)
    assert not is_clean({**row, "termination_reason": "max_new_tokens"}, 0.1)
    assert not is_clean({**row, "output_text": "a b c a b c a b c"}, 0.1)


def test_analyze_uses_base_threshold_and_reports_deltas(tmp_path):
    def row(condition, seed, pid, text, end="eos_token"):
        return {"condition": condition, "seed": seed, "prompt_id": pid, "output_text": text, "termination_reason": end}

    behavior = [
        row("base", 0, "p1", "one two three four"), row("base", 0, "p2", "alpha beta gamma delta"),
        row("explicit", 0, "p1", "one two three four five six seven eight"),
        row("explicit", 0, "p2", "alpha beta gamma delta epsilon zeta eta theta"),
        row("untrained", 1, "p1", "one two three four"), row("untrained", 1, "p2", "alpha beta gamma delta"),
        row("shuffled", 1, "p1", "one two three four five"), row("shuffled", 1, "p2", "alpha beta gamma delta"),
        row("trained", 1, "p1", "x y z " * 20, "max_new_tokens"), row("trained", 1, "p2", "a b c d e f g h i j"),
    ]
    (tmp_path / "controls").mkdir()
    (tmp_path / "controls" / "behavior_results_full.json").write_text(json.dumps(behavior))
    summary, threshold = analyze(tmp_path, "full", samples=50)
    assert threshold == 0.0
    trained = next(s for s in summary if s["condition"] == "trained" and s["seed"] == "all")
    assert trained["mean_raw_words"] == 35 and trained["mean_effective_words"] == 7.5  # (3 distinct + 2) and 10
    assert trained["proportion_clean"] == 0.5 and trained["proportion_hit_cap"] == 0.5
    assert trained["mean_delta_raw_clean_only"] == 6
    assert (tmp_path / "quality" / "quality_summary_full.csv").exists()
