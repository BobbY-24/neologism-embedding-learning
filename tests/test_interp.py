import math

import torch

from neologism_embedding.core import FrozenLMWithNeologism
from neologism_embedding.interp import (
    EMBEDDING_LAYER,
    decoder_layers,
    logit_lens,
    patching_sweep,
    pooled_restoration,
    residual_stream,
    stop_metric,
    vocab_neighbours,
)

NEO_ID = 64  # one past the tiny vocabulary, like the real added token
STOP_ID = 7


def make_wrapper(model):
    return FrozenLMWithNeologism(model, NEO_ID, torch.randn(model.config.hidden_size))


def test_logit_lens_at_last_layer_equals_model_output(tiny_qwen):
    wrapper = make_wrapper(tiny_qwen)
    ids = torch.tensor([[1, NEO_ID, 3, 4, 5, STOP_ID]])
    logits, residuals = residual_stream(wrapper, ids, wrapper.neo_embedding.detach())
    assert residuals.shape == (len(decoder_layers(tiny_qwen)) + 1, 6, 32)
    lens = logit_lens(tiny_qwen, residuals)
    assert torch.allclose(lens[-1], torch.log_softmax(logits, dim=-1), atol=1e-5)
    # Layer -1 of the residual stack is the embedding output, with E_w at the token position.
    assert torch.equal(residuals[0, 1], wrapper.neo_embedding.detach())


def test_patching_endpoints_and_hook_cleanup(tiny_qwen):
    torch.manual_seed(3)
    wrapper = make_wrapper(tiny_qwen)
    clean, corrupt = torch.randn(32) * 3, torch.randn(32) * 3
    ids = torch.tensor([[2, NEO_ID, 3, 4, 5, 6, STOP_ID]])
    stop_index = 6
    metric = lambda logits: stop_metric(logits, stop_index, STOP_ID)  # noqa: E731
    m_clean, m_corrupt, patched = patching_sweep(wrapper, ids, 1, clean, corrupt, metric)
    assert m_clean != m_corrupt
    assert set(patched) == {-1, 0, 1, 2}
    # Swapping the input vector reproduces the clean run exactly.
    assert math.isclose(patched[EMBEDDING_LAYER], m_clean, abs_tol=1e-5)
    # The last layer's output at the token position only feeds that position's logits.
    assert math.isclose(patched[2], m_corrupt, abs_tol=1e-5)
    assert all(not layer._forward_hooks for layer in decoder_layers(tiny_qwen))
    rows = [
        {"layer": layer, "metric_clean": m_clean, "metric_corrupt": m_corrupt, "metric_patched": value}
        for layer, value in patched.items()
    ]
    restored = pooled_restoration(rows)
    assert math.isclose(restored[-1], 1.0, abs_tol=1e-4) and math.isclose(restored[2], 0.0, abs_tol=1e-4)


def test_vocab_neighbours_finds_exact_row_and_respects_limit():
    matrix = torch.randn(20, 8)
    ids, scores = vocab_neighbours(matrix, matrix[5], k=3)
    assert ids[0] == 5 and math.isclose(scores[0], 1.0, abs_tol=1e-6)
    ids, _ = vocab_neighbours(matrix, matrix[15], k=3, limit=10)
    assert all(i < 10 for i in ids)


def test_stop_metric_reads_the_preceding_position():
    logits = torch.zeros(4, 10)
    logits[2, 7] = 5.0
    assert stop_metric(logits, 3, 7) > stop_metric(logits, 2, 7)
