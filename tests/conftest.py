import pytest
import torch


@pytest.fixture
def tiny_qwen():
    """A randomly initialised 3-layer Qwen2 so model-level tests run on CPU without downloads."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(config).eval()
