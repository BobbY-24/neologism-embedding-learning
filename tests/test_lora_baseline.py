import json
import math

import torch

from neologism_embedding.core import FrozenLMWithNeologism
from neologism_embedding.lora_baseline import (
    ARM_CONFIG_PATH,
    accounting,
    adapter_free_references,
    attach_lora,
    base_checksums,
    expected_lora_parameters,
    lora_state,
    mean_dpo_state,
    train_lora,
)

NEO_ID = 64


def arm(**overrides):
    return {"rank": 1, "alpha": 2, "dropout": 0.0, "target_modules": ["v_proj"], "layers": [0],
            "learning_rate": 0.05, **overrides}


def test_parameter_count_matches_formula_for_layer_subset_and_all_layers(tiny_qwen):
    # Tiny model: hidden 32, 2 kv heads x head_dim 8 -> v_proj is 32 -> 16.
    assert expected_lora_parameters(tiny_qwen, arm()) == 1 * (32 + 16)
    full = arm(rank=8, target_modules=["q_proj", "v_proj"], layers=None)
    assert expected_lora_parameters(tiny_qwen, full) == 3 * 8 * ((32 + 32) + (32 + 16))
    peft_model = attach_lora(tiny_qwen, full)
    assert accounting(peft_model)["trainable_parameters"] == 3 * 8 * ((32 + 32) + (32 + 16))


def test_matched_arm_on_real_qwen_shapes_is_1024_parameters():
    # Qwen2.5-0.5B: hidden 896, v_proj 896 -> 128; rank 1 on one layer.
    spec = json.loads(ARM_CONFIG_PATH.read_text())["arms"]["lora_matched"]
    assert spec["rank"] * (896 + 128) * len(spec["layers"]) == 1024


def test_lora_dpo_trains_only_adapters_and_starts_at_ln2(tiny_qwen):
    torch.manual_seed(1)
    wrapper = FrozenLMWithNeologism(tiny_qwen, NEO_ID, torch.randn(32))
    wrapper.neo_embedding.requires_grad_(False)
    vector = wrapper.neo_embedding.detach().clone()
    before = base_checksums(tiny_qwen)
    plain = torch.tensor([[1, 2, 3]])
    base_logits = wrapper(plain).logits.detach().clone()

    peft_model = attach_lora(tiny_qwen, arm(target_modules=["q_proj", "v_proj"], layers=None, rank=2))
    assert all("lora_" in n for n, p in peft_model.named_parameters() if p.requires_grad)
    pairs = [
        {"chosen": (torch.tensor([[1, NEO_ID, 3, 10, 11, 12, 13]]), 3), "rejected": (torch.tensor([[1, NEO_ID, 3, 10, 7]]), 3)},
        {"chosen": (torch.tensor([[2, NEO_ID, 4, 20, 21, 22, 23]]), 3), "rejected": (torch.tensor([[2, NEO_ID, 4, 20, 7]]), 3)},
    ]
    references = adapter_free_references(peft_model, wrapper, pairs, vector)
    initial_loss, _ = mean_dpo_state(wrapper, pairs, references, vector, beta=0.1)
    assert math.isclose(initial_loss, math.log(2), abs_tol=1e-6)  # LoRA B starts at zero

    start = lora_state(peft_model)
    metrics = train_lora(peft_model, wrapper, pairs, references, vector, epochs=5, lr=0.05, beta=0.1, label="t")
    final_loss, final_margin = mean_dpo_state(wrapper, pairs, references, vector, beta=0.1)
    assert len(metrics) == 10
    assert final_loss < initial_loss and final_margin > 0
    assert any(not torch.equal(start[n], v) for n, v in lora_state(peft_model).items())
    assert base_checksums(tiny_qwen) == before
    assert torch.equal(wrapper.neo_embedding.detach(), vector)
    with peft_model.disable_adapter():
        assert torch.allclose(wrapper(plain).logits, base_logits, atol=1e-6)
    # Unlike the neologism, adapters change outputs on prompts without the token.
    assert not torch.allclose(wrapper(plain).logits, base_logits, atol=1e-6)
