from types import SimpleNamespace

import torch
import torch.nn as nn

from neologism_embedding.core import FrozenLMWithNeologism, dpo_loss, response_log_probability


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 17, hidden: int = 896):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.projection = nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(hidden_size=hidden)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, **_kwargs):
        return SimpleNamespace(logits=self.projection(inputs_embeds.cumsum(dim=1)))


def test_only_one_896_dimensional_vector_is_trainable_and_receives_gradient():
    torch.manual_seed(0)
    model = TinyLM()
    wrapper = FrozenLMWithNeologism(model, 17, torch.randn(896))
    assert sum(p.numel() for p in wrapper.parameters() if p.requires_grad) == 896
    assert all(not p.requires_grad for p in model.parameters())

    loss = -response_log_probability(wrapper, torch.tensor([[1, 17, 3, 4]]), 2)
    loss.backward()
    assert wrapper.neo_embedding.grad is not None
    assert float(wrapper.neo_embedding.grad.norm()) > 0
    assert all(p.grad is None for p in model.parameters())


def test_plain_prompt_logits_and_pretrained_parameters_are_invariant():
    torch.manual_seed(1)
    model = TinyLM()
    wrapper = FrozenLMWithNeologism(model, 17, torch.randn(896))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    plain = torch.tensor([[1, 2, 3]])
    logits_before = wrapper(plain).logits.detach().clone()

    optimizer = torch.optim.AdamW([wrapper.neo_embedding], lr=0.01, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    (-response_log_probability(wrapper, torch.tensor([[1, 17, 3, 4]]), 2)).backward()
    optimizer.step()

    assert all(torch.equal(before[name], p.detach()) for name, p in model.named_parameters())
    assert torch.equal(logits_before, wrapper(plain).logits.detach())


def test_dpo_initial_loss_is_log_two():
    value = torch.tensor(-10.0)
    loss, margin = dpo_loss(value, value, value, value, beta=0.1)
    assert torch.isclose(loss, torch.tensor(0.69314718), atol=1e-7)
    assert float(margin) == 0.0
