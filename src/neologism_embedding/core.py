"""Small, explicit building blocks for one-vector neologism training."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

NEW_TOKEN = "<ENSURE_LONG>"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE='cuda' was requested, but torch.cuda.is_available() is false")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("DEVICE='mps' was requested, but torch.backends.mps.is_available() is false")
    return torch.device(requested)


def add_exactly_one_token(tokenizer) -> tuple[int, int, int]:
    """Add NEW_TOKEN and return (old tokenizer size, new size, token id)."""
    if NEW_TOKEN in tokenizer.get_vocab():
        raise ValueError(f"{NEW_TOKEN!r} already exists; this experiment requires a genuinely new token")

    from transformers import AddedToken

    old_size = len(tokenizer)
    added = tokenizer.add_tokens(
        [AddedToken(NEW_TOKEN, single_word=False, lstrip=False, rstrip=False, normalized=False)],
        special_tokens=False,
    )
    encoded = tokenizer.encode(NEW_TOKEN, add_special_tokens=False)
    if added != 1 or len(encoded) != 1:
        raise RuntimeError(f"Expected one newly added token and one id, got added={added}, ids={encoded}")
    return old_size, len(tokenizer), encoded[0]


def freeze_pretrained_model(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def random_initial_embedding(base_weight: torch.Tensor, seed: int) -> torch.Tensor:
    """Sample one vector with the base table's scalar mean and standard deviation."""
    cpu_weight = base_weight.detach().float().cpu()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.normal(
        mean=cpu_weight.mean(),
        std=cpu_weight.std(),
        size=(cpu_weight.shape[1],),
        generator=generator,
    )


class FrozenLMWithNeologism(nn.Module):
    """A frozen causal LM plus exactly one standalone trainable input vector.

    The tokenizer can grow by one without resizing the model. Every occurrence of
    neo_token_id is replaced in inputs_embeds, so no pretrained embedding row is
    writable and the LM output vocabulary remains untouched.

    trainable_dtype keeps the trainable vector in a wider dtype than a bf16/fp16
    model (a float32 master copy, cast down at the input). None keeps it in the
    model's dtype, which is a no-op for float32 models.
    """

    def __init__(
        self,
        base_model: nn.Module,
        neo_token_id: int,
        initial_embedding: torch.Tensor,
        trainable_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        freeze_pretrained_model(base_model)
        self.base_model = base_model
        self.neo_token_id = int(neo_token_id)
        embedding = base_model.get_input_embeddings().weight
        dtype = embedding.dtype if trainable_dtype is None else trainable_dtype
        self.neo_embedding = nn.Parameter(initial_embedding.to(device=embedding.device, dtype=dtype))

    def embed(self, input_ids: torch.Tensor, vector: torch.Tensor | None = None) -> torch.Tensor:
        mask = input_ids.eq(self.neo_token_id)
        # Replace the out-of-table new id before calling the frozen embedding layer.
        safe_ids = input_ids.masked_fill(mask, 0)
        inputs_embeds = self.base_model.get_input_embeddings()(safe_ids)
        # .to() returns the same tensor when the dtypes already match.
        chosen_vector = (self.neo_embedding if vector is None else vector).to(inputs_embeds.dtype)
        return torch.where(mask.unsqueeze(-1), chosen_vector.view(1, 1, -1), inputs_embeds)

    def forward(self, input_ids: torch.Tensor, vector: torch.Tensor | None = None, **kwargs):
        return self.base_model(inputs_embeds=self.embed(input_ids, vector), **kwargs)


def chat_prompt_ids(tokenizer, prompt: str) -> list[int]:
    """Render one user turn plus Qwen's assistant-generation header."""
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    return list(ids)


def render_chat_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenize_pair(tokenizer, prompt: str, response: str, device: torch.device) -> tuple[torch.Tensor, int]:
    """Tokenize a user/assistant preference pair with the Qwen chat template."""
    prompt_ids = chat_prompt_ids(tokenizer, prompt)
    full_ids = list(
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response.strip()},
            ],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("Chat-templated assistant response does not share the expected prompt prefix")
    if len(full_ids) == len(prompt_ids):
        raise ValueError("Response must tokenize to at least one token")
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    return ids, len(prompt_ids)


def response_log_probability(
    wrapper: FrozenLMWithNeologism,
    input_ids: torch.Tensor,
    prompt_length: int,
    vector: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute sum_t log p(y_t | x, y_<t), excluding all prompt targets."""
    attention_mask = torch.ones_like(input_ids)
    logits = wrapper(input_ids, vector=vector, attention_mask=attention_mask, use_cache=False).logits
    response_targets = input_ids[:, prompt_length:]
    # Logit at position prompt_length-1 predicts the first response token.
    response_logits = logits[:, prompt_length - 1 : -1, :]
    if int(response_targets.max()) >= response_logits.shape[-1]:
        raise RuntimeError("A response contains an id outside the base model output vocabulary")
    token_logps = F.log_softmax(response_logits.float(), dim=-1).gather(
        dim=-1, index=response_targets.unsqueeze(-1)
    )
    return token_logps.squeeze(-1).sum()


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reward_chosen = policy_chosen - reference_chosen
    reward_rejected = policy_rejected - reference_rejected
    preference_margin = reward_chosen - reward_rejected
    return -F.logsigmoid(beta * preference_margin), preference_margin


def parameter_checksums(model: nn.Module) -> dict[str, str]:
    """SHA-256 every pretrained parameter without keeping a second model in RAM."""
    checksums: dict[str, str] = {}
    for name, parameter in model.named_parameters():
        raw = parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        checksums[name] = hashlib.sha256(raw).hexdigest()
    return checksums


def assert_checksums_equal(before: dict[str, str], after: dict[str, str]) -> None:
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    if changed:
        raise AssertionError(f"Pretrained parameters changed: {changed}")


@torch.no_grad()
def logits_for_plain_prompt(wrapper: FrozenLMWithNeologism, tokenizer, prompt: str, device: torch.device):
    ids = torch.tensor([chat_prompt_ids(tokenizer, prompt)], dtype=torch.long, device=device)
    return wrapper(ids, use_cache=False).logits.detach().float().cpu()


@torch.no_grad()
def greedy_generate(
    wrapper: FrozenLMWithNeologism,
    tokenizer,
    prompt: str,
    vector: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, list[int], str]:
    """Deterministic decoding kept deliberately explicit and batch-size one."""
    prompt_ids = chat_prompt_ids(tokenizer, prompt)
    all_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(all_ids)
    generated: list[int] = []
    eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    outputs = wrapper(all_ids, vector=vector, attention_mask=attention_mask, use_cache=True)
    termination_reason = "max_new_tokens"
    for step in range(max_new_tokens):
        next_id = int(outputs.logits[0, -1].argmax())
        generated.append(next_id)
        if next_id in eos_ids:
            termination_reason = "eos_token"
            break
        if step + 1 < max_new_tokens:
            next_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1
            )
            outputs = wrapper(
                next_ids,
                vector=vector,
                attention_mask=attention_mask,
                past_key_values=outputs.past_key_values,
                use_cache=True,
            )
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), generated, termination_reason


def load_preferences(path: Path) -> list[dict[str, str]]:
    records = json.loads(path.read_text())
    if len(records) != 3:
        raise ValueError(f"This proof of concept requires exactly 3 examples, found {len(records)}")
    required = {"prompt", "chosen", "rejected"}
    if any(set(record) != required for record in records):
        raise ValueError(f"Every preference record must contain exactly {sorted(required)}")
    return records


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
