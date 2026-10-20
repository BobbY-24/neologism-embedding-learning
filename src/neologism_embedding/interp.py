"""Interpretability analyses of the learned <ENSURE_LONG> embedding E_w.

1. Nearest vocabulary neighbours of E_w, of its initialization and of the update
   E_w - E_w^(0), in the input-embedding and unembedding matrices.
2. Logit lens: each layer's residual stream decoded through the final norm and
   unembedding, (a) at the token position and (b) at the position that decides
   whether to stop after the base model's concise answer.
3. Activation patching at the token position: run with the untrained vector,
   restore one layer's residual-stream output at the token position from the
   trained run, and measure how much of the trained effect on the stop log-prob
   comes back.

The behavioral metric throughout is the teacher-forced log-probability of the
end-of-turn token right after the base model's own concise answer. The trained
token should lower it (the model prefers to keep going).
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Callable

import torch
import torch.nn.functional as F

from .core import NEW_TOKEN, json_dump, tokenize_pair
from .experiment1 import write_csv
from .experiment2 import OUTPUT_ROOT, load_config, runtime, seed_output_dir, stage_dataset, stage_screen

EMBEDDING_LAYER = -1  # patching "layer -1" means swapping the input vector itself


def decoder_layers(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        module = model
        try:
            for attr in path.split("."):
                module = getattr(module, attr)
        except AttributeError:
            continue
        return module
    raise ValueError(f"Cannot find decoder layers on {type(model).__name__}")


def final_norm(model):
    for path in ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm"):
        module = model
        try:
            for attr in path.split("."):
                module = getattr(module, attr)
        except AttributeError:
            continue
        return module
    raise ValueError(f"Cannot find the final norm on {type(model).__name__}")


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output, hidden):
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


def vocab_neighbours(matrix: torch.Tensor, vector: torch.Tensor, k: int = 20, limit: int | None = None):
    """Top-k rows of matrix by cosine similarity to vector; returns (ids, scores)."""
    rows = matrix[:limit].detach().float().cpu()
    scores = F.cosine_similarity(rows, vector.detach().float().cpu().view(1, -1), dim=1)
    values, ids = scores.topk(k)
    return ids.tolist(), values.tolist()


@torch.no_grad()
def residual_stream(wrapper, input_ids, vector):
    """Return (logits, residuals) with residuals[0] = embeddings, residuals[i + 1] = output of layer i."""
    layers = decoder_layers(wrapper.base_model)
    captured = []
    handles = [layer.register_forward_hook(lambda _m, _i, out: captured.append(_hidden(out)[0].detach())) for layer in layers]
    try:
        logits = wrapper(input_ids, vector=vector, use_cache=False).logits[0].float()
    finally:
        for handle in handles:
            handle.remove()
    embeddings = wrapper.embed(input_ids, vector)[0].detach()
    return logits, torch.stack([embeddings, *captured])


@torch.no_grad()
def logit_lens(model, residuals):
    """Decode residuals [..., hidden] into log-probabilities [..., vocab]."""
    normed = final_norm(model)(residuals.to(model.get_output_embeddings().weight.dtype))
    return F.log_softmax(model.get_output_embeddings()(normed).float(), dim=-1)


def stop_metric(logits, stop_index: int, stop_id: int) -> float:
    """log p(stop_id at stop_index); logits[stop_index - 1] predicts that token."""
    return float(F.log_softmax(logits[stop_index - 1].float(), dim=-1)[stop_id])


@torch.no_grad()
def patched_logits(wrapper, input_ids, corrupt_vector, clean_vector, clean_residuals, position, layer):
    """Corrupt run with the residual output of `layer` at `position` taken from the clean run."""
    if layer == EMBEDDING_LAYER:
        # The vector only enters at the token position, so this equals the clean run.
        return wrapper(input_ids, vector=clean_vector, use_cache=False).logits[0].float()
    replacement = clean_residuals[layer + 1, position]

    def hook(_module, _inputs, output):
        hidden = _hidden(output).clone()
        hidden[0, position] = replacement.to(hidden.dtype)
        return _replace_hidden(output, hidden)

    handle = decoder_layers(wrapper.base_model)[layer].register_forward_hook(hook)
    try:
        return wrapper(input_ids, vector=corrupt_vector, use_cache=False).logits[0].float()
    finally:
        handle.remove()


def patching_sweep(wrapper, input_ids, position, clean_vector, corrupt_vector, metric: Callable[[torch.Tensor], float]):
    """Metric for clean, corrupt and each single-layer patch at `position` (layers -1..L-1)."""
    clean_logits, clean_residuals = residual_stream(wrapper, input_ids, clean_vector)
    corrupt_logits, _ = residual_stream(wrapper, input_ids, corrupt_vector)
    n_layers = len(decoder_layers(wrapper.base_model))
    patched = {
        layer: metric(patched_logits(wrapper, input_ids, corrupt_vector, clean_vector, clean_residuals, position, layer))
        for layer in range(EMBEDDING_LAYER, n_layers)
    }
    return metric(clean_logits), metric(corrupt_logits), patched


def pooled_restoration(rows):
    """Per layer: sum(patched - corrupt) / sum(clean - corrupt) over prompts."""
    layers = sorted({r["layer"] for r in rows})
    out = {}
    for layer in layers:
        selected = [r for r in rows if r["layer"] == layer]
        denominator = sum(r["metric_clean"] - r["metric_corrupt"] for r in selected)
        numerator = sum(r["metric_patched"] - r["metric_corrupt"] for r in selected)
        out[layer] = numerator / denominator if abs(denominator) > 1e-9 else float("nan")
    return out


def run(config, limit_prompts=None, k=20, lens_top=5):
    tokenizer, model, wrapper, device, neo_id, old_vocab = runtime(config)
    _, heldout = stage_dataset(config)
    heldout = heldout[:limit_prompts] if limit_prompts else heldout
    screen = {row["id"]: row for row in stage_screen(config)}
    stop_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    seeds = config["mode_config"]["seeds"]
    out = OUTPUT_ROOT / "interp" / config["mode"]
    embed_matrix = model.get_input_embeddings().weight
    unembed_matrix = model.get_output_embeddings().weight
    tied = embed_matrix.data_ptr() == unembed_matrix.data_ptr()
    spaces = {"input_embedding": embed_matrix} if tied else {"input_embedding": embed_matrix, "unembedding": unembed_matrix}
    to_device = lambda v: v.to(device=device, dtype=wrapper.neo_embedding.dtype)  # noqa: E731

    vectors = {}
    for seed in seeds:
        artifact = torch.load(seed_output_dir(config, seed, "genuine") / "embedding.pt", map_location="cpu", weights_only=True)
        vectors[seed] = (artifact["initial_embedding"].float(), artifact["learned_embedding"].float())

    neighbours = []
    directions = {}
    for seed, (initial, learned) in vectors.items():
        directions[(seed, "initial")] = initial
        directions[(seed, "learned")] = learned
        directions[(seed, "update")] = learned - initial
    directions[("all", "mean_update")] = torch.stack([l - i for i, l in vectors.values()]).mean(0)
    for (seed, which), vector in directions.items():
        for space, matrix in spaces.items():
            ids, scores = vocab_neighbours(matrix, vector, k, old_vocab)
            for rank, (token_id, score) in enumerate(zip(ids, scores), start=1):
                neighbours.append({
                    "seed": seed, "vector": which, "space": space, "rank": rank, "token_id": token_id,
                    "token": tokenizer.convert_ids_to_tokens(token_id), "decoded": tokenizer.decode([token_id]),
                    "cosine_similarity": score,
                })

    lens_tokens, lens_stop, deltas, patch_rows = [], [], [], []
    for seed, (initial, learned) in vectors.items():
        clean, corrupt = to_device(learned), to_device(initial)
        for item in heldout:
            answer = screen[item["id"]]["output_text"]
            ids, prompt_length = tokenize_pair(tokenizer, config["primary_template"].format(prompt=item["prompt"]), answer, device)
            plain_ids, _ = tokenize_pair(tokenizer, item["prompt"], answer, device)
            token_position = ids[0].tolist().index(neo_id)
            stop_index = max(i for i, t in enumerate(ids[0].tolist()) if t == stop_id)
            plain_stop_index = max(i for i, t in enumerate(plain_ids[0].tolist()) if t == stop_id)
            metric = lambda logits: stop_metric(logits, stop_index, stop_id)  # noqa: E731

            plain_logits, plain_residuals = residual_stream(wrapper, plain_ids, corrupt)
            plain_lens = logit_lens(model, plain_residuals[:, plain_stop_index - 1])[:, stop_id]
            runs = {}
            for condition, vector in (("trained", clean), ("untrained", corrupt)):
                logits, residuals = residual_stream(wrapper, ids, vector)
                runs[condition] = residuals
                at_token = logit_lens(model, residuals[:, token_position])
                at_stop = logit_lens(model, residuals[:, stop_index - 1])
                for layer in range(residuals.shape[0]):
                    values, top_ids = at_token[layer].topk(lens_top)
                    for rank, (value, token_id) in enumerate(zip(values.tolist(), top_ids.tolist()), start=1):
                        lens_tokens.append({
                            "seed": seed, "prompt_id": item["id"], "condition": condition, "layer": layer - 1,
                            "rank": rank, "token": tokenizer.decode([token_id]), "log_probability": value,
                        })
                    lens_stop.append({
                        "seed": seed, "prompt_id": item["id"], "condition": condition, "layer": layer - 1,
                        "stop_log_probability": float(at_stop[layer, stop_id]),
                    })
            for layer in range(plain_residuals.shape[0]):
                lens_stop.append({
                    "seed": seed, "prompt_id": item["id"], "condition": "base_no_token", "layer": layer - 1,
                    "stop_log_probability": float(plain_lens[layer]),
                })
            for layer in range(runs["trained"].shape[0]):
                for name, position in (("token", token_position), ("last_prompt", prompt_length - 1), ("stop_decision", stop_index - 1)):
                    trained_h, untrained_h = runs["trained"][layer, position].float(), runs["untrained"][layer, position].float()
                    deltas.append({
                        "seed": seed, "prompt_id": item["id"], "layer": layer - 1, "position": name,
                        "relative_residual_change": float((trained_h - untrained_h).norm() / untrained_h.norm()),
                        "cosine_trained_untrained": float(F.cosine_similarity(trained_h[None], untrained_h[None])),
                    })

            metric_clean, metric_corrupt, patched = patching_sweep(wrapper, ids, token_position, clean, corrupt, metric)
            for layer, value in patched.items():
                patch_rows.append({
                    "seed": seed, "prompt_id": item["id"], "layer": layer, "token_position": token_position,
                    "metric_clean": metric_clean, "metric_corrupt": metric_corrupt, "metric_patched": value,
                    "metric_base_no_token": stop_metric(plain_logits, plain_stop_index, stop_id),
                })
            print(f"interp seed={seed} {item['id']}: stop logp trained={metric_clean:.3f} untrained={metric_corrupt:.3f}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "neighbours.csv", neighbours)
    write_csv(out / "logit_lens_token_position.csv", lens_tokens)
    write_csv(out / "logit_lens_stop.csv", lens_stop)
    write_csv(out / "residual_change.csv", deltas)
    write_csv(out / "patching.csv", patch_rows)
    restoration = {str(seed): pooled_restoration([r for r in patch_rows if r["seed"] == seed]) for seed in seeds}
    restoration["pooled"] = pooled_restoration(patch_rows)
    clean_effects = [r["metric_clean"] - r["metric_corrupt"] for r in patch_rows if r["layer"] == EMBEDDING_LAYER]
    json_dump(out / "interp_summary.json", {
        "mode": config["mode"], "seeds": seeds, "prompt_ids": [item["id"] for item in heldout],
        "metric": "teacher-forced log p(<|im_end|>) right after the base model's concise answer",
        "embedding_unembedding_tied": tied,
        "mean_stop_logprob_effect_trained_minus_untrained": statistics.mean(clean_effects),
        "fraction_prompts_trained_lowers_stop": sum(e < 0 for e in clean_effects) / len(clean_effects),
        "patching_restoration_by_layer": restoration,
        "token": NEW_TOKEN,
    })
    print(f"Wrote interpretability outputs to {out}")


def main():
    parser = argparse.ArgumentParser(description="Interpretability analyses of the learned neologism embedding")
    parser.add_argument("--mode", choices=["quick", "full", "smoke"], default="full")
    parser.add_argument("--limit-prompts", type=int, default=None)
    args = parser.parse_args()
    run(load_config(args.mode), limit_prompts=args.limit_prompts)


if __name__ == "__main__":
    main()
