"""Build text-derived directions and edit official BAGEL inference."""
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Literal

import torch

from bagel_space.capture import Capture

Selection = int | Sequence[int] | slice | Literal["first", "last", "all"]
Reduction = Literal["none", "sum", "mean", "average"] | Callable
REDUCERS = {"sum": torch.sum, "mean": torch.mean, "average": torch.mean}
DIRECTIONS = {"difference": torch.sub}


def _indices(selection: Selection, length: int) -> list[int]:
    if isinstance(selection, str):
        return {"all": list(range(length)), "first": [0], "last": [length - 1]}[selection]
    if isinstance(selection, slice):
        return list(range(length))[selection]
    selected = [selection] if isinstance(selection, int) else list(selection)
    return list(dict.fromkeys(list(range(length))[index] for index in selected))


def reduce_states(states: torch.Tensor, reduction: Reduction, dim=0) -> torch.Tensor:
    """Pool one observation axis, preserving hidden coordinates."""
    if reduction == "none":
        return states.clone()
    reducer = reduction if callable(reduction) else REDUCERS[reduction]
    return reducer(states, dim=dim)


def repe_direction(differences: torch.Tensor) -> torch.Tensor:
    """Return the first centered principal component of [pairs, hidden] differences."""
    centered = differences - differences.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    orientation = torch.dot(direction, differences.mean(dim=0))
    largest = direction[direction.abs().argmax()]
    return direction * torch.where(orientation == 0, largest, orientation).sign()


def fit_iti_direction(
    positive: torch.Tensor, negative: torch.Tensor, *, l2: float = 1.0, max_iter: int = 100,
) -> dict:
    """Fit L2 logistic regression on CPU and scale its direction by projected activation std."""
    states = torch.cat([positive, negative]).detach().cpu().double()
    labels = torch.cat([torch.ones(len(positive)), torch.zeros(len(negative))]).double()
    center = states.mean(dim=0)
    scale = (states - center).square().mean().sqrt()
    inputs = (states - center) / scale
    classifier = torch.nn.Linear(states.shape[1], 1, dtype=torch.float64)
    torch.nn.init.zeros_(classifier.weight)
    torch.nn.init.zeros_(classifier.bias)
    optimizer = torch.optim.LBFGS(
        classifier.parameters(), max_iter=max_iter, line_search_fn="strong_wolfe",
    )

    def objective():
        optimizer.zero_grad()
        logits = classifier(inputs).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss = loss + 0.5 * l2 * classifier.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(objective)
    weight = classifier.weight.detach()[0] / scale
    bias = classifier.bias.detach()[0] - torch.dot(weight, center)
    unit = weight / weight.norm()
    projection_std = (states @ unit).std(correction=0)
    accuracy = ((states @ weight + bias >= 0) == labels.bool()).double().mean()
    return dict(
        vector=(unit * projection_std).float(), weight=weight.float(), bias=bias.float(),
        projection_std=projection_std.item(), training_accuracy=accuracy.item(),
    )


def prepare_pair(adapter, pair_pos: str, pair_neg: str, *, alignment="none") -> dict:
    """Tokenize a raw pair with shared BOS, no EOS, and optional space padding."""
    token_ids = {
        side: [adapter.token_ids["bos_token_id"]]
        + adapter.tokenizer(text, add_special_tokens=False)["input_ids"]
        for side, text in (("positive", pair_pos), ("negative", pair_neg))
    }
    padding = dict(positive=0, negative=0)
    if alignment == "right_pad":
        space, = adapter.tokenizer(" ", add_special_tokens=False)["input_ids"]
        length = max(map(len, token_ids.values()))
        for side, ids in token_ids.items():
            padding[side] = length - len(ids)
            ids.extend([space] * padding[side])
    return dict(
        token_ids=token_ids, padding=padding, alignment=alignment,
        pair_pos=pair_pos, pair_neg=pair_neg,
    )


@torch.no_grad()
def capture_pair(adapter, pair: dict, *, layers: Selection = "all") -> dict:
    """Run two independent text prefills through BAGEL's understanding route."""
    selected = _indices(layers, len(adapter.layers))
    states = {}
    for side, ids in pair["token_ids"].items():
        length = len(ids)
        positions = torch.arange(length, device=adapter.device)
        context = adapter.inferencer.init_gen_context()
        with Capture(adapter.layers, selected, "text") as captured, adapter.autocast():
            adapter.model.forward_cache_update_text(
                past_key_values=context["past_key_values"],
                packed_text_ids=torch.tensor(ids, device=adapter.device),
                packed_text_position_ids=positions,
                text_token_lens=torch.tensor([length], dtype=torch.int32, device=adapter.device),
                packed_text_indexes=positions,
                packed_key_value_indexes=torch.empty(0, dtype=torch.long, device=adapter.device),
                key_values_lens=torch.tensor([0], dtype=torch.int32, device=adapter.device),
            )
        states[side] = {layer: hidden.float() for layer, hidden in captured.states.items()}
    return dict(pair=pair, layers=selected, **states)


def build_steering_vector(
    captured: dict, *, tokens: Selection = "last", token_reduction: Reduction = "mean",
    layer_reduction: Reduction = "none", direction_method="difference",
) -> dict:
    """Select tokens, pool each side, and subtract independently at each layer."""
    positive, negative, differences, vectors = {}, {}, {}, {}
    direction = direction_method if callable(direction_method) else DIRECTIONS[direction_method]
    for layer in captured["layers"]:
        positive[layer] = captured["positive"][layer][
            _indices(tokens, len(captured["positive"][layer]))
        ]
        negative[layer] = captured["negative"][layer][
            _indices(tokens, len(captured["negative"][layer]))
        ]
        positive[layer] = reduce_states(positive[layer], token_reduction)
        negative[layer] = reduce_states(negative[layer], token_reduction)
        differences[layer] = positive[layer] - negative[layer]
        vectors[layer] = direction(positive[layer], negative[layer])
    combined = None
    if layer_reduction != "none":
        combined = reduce_states(torch.stack(list(vectors.values())), layer_reduction)
    return dict(
        vectors=vectors, positive=positive, negative=negative,
        differences=differences, combined=combined,
        metadata=dict(
            **captured["pair"], layers=captured["layers"], tokens=tokens,
            token_reduction=getattr(token_reduction, "__name__", token_reduction),
            layer_reduction=getattr(layer_reduction, "__name__", layer_reduction),
            direction_method=getattr(direction_method, "__name__", direction_method),
            difference="positive - negative",
            capture="understanding block output before final norm",
        ),
    )


def make_steering_vector(
    adapter, *, pair_pos: str, pair_neg: str, alignment="none",
    tokens: Selection = "last", layers: Selection = "all",
    token_reduction: Reduction = "mean", layer_reduction: Reduction = "none",
    direction_method="difference",
) -> dict:
    """Build one pair's direction using the same stepwise helpers."""
    pair = prepare_pair(adapter, pair_pos, pair_neg, alignment=alignment)
    captured = capture_pair(adapter, pair, layers=layers)
    return build_steering_vector(
        captured, tokens=tokens, token_reduction=token_reduction,
        layer_reduction=layer_reduction, direction_method=direction_method,
    )


def add_direction(hidden, vector, *, alpha):
    return hidden + alpha * vector


def edit_component(hidden, vector, *, alpha):
    """Scale the existing component along the unit direction by 1 + alpha."""
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        unit = torch.nn.functional.normalize(vector.float(), dim=-1)
        states = hidden.float()
        edited = states + alpha * (states @ unit).unsqueeze(-1) * unit
    return edited.to(hidden)


OPERATIONS = {"add": add_direction, "component": edit_component}


def _conditional_visual(kwargs):
    return kwargs["mode"] == "gen" and kwargs["key_values_lens"][0].item() > 0


@contextmanager
def count_denoising_evaluations(adapter):
    """Count conditional visual forwards, excluding prompt and CFG reference calls."""
    trace = dict(evaluations=0)

    def count(module, args, kwargs):
        if _conditional_visual(kwargs):
            trace["evaluations"] += 1

    handle = adapter.layers[0].register_forward_pre_hook(count, with_kwargs=True)
    try:
        yield trace
    finally:
        handle.remove()


@contextmanager
def steer_denoising(
    adapter, *, vectors: dict, total_evaluations: int,
    layers: Selection = "all", iterations: Selection = "all",
    text: dict | None = None, visual: dict | None = None,
):
    """Edit prompt prefill once and conditional visual states during selected evaluations."""
    selected_layers = _indices(layers, len(adapter.layers))
    selected_iterations = set(_indices(iterations, total_evaluations))
    rules = {part: rule for part, rule in (("text", text), ("visual", visual)) if rule is not None}
    trace = dict(evaluations=0, text_prefills=0, injections=[], preview={})
    current = dict(part=None, positions=[], iteration=None)
    handles = []

    def locate(module, args, kwargs):
        current.update(part=None, positions=[], iteration=None)
        if kwargs["mode"] == "und" and kwargs["is_causal"]:
            trace["text_prefills"] += 1
            if trace["text_prefills"] == 1:
                current.update(
                    part="text", positions=list(range(1, kwargs["packed_query_sequence"].shape[0] - 1)),
                )
        elif _conditional_visual(kwargs):
            iteration = trace["evaluations"]
            trace["evaluations"] += 1
            if iteration in selected_iterations:
                current.update(
                    part="visual", iteration=iteration,
                    positions=kwargs["packed_vae_token_indexes"].tolist(),
                )

    def hook(layer):
        def apply(module, args, output):
            part = current["part"]
            if part not in rules:
                return output
            rule = rules[part]
            target = [current["positions"][i] for i in _indices(rule["tokens"], len(current["positions"]))]
            hidden = output[0].clone()
            before = hidden[target]
            vector = vectors[layer].to(hidden)
            operation = rule["operation"]
            operation = operation if callable(operation) else OPERATIONS[operation]
            after = operation(before, vector, alpha=rule["alpha"]).to(hidden)
            hidden[target] = after
            trace["injections"].append(dict(
                layer=layer, part=part, iteration=current["iteration"], tokens=len(target),
            ))
            key = f"{layer}:{part}"
            if key not in trace["preview"]:
                trace["preview"][key] = dict(
                    position=target[0], before=before[0, :6].detach().float().cpu(),
                    after=after[0, :6].detach().float().cpu(),
                )
            return (hidden, *output[1:])
        return apply

    try:
        handles.append(adapter.layers[0].register_forward_pre_hook(locate, with_kwargs=True))
        for layer in selected_layers:
            handles.append(adapter.layers[layer].register_forward_hook(hook(layer)))
        yield trace
    finally:
        for handle in handles:
            handle.remove()
