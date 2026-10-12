"""
Direct Logit Attribution (DLA) + Logit Lens

Paper §3.3, step 1: For each node v ∈ V_C, compute its DLA score δ_v ∈ R:
the additive contribution of v's output to the correct-answer logit.

DLA projects each component's output through W_U (the unembedding matrix):
    DLA_v = (v_output @ W_U)[target] − mean((v_output @ W_U)[foils])

No layer norm is applied — attn_out/mlp_out are residual deltas, not full
residual stream states. ln_final is calibrated for the final layer only.

Logit Lens projects the full residual stream at each layer through W_U:
    LL_l = (resid_post[l] @ W_U)[target] − mean((resid_post[l] @ W_U)[foils])

This shows WHEN information about the correct answer emerges across layers.

Components with |δ_v| < ε_DLA are flagged as structurally present
but causally marginal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import numpy as np


@dataclass
class DLAResult:
    """DLA scores for all components in a circuit."""
    # component_name -> δ_v (averaged over dataset)
    scores: Dict[str, float] = field(default_factory=dict)
    # component_name -> per-example scores (for variance analysis)
    scores_per_example: Dict[str, List[float]] = field(default_factory=dict)
    # Per-layer DLA aggregates: layer -> {"attn": mean_δ, "mlp": mean_δ}
    per_layer: Dict[int, Dict[str, float]] = field(default_factory=dict)
    # threshold for "causally marginal"
    epsilon_dla: float = 0.05
    num_examples: int = 0

    def is_marginal(self, component: str) -> bool:
        """True if |δ_v| < ε_DLA (structurally present but causally marginal)."""
        return abs(self.scores.get(component, 0.0)) < self.epsilon_dla

    def causal_components(self) -> List[str]:
        """Components with |δ_v| >= ε_DLA, sorted by |δ_v| descending."""
        return sorted(
            [c for c in self.scores if not self.is_marginal(c)],
            key=lambda c: abs(self.scores[c]),
            reverse=True,
        )


@dataclass
class LogitLensResult:
    """Logit lens: how target-vs-foil logit difference evolves across layers."""
    # Per-layer logit difference: layer -> delta_logit (averaged over dataset)
    per_layer: List[float] = field(default_factory=list)
    # Per-layer DLA breakdown: layer -> {"attn": delta, "mlp": delta}
    per_layer_dla: List[Dict[str, float]] = field(default_factory=list)
    # Per-example curves for variance
    curves: List[List[float]] = field(default_factory=list)
    num_examples: int = 0
    n_layers: int = 0

    def crossing_layer(self) -> Optional[int]:
        """First layer where the logit lens goes positive (target wins)."""
        for i, v in enumerate(self.per_layer):
            if v > 0:
                return i
        return None


def _dla_score(h: torch.Tensor, W_U: torch.Tensor,
               b_U: Optional[torch.Tensor],
               target_id: int, foil_ids: List[int]) -> float:
    """
    Compute DLA for a single component output vector.

    DLA = logit(target) - mean(logit(foils))

    No layer norm applied — component outputs are residual deltas.
    """
    logits = h @ W_U
    if b_U is not None:
        logits = logits + b_U
    target_logit = logits[target_id].item()
    foil_logit = float(np.mean([logits[fid].item() for fid in foil_ids])) if foil_ids else 0.0
    return target_logit - foil_logit


def compute_dla(
    model,
    graph,
    examples: List[Dict],
    tokenizer,
    epsilon_dla: float = 0.05,
) -> DLAResult:
    """
    Compute Direct Logit Attribution for each circuit component.

    Uses contrastive scoring: δ_v = logit(target) - mean(logit(foils))
    projected through W_U without layer norm.

    Args:
        model: HookedTransformer (TransformerLens)
        graph: EAP Graph with .nodes dict
        examples: list of dicts with keys:
                  'clean'/'prompt' (input text)
                  'correct'/'answer' (correct completion — target word)
                  'incorrect'/'foil' (optional — foil word(s) for contrastive scoring)
        tokenizer: tokenizer for encoding target/foil tokens
        epsilon_dla: threshold below which a component is flagged marginal

    Returns:
        DLAResult with per-component DLA scores
    """
    result = DLAResult(epsilon_dla=epsilon_dla)
    n_layers = model.cfg.n_layers

    # Collect circuit node names (only in-graph attention heads and MLPs)
    circuit_nodes = {}
    for name, node in graph.nodes.items():
        if not node.in_graph:
            continue
        if name.startswith("a") and ".h" in name:
            circuit_nodes[name] = ("attn", node.layer, int(name.split(".h")[1]))
        elif name.startswith("m"):
            circuit_nodes[name] = ("mlp", node.layer, None)

    if not circuit_nodes:
        return result

    W_U = model.W_U  # [d_model, vocab_size]
    b_U = model.b_U if hasattr(model, "b_U") and model.b_U is not None else None

    # Build hook filter for efficiency
    hooks_needed = set()
    for name, (comp_type, layer, _) in circuit_nodes.items():
        if comp_type == "attn":
            hooks_needed.add(f"blocks.{layer}.attn.hook_result")
        else:
            hooks_needed.add(f"blocks.{layer}.hook_mlp_out")

    accum = {name: [] for name in circuit_nodes}

    for ex in examples:
        prompt = ex.get("clean", ex.get("prompt", ""))
        correct = ex.get("correct", ex.get("answer", ""))
        if not prompt or not correct:
            continue

        # Encode target token
        target_ids = tokenizer.encode(str(correct), add_special_tokens=False)
        if not target_ids:
            continue
        target_id = target_ids[0]

        # Encode foil tokens (if available, otherwise use empty list)
        foil_ids = []
        foil_text = ex.get("incorrect", ex.get("foil", ex.get("incorrect_word", "")))
        if foil_text:
            fids = tokenizer.encode(str(foil_text), add_special_tokens=False)
            if fids:
                foil_ids = fids

        # Forward pass with cache (filtered to only needed hooks)
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=list(hooks_needed))

        # Compute DLA for each component
        for name, (comp_type, layer, head_idx) in circuit_nodes.items():
            if comp_type == "attn":
                hook_key = f"blocks.{layer}.attn.hook_result"
                if hook_key not in cache:
                    continue
                # Shape: [batch, seq, n_heads, d_model] — take last position, specific head
                v_output = cache[hook_key][0, -1, head_idx, :]
            else:
                hook_key = f"blocks.{layer}.hook_mlp_out"
                if hook_key not in cache:
                    continue
                # Shape: [batch, seq, d_model] — take last position
                v_output = cache[hook_key][0, -1, :]

            delta_v = _dla_score(v_output, W_U, b_U, target_id, foil_ids)
            accum[name].append(delta_v)

    # Average over examples
    for name, values in accum.items():
        if values:
            result.scores[name] = float(np.mean(values))
            result.scores_per_example[name] = values
    result.num_examples = len(examples)

    # Compute per-layer aggregates
    layer_attn = {}
    layer_mlp = {}
    for name, (comp_type, layer, _) in circuit_nodes.items():
        score = result.scores.get(name, 0.0)
        if comp_type == "attn":
            layer_attn.setdefault(layer, []).append(score)
        else:
            layer_mlp.setdefault(layer, []).append(score)

    all_layers = set(layer_attn.keys()) | set(layer_mlp.keys())
    for layer in all_layers:
        result.per_layer[layer] = {
            "attn": float(np.mean(layer_attn.get(layer, [0.0]))),
            "mlp": float(np.mean(layer_mlp.get(layer, [0.0]))),
        }

    return result


def compute_logit_lens(
    model,
    examples: List[Dict],
    tokenizer,
) -> LogitLensResult:
    """
    Compute logit lens: how target-vs-foil logit difference evolves across layers.

    Projects the full residual stream at each layer through W_U (no ln_final).

    Args:
        model: HookedTransformer
        examples: list of dicts with 'clean'/'prompt', 'correct'/'answer',
                  optionally 'incorrect'/'foil'
        tokenizer: tokenizer

    Returns:
        LogitLensResult with per-layer logit differences
    """
    n_layers = model.cfg.n_layers
    W_U = model.W_U
    b_U = model.b_U if hasattr(model, "b_U") and model.b_U is not None else None

    hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
    attn_hooks = [f"blocks.{l}.hook_attn_out" for l in range(n_layers)]
    mlp_hooks = [f"blocks.{l}.hook_mlp_out" for l in range(n_layers)]
    all_hooks = hook_names + attn_hooks + mlp_hooks

    curves = []
    dla_curves_attn = []
    dla_curves_mlp = []

    for ex in examples:
        prompt = ex.get("clean", ex.get("prompt", ""))
        correct = ex.get("correct", ex.get("answer", ""))
        if not prompt or not correct:
            continue

        target_ids = tokenizer.encode(str(correct), add_special_tokens=False)
        if not target_ids:
            continue
        target_id = target_ids[0]

        foil_ids = []
        foil_text = ex.get("incorrect", ex.get("foil", ex.get("incorrect_word", "")))
        if foil_text:
            fids = tokenizer.encode(str(foil_text), add_special_tokens=False)
            if fids:
                foil_ids = fids

        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=all_hooks)

        # Logit lens: residual stream at each layer
        ll = []
        for l in range(n_layers):
            h = cache[f"blocks.{l}.hook_resid_post"][0, -1, :]
            ll.append(_dla_score(h, W_U, b_U, target_id, foil_ids))
        curves.append(ll)

        # Per-layer DLA breakdown
        attn_dla = []
        mlp_dla = []
        for l in range(n_layers):
            attn_key = f"blocks.{l}.hook_attn_out"
            mlp_key = f"blocks.{l}.hook_mlp_out"
            a_score = _dla_score(cache[attn_key][0, -1, :], W_U, b_U, target_id, foil_ids) if attn_key in cache else 0.0
            m_score = _dla_score(cache[mlp_key][0, -1, :], W_U, b_U, target_id, foil_ids) if mlp_key in cache else 0.0
            attn_dla.append(a_score)
            mlp_dla.append(m_score)
        dla_curves_attn.append(attn_dla)
        dla_curves_mlp.append(mlp_dla)

    result = LogitLensResult(n_layers=n_layers, num_examples=len(curves))

    if curves:
        result.per_layer = [float(np.mean(col)) for col in zip(*curves)]
        result.curves = curves
        attn_means = [float(np.mean(col)) for col in zip(*dla_curves_attn)]
        mlp_means = [float(np.mean(col)) for col in zip(*dla_curves_mlp)]
        result.per_layer_dla = [
            {"attn": a, "mlp": m} for a, m in zip(attn_means, mlp_means)
        ]

    return result
