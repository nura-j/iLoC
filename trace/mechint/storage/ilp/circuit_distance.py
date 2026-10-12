"""
Circuit-level structural distance — d_circuit

Complements the ILP signature distance (d_signature) by comparing
actual circuit graph properties rather than learned clause abstractions.

    d_signature: "Do these circuits have similar learned theories?"
    d_circuit:   "Do these circuits look structurally similar?"

The gap between the two is informative:
    - Low d_signature, high d_circuit → same mechanism, different realisation
    - Low d_circuit, high d_signature → similar structure, different function

All features are normalised (relative depth, ratios, distributions) so the
distance is scale-invariant and meaningful across model sizes.

Components:
    1. Depth envelope similarity (positional overlap)
    2. Component composition (attn ratio, relative size)
    3. Edge topology (edge-type distribution cosine)
    4. Layer distribution (per-layer component density)

Paper ref: §4.3 (component overlap baseline), §3.7 (transfer fingerprinting)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# Default weights for d_circuit components
_DEFAULT_WEIGHTS = {
    "depth":       0.30,   # depth envelope overlap
    "composition": 0.25,   # attn ratio + relative size
    "edge_topo":   0.25,   # edge-type distribution
    "layer_dist":  0.20,   # per-layer component density
}


def circuit_distance(
    c1: Dict[str, Any],
    c2: Dict[str, Any],
    model_n_layers: int = 6,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute structural distance between two circuits.

    Args:
        c1, c2: circuit dicts from layer1.circuit in results JSON,
                 or any dict with 'nodes' and 'edges' lists.
        model_n_layers: number of layers in the model (for normalisation)
        weights: override default component weights

    Returns:
        float in [0, 1] — 0 = identical structure, 1 = maximally different
    """
    w = _DEFAULT_WEIGHTS.copy()
    if weights:
        w.update(weights)

    f1 = _extract_features(c1, model_n_layers)
    f2 = _extract_features(c2, model_n_layers)

    if not f1 or not f2:
        return 1.0  # one circuit is empty

    d_depth = _depth_distance(f1, f2)
    d_comp = _composition_distance(f1, f2)
    d_edge = _edge_topology_distance(f1, f2)
    d_layer = _layer_distribution_distance(f1, f2)

    d = (w["depth"]       * d_depth
         + w["composition"] * d_comp
         + w["edge_topo"]   * d_edge
         + w["layer_dist"]  * d_layer)

    return round(min(1.0, max(0.0, d)), 4)


def circuit_distance_breakdown(
    c1: Dict[str, Any],
    c2: Dict[str, Any],
    model_n_layers: int = 6,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Like circuit_distance but returns per-component contributions.
    """
    w = _DEFAULT_WEIGHTS.copy()
    if weights:
        w.update(weights)

    f1 = _extract_features(c1, model_n_layers)
    f2 = _extract_features(c2, model_n_layers)

    if not f1 or not f2:
        return {"distance": 1.0, "reason": "empty circuit"}

    d_depth = _depth_distance(f1, f2)
    d_comp = _composition_distance(f1, f2)
    d_edge = _edge_topology_distance(f1, f2)
    d_layer = _layer_distribution_distance(f1, f2)

    total = (w["depth"] * d_depth + w["composition"] * d_comp
             + w["edge_topo"] * d_edge + w["layer_dist"] * d_layer)

    return {
        "distance": round(min(1.0, total), 4),
        "depth_distance": round(d_depth, 4),
        "composition_distance": round(d_comp, 4),
        "edge_topology_distance": round(d_edge, 4),
        "layer_distribution_distance": round(d_layer, 4),
        "weighted_contributions": {
            "depth": round(w["depth"] * d_depth, 4),
            "composition": round(w["composition"] * d_comp, 4),
            "edge_topo": round(w["edge_topo"] * d_edge, 4),
            "layer_dist": round(w["layer_dist"] * d_layer, 4),
        },
        "features_c1": f1,
        "features_c2": f2,
    }


def circuit_distance_matrix(
    circuits: Dict[str, Dict[str, Any]],
    model_n_layers: int = 6,
) -> Dict[str, Dict[str, float]]:
    """
    Pairwise d_circuit matrix.

    Args:
        circuits: {name: layer1_circuit_dict}
        model_n_layers: for normalisation

    Returns:
        {name1: {name2: distance}}
    """
    names = sorted(circuits.keys())
    matrix = {}
    for n1 in names:
        matrix[n1] = {}
        for n2 in names:
            if n1 == n2:
                matrix[n1][n2] = 0.0
            else:
                matrix[n1][n2] = circuit_distance(
                    circuits[n1], circuits[n2], model_n_layers
                )
    return matrix


# ---------------------------------------------------------------------------
# Feature extraction (all normalised to [0, 1])
# ---------------------------------------------------------------------------

def _extract_features(circuit: Dict[str, Any], n_layers: int) -> Optional[Dict]:
    """Extract normalised structural features from a circuit dict."""
    nodes = circuit.get("nodes", [])
    edges = circuit.get("edges", [])

    if not nodes:
        return None

    n_layers = max(n_layers, 1)

    # Component counts
    attn_nodes = [n for n in nodes if n.get("type") == "attn"]
    mlp_nodes = [n for n in nodes if n.get("type") == "mlp"]
    # Exclude input/logits from structural counts
    struct_nodes = attn_nodes + mlp_nodes
    n_attn = len(attn_nodes)
    n_mlp = len(mlp_nodes)
    n_total = max(n_attn + n_mlp, 1)

    # Depth envelope (normalised)
    layers = [n.get("layer", 0) for n in struct_nodes]
    if not layers:
        layers = [0]
    depth_min = min(layers) / n_layers
    depth_max = max(layers) / n_layers
    depth_mean = sum(layers) / len(layers) / n_layers

    # Component ratio
    attn_ratio = n_attn / n_total

    # Relative size (fraction of model capacity used)
    model_total = n_layers * 4  # heads per layer estimate
    rel_size = n_total / max(model_total, 1)

    # Edge type distribution
    node_types = {}
    for n in nodes:
        name = n.get("name", "")
        if n.get("type") == "attn":
            node_types[name] = "attn"
        elif n.get("type") == "mlp":
            node_types[name] = "mlp"
        else:
            node_types[name] = "other"

    edge_counts = {"aa": 0, "am": 0, "ma": 0, "mm": 0}
    for e in edges:
        st = node_types.get(e.get("source", ""), "other")
        tt = node_types.get(e.get("target", ""), "other")
        if st == "attn" and tt == "attn":
            edge_counts["aa"] += 1
        elif st == "attn" and tt == "mlp":
            edge_counts["am"] += 1
        elif st == "mlp" and tt == "attn":
            edge_counts["ma"] += 1
        elif st == "mlp" and tt == "mlp":
            edge_counts["mm"] += 1

    n_edges = max(sum(edge_counts.values()), 1)
    edge_dist = {k: v / n_edges for k, v in edge_counts.items()}

    # Per-layer component density (histogram)
    layer_hist = [0.0] * n_layers
    for l in layers:
        if 0 <= l < n_layers:
            layer_hist[l] += 1
    hist_total = max(sum(layer_hist), 1)
    layer_dist = [c / hist_total for c in layer_hist]

    return {
        "depth_min": depth_min,
        "depth_max": depth_max,
        "depth_mean": depth_mean,
        "attn_ratio": attn_ratio,
        "rel_size": rel_size,
        "n_attn": n_attn,
        "n_mlp": n_mlp,
        "edge_dist": edge_dist,
        "layer_dist": layer_dist,
    }


# ---------------------------------------------------------------------------
# Distance components (each returns value in [0, 1])
# ---------------------------------------------------------------------------

def _depth_distance(f1: Dict, f2: Dict) -> float:
    """Distance based on depth envelope overlap."""
    # Intersection-over-union of depth ranges
    min1, max1 = f1["depth_min"], f1["depth_max"]
    min2, max2 = f2["depth_min"], f2["depth_max"]

    overlap = max(0.0, min(max1, max2) - max(min1, min2))
    union = max(max1, max2) - min(min1, min2)

    if union <= 0:
        iou = 1.0 if (min1 == min2) else 0.0
    else:
        iou = overlap / union

    # Also penalise mean depth difference
    mean_diff = abs(f1["depth_mean"] - f2["depth_mean"])

    return 1.0 - (0.7 * iou + 0.3 * (1.0 - mean_diff))


def _composition_distance(f1: Dict, f2: Dict) -> float:
    """Distance based on component composition (attn ratio, size)."""
    ratio_diff = abs(f1["attn_ratio"] - f2["attn_ratio"])
    size_diff = abs(f1["rel_size"] - f2["rel_size"])
    return min(1.0, ratio_diff + size_diff)


def _edge_topology_distance(f1: Dict, f2: Dict) -> float:
    """Cosine distance on edge-type distribution vectors."""
    keys = ["aa", "am", "ma", "mm"]
    v1 = [f1["edge_dist"].get(k, 0.0) for k in keys]
    v2 = [f2["edge_dist"].get(k, 0.0) for k in keys]

    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))

    if n1 * n2 < 1e-9:
        return 1.0  # one or both have no typed edges

    cosine_sim = dot / (n1 * n2)
    return 1.0 - max(0.0, cosine_sim)


def _layer_distribution_distance(f1: Dict, f2: Dict) -> float:
    """Jensen-Shannon-like distance on per-layer component density."""
    d1 = f1["layer_dist"]
    d2 = f2["layer_dist"]

    # Pad to same length
    max_len = max(len(d1), len(d2))
    d1 = d1 + [0.0] * (max_len - len(d1))
    d2 = d2 + [0.0] * (max_len - len(d2))

    # L1 distance (normalised, max = 2.0 for distributions)
    l1 = sum(abs(a - b) for a, b in zip(d1, d2))
    return min(1.0, l1 / 2.0)
