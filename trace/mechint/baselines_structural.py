"""
baselines_structural.py
=======================
Structural baselines for comparing against ILP signature distances.

1. Weisfeiler-Leman (WL) graph kernel — computes structural similarity
   using 1-WL neighbourhood aggregation on circuit graphs.

2. Random forest on structural features — trains a classifier on the same
   features ILP uses (component_ratio, has_motif, etc.) but without logical
   structure. Shows whether the *features* or the *logical formalism* does
   the work.

3. Feature-vector cosine/Euclidean distance — direct distance on the
   structural feature vector, the simplest possible baseline.

These serve as controls for Section 5.2 (RQ2).
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Circuit graph construction from results JSON
# ---------------------------------------------------------------------------

def _load_circuit_graph(results_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a typed directed graph from a results JSON record.

    Returns dict with:
        nodes: list of (name, type, layer)
        edges: list of (src, tgt)
        node_types: {name: type}
        node_layers: {name: layer}
    """
    circuit = (results_json.get("layer1") or {}).get("circuit", {})
    nodes_raw = circuit.get("nodes", [])
    edges_raw = circuit.get("edges", [])

    nodes = []
    node_types = {}
    node_layers = {}
    for n in nodes_raw:
        name = n["name"]
        ntype = n.get("type", "unknown")
        layer = n.get("layer", 0)
        nodes.append((name, ntype, layer))
        node_types[name] = ntype
        node_layers[name] = layer

    edges = []
    for e in edges_raw:
        src = e.get("source", e.get("parent", ""))
        tgt = e.get("target", e.get("child", ""))
        if src and tgt:
            edges.append((src, tgt))

    return {
        "nodes": nodes,
        "edges": edges,
        "node_types": node_types,
        "node_layers": node_layers,
    }


# =========================================================================
# 1. Weisfeiler-Leman (WL) graph kernel
# =========================================================================

def _wl_hash_iteration(
    adj: Dict[str, List[str]],
    labels: Dict[str, str],
) -> Dict[str, str]:
    """
    One iteration of 1-WL colour refinement.

    For each node, concatenate its label with sorted neighbour labels,
    then hash to produce a new label.
    """
    new_labels = {}
    for node in labels:
        neighbours = adj.get(node, [])
        neighbour_labels = sorted(labels.get(n, "?") for n in neighbours)
        combined = labels[node] + "|" + ",".join(neighbour_labels)
        new_labels[node] = combined
    return new_labels


def wl_feature_vector(
    graph: Dict[str, Any],
    iterations: int = 3,
    attributed: bool = False,
) -> Counter:
    """
    Compute the WL subtree feature vector for a circuit graph.

    Initial labels are node types (attn/mlp). With attributed=True, initial
    labels are (type, layer-quartile) pairs — a stronger baseline that uses
    the same node attributes available to ILP. Each WL
    iteration refines labels using neighbourhood aggregation. The feature
    vector is the multiset (Counter) of all labels across all iterations.

    Args:
        graph: output of _load_circuit_graph()
        iterations: number of WL iterations (default 3)
        attributed: label nodes with (type, layer quartile) instead of type

    Returns:
        Counter mapping label strings to counts
    """
    # Build adjacency list (directed — include both in and out neighbours
    # for WL, which is typically on undirected graphs)
    adj: Dict[str, List[str]] = defaultdict(list)
    for src, tgt in graph["edges"]:
        adj[src].append(tgt)
        adj[tgt].append(src)

    if attributed:
        # Layer quartile relative to the deepest layer in this circuit —
        # scale-invariant, mirroring the normalised-depth predicates in ILP
        max_layer = max((layer for _, _, layer in graph["nodes"]), default=0)
        denom = max(max_layer, 1)
        labels = {
            name: f"{ntype}:q{min(3, int(4 * layer / (denom + 1)))}"
            for name, ntype, layer in graph["nodes"]
        }
    else:
        # Initial labels: node type
        labels = {name: ntype for name, ntype, _ in graph["nodes"]}

    feature_vector: Counter = Counter()
    # Iteration 0: count initial labels
    feature_vector.update(labels.values())

    for _ in range(iterations):
        labels = _wl_hash_iteration(adj, labels)
        feature_vector.update(labels.values())

    return feature_vector


def wl_kernel(
    graph_a: Dict[str, Any],
    graph_b: Dict[str, Any],
    iterations: int = 3,
    attributed: bool = False,
) -> float:
    """
    Compute the WL subtree kernel between two circuit graphs.

    K(G_a, G_b) = <phi(G_a), phi(G_b)>  (dot product of WL feature vectors)

    Returns the normalised kernel value in [0, 1]:
        K_norm = K(a,b) / sqrt(K(a,a) * K(b,b))
    """
    fv_a = wl_feature_vector(graph_a, iterations, attributed=attributed)
    fv_b = wl_feature_vector(graph_b, iterations, attributed=attributed)

    # Dot product over shared keys
    all_keys = set(fv_a.keys()) | set(fv_b.keys())
    dot = sum(fv_a.get(k, 0) * fv_b.get(k, 0) for k in all_keys)
    norm_a = sum(v * v for v in fv_a.values()) ** 0.5
    norm_b = sum(v * v for v in fv_b.values()) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def wl_distance(
    graph_a: Dict[str, Any],
    graph_b: Dict[str, Any],
    iterations: int = 3,
    attributed: bool = False,
) -> float:
    """
    WL kernel distance: d = 1 - K_norm.

    0 = identical structure, 1 = maximally different.
    """
    return 1.0 - wl_kernel(graph_a, graph_b, iterations, attributed=attributed)


def wl_distance_matrix(
    records: Dict[str, Dict[str, Any]],
    iterations: int = 3,
    attributed: bool = False,
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """
    Compute pairwise WL kernel distance for all circuits.

    Args:
        records: {task_name: results_json_dict}
        iterations: WL iterations
        attributed: use (type, layer-quartile) node labels (stronger baseline)

    Returns:
        (task_list, matrix) where matrix[t1][t2] = wl_distance
    """
    tasks = sorted(records.keys())
    graphs = {t: _load_circuit_graph(records[t]) for t in tasks}

    matrix: Dict[str, Dict[str, float]] = {}
    for t1 in tasks:
        matrix[t1] = {}
        for t2 in tasks:
            if t1 == t2:
                matrix[t1][t2] = 0.0
            else:
                matrix[t1][t2] = round(
                    wl_distance(graphs[t1], graphs[t2], iterations,
                                attributed=attributed), 4
                )
    return tasks, matrix


def _find_ioi_key(tasks: List[str]) -> Optional[str]:
    """First task whose name contains 'ioi' (the separation reference task)."""
    for t in tasks:
        if "ioi" in t.lower():
            return t
    return None


def bootstrap_separation_ci(
    matrix: Dict[str, Dict[str, float]],
    tasks: List[str],
    ioi_key: Optional[str] = None,
    B: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Optional[List[float]]:
    """
    Bootstrap CI for the IOI-separation ratio.

    The separation ratio is mean(IOI-to-others) / mean(non-IOI pairwise).
    Non-IOI tasks are resampled with replacement B times; identical-task
    pairs in a resample are skipped (their distance is 0 by construction).

    Returns [point, lo, hi] (percentile CI at level 1-alpha), or None when
    there are fewer than 3 non-IOI tasks or no IOI task.
    """
    ioi_key = ioi_key or _find_ioi_key(tasks)
    if not ioi_key:
        return None
    non_ioi = [t for t in tasks if t != ioi_key]
    if len(non_ioi) < 3:
        return None

    def _ratio(sample: List[str]) -> Optional[float]:
        ioi_dists = [matrix[ioi_key][t] for t in sample]
        other = [matrix[t1][t2]
                 for i, t1 in enumerate(sample)
                 for t2 in sample[i + 1:]
                 if t1 != t2]
        if not other:
            return None
        mean_other = float(np.mean(other))
        if mean_other <= 0:
            return None
        return float(np.mean(ioi_dists)) / mean_other

    point = _ratio(non_ioi)
    if point is None:
        return None

    rng = np.random.default_rng(seed)
    ratios = []
    while len(ratios) < B:
        sample = list(rng.choice(non_ioi, size=len(non_ioi), replace=True))
        r = _ratio(sample)
        if r is not None:
            ratios.append(r)

    lo = float(np.percentile(ratios, 100 * alpha / 2))
    hi = float(np.percentile(ratios, 100 * (1 - alpha / 2)))
    return [round(point, 4), round(lo, 4), round(hi, 4)]


# =========================================================================
# 2. Structural feature extraction (shared by RF and feature-vector baselines)
# =========================================================================

def extract_structural_features(
    results_json: Dict[str, Any],
) -> Dict[str, float]:
    """
    Extract the same structural features that ILP_arch uses as background
    knowledge, represented as a flat numeric feature dict.

    Features:
        - component_ratio_attn: fraction of attention nodes
        - n_nodes, n_edges
        - n_attn, n_mlp
        - layer_span: max_layer - min_layer
        - depth_mean, depth_min, depth_max (normalised)
        - density: n_edges / (n_nodes * (n_nodes - 1))
        - edge_type_aa, edge_type_am, edge_type_ma, edge_type_mm (fractions)
        - motif_* : binary indicators for each named motif
    """
    circuit = (results_json.get("layer1") or {}).get("circuit", {})
    nodes = circuit.get("nodes", [])
    edges = circuit.get("edges", [])
    l3 = (results_json.get("layer3") or {})

    n_nodes = len(nodes)
    n_attn = sum(1 for n in nodes if n.get("type") == "attn")
    n_mlp = sum(1 for n in nodes if n.get("type") == "mlp")
    attn_ratio = n_attn / n_nodes if n_nodes else 0.0

    layers = [n.get("layer", 0) for n in nodes]
    n_layers = (results_json.get("layer1") or {}).get("circuit", {}).get("n_layers")
    if not n_layers:
        n_layers = max(layers) + 1 if layers else 1
    layer_min = min(layers) if layers else 0
    layer_max = max(layers) if layers else 0
    layer_span = layer_max - layer_min
    depth_mean = np.mean(layers) / n_layers if layers and n_layers else 0.0
    depth_min = layer_min / n_layers if n_layers else 0.0
    depth_max = layer_max / n_layers if n_layers else 0.0

    n_edges = len(edges)
    max_edges = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
    density = n_edges / max_edges

    # Edge type distribution
    node_types = {n["name"]: n.get("type", "unknown") for n in nodes}
    et = {"aa": 0, "am": 0, "ma": 0, "mm": 0}
    for e in edges:
        src = e.get("source", e.get("parent", ""))
        tgt = e.get("target", e.get("child", ""))
        if tgt == "logits":
            continue
        st = "a" if node_types.get(src, "").startswith("attn") or (
            src.startswith("a") and ".h" in src) else "m"
        tt = "a" if node_types.get(tgt, "").startswith("attn") or (
            tgt.startswith("a") and ".h" in tgt) else "m"
        et[st + tt] += 1
    total_et = sum(et.values()) or 1

    # Motifs (binary)
    motifs = l3.get("motifs", [])
    all_possible_motifs = [
        "attn_to_attn", "attn_feeds_mlp", "mlp_gates_attn",
        "attn_chain_3", "attn_mlp_attn_sandwich", "scaffold_event_entity",
        "early_circuit", "late_circuit", "deep_span",
        "balanced_mix", "mlp_heavy", "attn_dominant",
        "has_skip_connections", "has_hub_node",
        "early_attn_peak", "mid_layer_attn_peak", "late_attn_peak",
    ]
    motif_set = set(motifs)

    feats = {
        "component_ratio_attn": attn_ratio,
        "n_nodes": float(n_nodes),
        "n_edges": float(n_edges),
        "n_attn": float(n_attn),
        "n_mlp": float(n_mlp),
        "layer_span": float(layer_span),
        "depth_mean": depth_mean,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "density": density,
        "edge_aa_frac": et["aa"] / total_et,
        "edge_am_frac": et["am"] / total_et,
        "edge_ma_frac": et["ma"] / total_et,
        "edge_mm_frac": et["mm"] / total_et,
    }

    for m in all_possible_motifs:
        feats[f"motif_{m}"] = 1.0 if m in motif_set else 0.0

    return feats


def extract_feature_matrix(
    records: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Extract structural feature matrix for all circuits.

    Returns:
        (task_names, feature_matrix [n_tasks x n_features], feature_names)
    """
    tasks = sorted(records.keys())
    feat_dicts = [extract_structural_features(records[t]) for t in tasks]
    feature_names = sorted(feat_dicts[0].keys())
    matrix = np.array([[fd[f] for f in feature_names] for fd in feat_dicts])
    return tasks, matrix, feature_names


# =========================================================================
# 3. Feature-vector distance baseline
# =========================================================================

def feature_vector_distance_matrix(
    records: Dict[str, Dict[str, Any]],
    metric: str = "euclidean",
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """
    Compute pairwise distance on standardised structural feature vectors.

    Args:
        records: {task_name: results_json_dict}
        metric: "euclidean" or "cosine"

    Returns:
        (task_list, matrix)
    """
    from scipy.spatial.distance import pdist, squareform

    tasks, X, _ = extract_feature_matrix(records)

    # Standardise
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_norm = (X - mu) / sigma

    if metric == "cosine":
        dist_vec = pdist(X_norm, metric="cosine")
    else:
        dist_vec = pdist(X_norm, metric="euclidean")
        # Normalise to [0, 1] for comparability
        max_d = dist_vec.max() if len(dist_vec) > 0 else 1.0
        if max_d > 0:
            dist_vec = dist_vec / max_d

    D = squareform(dist_vec)

    matrix: Dict[str, Dict[str, float]] = {}
    for i, t1 in enumerate(tasks):
        matrix[t1] = {}
        for j, t2 in enumerate(tasks):
            matrix[t1][t2] = round(float(D[i, j]), 4)

    return tasks, matrix


# =========================================================================
# 4. Random forest classifier on structural features
# =========================================================================

def random_forest_classify(
    records: Dict[str, Dict[str, Any]],
    task_families: Dict[str, str],
    n_estimators: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a random forest on the same structural features ILP uses.

    Uses leave-one-out cross-validation (since n=5 is tiny).
    Reports per-class accuracy and feature importances.

    This tests whether the *features* or the *logical formalism* (ILP's
    conjunctive clauses + subsumption) does the discriminative work.

    Args:
        records: {task_name: results_json_dict}
        task_families: {task_name: family_label}
        n_estimators: number of trees
        seed: random seed

    Returns:
        dict with accuracy, predictions, feature importances, confusion
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneOut

    tasks, X, feature_names = extract_feature_matrix(records)
    y = np.array([task_families.get(t, "unknown") for t in tasks])

    # Standardise
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_norm = (X - mu) / sigma

    loo = LeaveOneOut()
    predictions = []
    probas = []

    for train_idx, test_idx in loo.split(X_norm):
        clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=seed, max_depth=3
        )
        clf.fit(X_norm[train_idx], y[train_idx])
        pred = clf.predict(X_norm[test_idx])[0]
        proba = clf.predict_proba(X_norm[test_idx])[0]
        predictions.append(pred)
        probas.append(dict(zip(clf.classes_, proba.tolist())))

    # Full model for feature importances
    clf_full = RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, max_depth=3
    )
    clf_full.fit(X_norm, y)
    importances = dict(zip(feature_names, clf_full.feature_importances_.tolist()))

    correct = sum(1 for p, t in zip(predictions, y) if p == t)
    accuracy = correct / len(y)

    # Top features
    top_feats = sorted(importances.items(), key=lambda x: -x[1])[:10]

    return {
        "tasks": tasks,
        "true_labels": y.tolist(),
        "predictions": predictions,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(y),
        "feature_importances": importances,
        "top_features": top_feats,
        "per_task": [
            {
                "task": tasks[i],
                "true": y[i],
                "predicted": predictions[i],
                "correct": predictions[i] == y[i],
                "probabilities": probas[i],
            }
            for i in range(len(tasks))
        ],
    }


# =========================================================================
# 5. Decision stump baseline (simplest possible)
# =========================================================================

def decision_stump_classify(
    records: Dict[str, Dict[str, Any]],
    task_families: Dict[str, str],
) -> Dict[str, Any]:
    """
    Find the single best feature + threshold that separates task families.

    This is the simplest possible baseline: equivalent to a depth-1 decision
    tree. If this matches ILP performance, the conjunctive structure of
    ILP clauses adds no value at the current sample size.

    Returns:
        dict with best_feature, threshold, accuracy, predictions
    """
    from sklearn.tree import DecisionTreeClassifier

    tasks, X, feature_names = extract_feature_matrix(records)
    y = np.array([task_families.get(t, "unknown") for t in tasks])

    # Depth-1 tree = decision stump
    stump = DecisionTreeClassifier(max_depth=1, random_state=42)
    stump.fit(X, y)
    predictions = stump.predict(X).tolist()
    correct = sum(1 for p, t in zip(predictions, y) if p == t)
    accuracy = correct / len(y)

    # Extract the split feature
    tree = stump.tree_
    if tree.feature[0] >= 0:
        best_feature = feature_names[tree.feature[0]]
        threshold = float(tree.threshold[0])
    else:
        best_feature = "none"
        threshold = 0.0

    return {
        "tasks": tasks,
        "true_labels": y.tolist(),
        "predictions": predictions,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(y),
        "best_feature": best_feature,
        "threshold": round(threshold, 4),
        "per_task": [
            {
                "task": tasks[i],
                "true": y[i],
                "predicted": predictions[i],
                "correct": predictions[i] == y[i],
            }
            for i in range(len(tasks))
        ],
    }


# =========================================================================
# 6. Comprehensive comparison report
# =========================================================================

def full_baseline_comparison(
    records: Dict[str, Dict[str, Any]],
    task_families: Dict[str, str],
    ilp_distance_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    wl_iterations: int = 3,
) -> Dict[str, Any]:
    """
    Run all baselines and compare against ILP signature distances.

    Returns a comprehensive report with:
        - WL kernel distance matrix
        - Feature-vector distance matrices (euclidean, cosine)
        - Random forest LOO accuracy + feature importances
        - Decision stump accuracy
        - Correlation between distance matrices (Mantel-like)
        - Summary comparison table
    """
    tasks = sorted(records.keys())

    # Distance matrices
    _, wl_matrix = wl_distance_matrix(records, iterations=wl_iterations)
    _, wl_attr_matrix = wl_distance_matrix(records, iterations=wl_iterations,
                                           attributed=True)
    _, feat_euc_matrix = feature_vector_distance_matrix(records, "euclidean")
    _, feat_cos_matrix = feature_vector_distance_matrix(records, "cosine")

    # Classification
    rf_result = random_forest_classify(records, task_families)
    stump_result = decision_stump_classify(records, task_families)

    # Flatten upper triangle for correlation analysis
    def _upper_tri(matrix):
        vals = []
        for i, t1 in enumerate(tasks):
            for j, t2 in enumerate(tasks):
                if i < j:
                    vals.append(matrix[t1][t2])
        return np.array(vals)

    wl_vec = _upper_tri(wl_matrix)
    feat_euc_vec = _upper_tri(feat_euc_matrix)
    feat_cos_vec = _upper_tri(feat_cos_matrix)

    correlations = {}
    if ilp_distance_matrix:
        ilp_vec = _upper_tri(ilp_distance_matrix)

        def _pearson(a, b):
            if len(a) < 3:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])

        correlations = {
            "ilp_vs_wl": _pearson(ilp_vec, wl_vec),
            "ilp_vs_feat_euclidean": _pearson(ilp_vec, feat_euc_vec),
            "ilp_vs_feat_cosine": _pearson(ilp_vec, feat_cos_vec),
            "wl_vs_feat_euclidean": _pearson(wl_vec, feat_euc_vec),
        }

    # Cluster quality: does each method separate IOI from the rest?
    def _separation_score(matrix):
        """
        Ratio of mean(IOI-to-others) / mean(non-IOI-pairwise).
        Higher = better separation of IOI.
        """
        ioi_key = _find_ioi_key(tasks)
        if not ioi_key:
            return float("nan")

        ioi_dists = [matrix[ioi_key][t] for t in tasks if t != ioi_key]
        other_dists = []
        for i, t1 in enumerate(tasks):
            if t1 == ioi_key:
                continue
            for t2 in tasks[i+1:]:
                if t2 == ioi_key:
                    continue
                other_dists.append(matrix[t1][t2])

        mean_ioi = np.mean(ioi_dists) if ioi_dists else 0
        mean_other = np.mean(other_dists) if other_dists else 1
        return float(mean_ioi / mean_other) if mean_other > 0 else float("inf")

    separation = {
        "wl": _separation_score(wl_matrix),
        "wl_attributed": _separation_score(wl_attr_matrix),
        "feat_euclidean": _separation_score(feat_euc_matrix),
        "feat_cosine": _separation_score(feat_cos_matrix),
    }
    if ilp_distance_matrix:
        separation["ilp"] = _separation_score(ilp_distance_matrix)

    # Bootstrap 95% CIs on the separation ratios:
    # [point, lo95, hi95] per method; None when too few tasks
    ci_inputs = {
        "wl": wl_matrix,
        "wl_attributed": wl_attr_matrix,
        "feat_euclidean": feat_euc_matrix,
        "feat_cosine": feat_cos_matrix,
    }
    if ilp_distance_matrix:
        ci_inputs["ilp"] = ilp_distance_matrix
    separation_ci = {
        name: bootstrap_separation_ci(m, tasks)
        for name, m in ci_inputs.items()
    }

    return {
        "tasks": tasks,
        "task_families": task_families,
        "distance_matrices": {
            "wl_kernel": wl_matrix,
            "wl_attributed": wl_attr_matrix,
            "feature_euclidean": feat_euc_matrix,
            "feature_cosine": feat_cos_matrix,
        },
        "classification": {
            "random_forest": rf_result,
            "decision_stump": stump_result,
        },
        "correlations": correlations,
        "separation_scores": separation,
        "separation_ci": separation_ci,
    }


# =========================================================================
# 7. δ_ILP sensitivity sweep
# =========================================================================

def delta_ilp_sensitivity(
    distance_matrix: Dict[str, Dict[str, float]],
    task_families: Dict[str, str],
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Sweep δ_ILP threshold and measure how transfer pair partitioning changes.

    For each threshold, count:
      - true_within: same-family pairs with d < threshold (correct accepts)
      - false_within: cross-family pairs with d < threshold (incorrect accepts)
      - true_cross: cross-family pairs with d >= threshold (correct rejects)
      - false_cross: same-family pairs with d >= threshold (incorrect rejects)

    Shows whether the IOI/non-IOI boundary is stable or threshold-dependent.

    Args:
        distance_matrix: {task: {task: distance}}
        task_families: {task: family_label}
        thresholds: list of δ values to test (default: 0.05 to 0.50 step 0.05)

    Returns:
        dict with per-threshold stats and the best threshold by F1
    """
    if thresholds is None:
        thresholds = [round(0.05 * i, 2) for i in range(1, 11)]

    tasks = sorted(distance_matrix.keys())

    # Build all pairs with labels
    pairs = []
    for i, t1 in enumerate(tasks):
        for t2 in tasks[i+1:]:
            d = distance_matrix[t1][t2]
            f1 = task_families.get(t1, "?")
            f2 = task_families.get(t2, "?")
            same = f1 == f2
            pairs.append((t1, t2, d, same))

    n_same = sum(1 for _, _, _, s in pairs if s)
    n_cross = sum(1 for _, _, _, s in pairs if not s)

    results_per_threshold = []
    for delta in thresholds:
        tp = sum(1 for _, _, d, s in pairs if s and d < delta)       # same-family, accepted
        fp = sum(1 for _, _, d, s in pairs if not s and d < delta)   # cross-family, accepted (bad)
        tn = sum(1 for _, _, d, s in pairs if not s and d >= delta)  # cross-family, rejected
        fn = sum(1 for _, _, d, s in pairs if s and d >= delta)      # same-family, rejected (bad)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(pairs) if pairs else 0.0

        results_per_threshold.append({
            "delta": delta,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        })

    best = max(results_per_threshold, key=lambda r: r["f1"])

    return {
        "n_pairs": len(pairs),
        "n_same_family": n_same,
        "n_cross_family": n_cross,
        "thresholds": results_per_threshold,
        "best_threshold": best["delta"],
        "best_f1": best["f1"],
    }
