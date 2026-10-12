"""
ilp_signature_distance.py
=========================
ILP signature distance metric (paper §3.6 / Eq. 26 / Appendix F).

The paper defines:

    d(S1, S2) = w1 * |n_clauses_1 - n_clauses_2|
              + w2 * |d_bar_1 - d_bar_2|
              + w3 * (1 - Jaccard(P1, P2))
              + w4 * |l_bar_1 - l_bar_2|
              + w5 * |complexity_1 - complexity_2|

where w3 (predicate overlap) is weighted highest because circuits
implementing the same logical function should use similar predicates.

Also implements:
  - transfer_pair_check: Def. 5 (is (Ca, Cb) a valid transfer pair?)
  - signature_distance_matrix: pairwise d for a list of signatures
  - closest_match: find the most similar known signature in a registry

Public API
----------
    d = ilp_signature_distance(S1, S2)
    d = ilp_signature_distance(S1, S2, weights={"w3": 0.6, ...})
    is_pair = is_transfer_pair(Sa, Sb, threshold=0.3)
    mat = signature_distance_matrix(sigs)
    best = closest_match(query_sig, registry_sigs, threshold=0.3)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# ─── Default weights (paper Eq. 26) ──────────────────────────────────────────
# w3 is largest: predicate set similarity is the primary signal.
_DEFAULT_WEIGHTS = {
    "w1": 0.10,   # |n_clauses_1 - n_clauses_2|
    "w2": 0.10,   # |d_bar_1 - d_bar_2|           (depth)
    "w3": 0.20,   # 1 - Jaccard(P1, P2)            (predicate overlap)
    "w4": 0.10,   # |l_bar_1 - l_bar_2|            (clause length)
    "w5": 0.10,   # |complexity_1 - complexity_2|
    "w6": 0.40,   # 1 - Jaccard(motifs_1, motifs_2) (topology — highest weight)
}


# ─── Core distance function ───────────────────────────────────────────────────

def ilp_signature_distance(
        S1: Dict[str, Any],
        S2: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute d(S1, S2) as defined in paper Eq. 26 / Appendix F.

    Parameters
    ----------
    S1, S2 : ILP signature dicts, expected keys:
        n_clauses  : int
        avg_depth  : float   (d-bar)
        predicates : list[str]   (P_H)
        avg_length : float   (l-bar)
        complexity : float

    weights : override default weights {w1..w5}

    Returns
    -------
    float in [0, 1+] — lower is more similar.
    d=0 means identical signatures.
    """
    w = _DEFAULT_WEIGHTS.copy()
    if weights:
        w.update(weights)

    # ── n_clauses ─────────────────────────────────────────────────────────
    n1 = S1.get("n_clauses", 1)
    n2 = S2.get("n_clauses", 1)
    delta_n = abs(n1 - n2)

    # ── avg_depth (d-bar) ─────────────────────────────────────────────────
    d1 = float(S1.get("avg_depth", 1.0))
    d2 = float(S2.get("avg_depth", 1.0))
    delta_d = abs(d1 - d2) / max(max(d1, d2), 1.0)   # normalise by max depth

    # ── predicate overlap (Jaccard) ───────────────────────────────────────
    p1 = set(S1.get("predicates", []))
    p2 = set(S2.get("predicates", []))
    jaccard = _jaccard(p1, p2)
    pred_dist = 1.0 - jaccard

    # ── avg_length (l-bar) ────────────────────────────────────────────────
    l1 = float(S1.get("avg_length", 1.0))
    l2 = float(S2.get("avg_length", 1.0))
    delta_l = abs(l1 - l2) / max(max(l1, l2), 1.0)

    # ── complexity ────────────────────────────────────────────────────────
    c1 = float(S1.get("complexity", 0.0))
    c2 = float(S2.get("complexity", 0.0))
    max_c = max(c1, c2, 1.0)
    delta_c = abs(c1 - c2) / max_c

    # ── motif topology (Jaccard on motif sets) ───────────────────────────────
    m1 = set(S1.get("motifs", []))
    m2 = set(S2.get("motifs", []))
    motif_dist = 1.0 - _jaccard(m1, m2) if (m1 or m2) else 0.0

    d = (w["w1"] * delta_n
         + w["w2"] * delta_d
         + w["w3"] * pred_dist
         + w["w4"] * delta_l
         + w["w5"] * delta_c
         + w.get("w6", 0.40) * motif_dist)

    return round(d, 6)


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ─── Component-level breakdown ────────────────────────────────────────────────

def signature_distance_breakdown(
        S1: Dict[str, Any],
        S2: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Return d(S1, S2) with per-component contributions and shared predicates.
    Useful for debugging and the paper's Table of cross-task comparisons.
    """
    w = _DEFAULT_WEIGHTS.copy()
    if weights:
        w.update(weights)

    n1, n2 = S1.get("n_clauses", 1),      S2.get("n_clauses", 1)
    d1, d2 = float(S1.get("avg_depth", 1.0)),  float(S2.get("avg_depth", 1.0))
    l1, l2 = float(S1.get("avg_length", 1.0)), float(S2.get("avg_length", 1.0))
    c1, c2 = float(S1.get("complexity", 0.0)), float(S2.get("complexity", 0.0))
    p1, p2 = set(S1.get("predicates", [])),    set(S2.get("predicates", []))

    jaccard    = _jaccard(p1, p2)
    pred_dist  = 1.0 - jaccard
    delta_n    = abs(n1 - n2)
    delta_d    = abs(d1 - d2) / max(max(d1, d2), 1.0)
    delta_l    = abs(l1 - l2) / max(max(l1, l2), 1.0)
    delta_c    = abs(c1 - c2) / max(max(c1, c2), 1.0)

    m1         = set(S1.get("motifs", []))
    m2         = set(S2.get("motifs", []))
    motif_dist = 1.0 - _jaccard(m1, m2) if (m1 or m2) else 0.0

    components = {
        "n_clauses_delta":  (w["w1"] * delta_n, delta_n),
        "motif_dist":       (w.get("w6", 0.40) * motif_dist, motif_dist),
        "depth_delta":      (w["w2"] * delta_d, delta_d),
        "predicate_dist":   (w["w3"] * pred_dist, pred_dist),
        "length_delta":     (w["w4"] * delta_l, delta_l),
        "complexity_delta": (w["w5"] * delta_c, delta_c),
    }
    total = sum(v for v, _ in components.values())

    return {
        "distance":           round(total, 6),
        "jaccard_similarity": round(jaccard, 4),
        "shared_predicates":  sorted(p1 & p2),
        "only_in_S1":         sorted(p1 - p2),
        "only_in_S2":         sorted(p2 - p1),
        "shared_motifs":      sorted(m1 & m2),
        "only_motifs_S1":     sorted(m1 - m2),
        "only_motifs_S2":     sorted(m2 - m1),
        "components":         {k: round(v, 6) for k, (v, _) in components.items()},
        "raw_deltas":         {k: round(raw, 6) for k, (_, raw) in components.items()},
    }


# ─── Transfer pair check (Def. 5) ────────────────────────────────────────────

def is_transfer_pair(
        sig_alpha:          Dict[str, Any],
        sig_beta:           Dict[str, Any],
        threshold:          float = 0.30,
        weights:            Optional[Dict[str, float]] = None,
        # Gap 7: full Def. 5 conditions
        acc_t_alpha:        Optional[float] = None,  # σ_sem for circuit α (AccT)
        acc_t_beta:         Optional[float] = None,  # σ_sem for circuit β (AccT)
        acc_t_threshold:    float = 0.10,            # |AccT_α - AccT_β| < ε_σ
        kappa_alpha:        Optional[str]  = None,   # operation type κ for α
        kappa_beta:         Optional[str]  = None,   # operation type κ for β
        tau_alpha:          Optional[str]  = None,   # logical category τ for α
        tau_beta:           Optional[str]  = None,   # logical category τ for β
) -> Tuple[bool, float]:
    """
    Check whether (C_alpha, C_beta) form a valid transfer pair (paper Def. 5).

    Paper Def. 5 requires ALL of:
        1. d(S_ILP_sem(φ_α), S_ILP_sem(φ_β)) < δ_ILP        (signature distance)
        2. |AccT(C_α) - AccT(C_β)| < ε_σ                     (σ_sem ≃ behavioural)
        3. κ_α = κ_β                                           (same operation type)
        4. τ_α = τ_β                                           (same logical category)

    Conditions 2-4 are checked only when the corresponding values are provided.
    If omitted, those conditions are treated as satisfied (conservative / backward-compat).

    Parameters
    ----------
    sig_alpha, sig_beta   : ILP semantic signatures S_ILP_sem
    threshold             : δ_ILP — signature distance threshold (default 0.30)
    weights               : override distance weight dict
    acc_t_alpha/beta      : AccT(C,φ) for each circuit (from layer1.metrics)
    acc_t_threshold       : ε_σ — max allowed |AccT_α - AccT_β| (default 0.10)
    kappa_alpha/beta      : operation_type κ string (from layer0)
    tau_alpha/beta        : logical_category τ string (from layer0)

    Returns
    -------
    (is_pair: bool, distance: float, reasons: dict)
      is_pair  — True iff all checked conditions pass
      distance — raw signature distance
      reasons  — dict of per-condition verdicts for diagnostics
    """
    d = ilp_signature_distance(sig_alpha, sig_beta, weights)

    reasons: Dict[str, Any] = {}

    # Condition 1: signature distance
    sig_ok = d < threshold
    reasons["signature_distance"] = {"value": round(d, 4), "threshold": threshold, "pass": sig_ok}

    # Condition 2: σ_sem ≃ (behavioural similarity via AccT)
    if acc_t_alpha is not None and acc_t_beta is not None:
        acc_diff = abs(acc_t_alpha - acc_t_beta)
        acc_ok = acc_diff < acc_t_threshold
        reasons["sigma_sem"] = {
            "acc_t_alpha": round(acc_t_alpha, 4),
            "acc_t_beta":  round(acc_t_beta, 4),
            "diff":        round(acc_diff, 4),
            "threshold":   acc_t_threshold,
            "pass":        acc_ok,
        }
    else:
        acc_ok = True   # not checked — treated as satisfied
        reasons["sigma_sem"] = {"pass": True, "note": "not checked (AccT not provided)"}

    # Condition 3: κ ≃ (same operation type)
    if kappa_alpha is not None and kappa_beta is not None:
        kappa_ok = (kappa_alpha.strip().lower() == kappa_beta.strip().lower())
        reasons["kappa"] = {
            "kappa_alpha": kappa_alpha,
            "kappa_beta":  kappa_beta,
            "pass":        kappa_ok,
        }
    else:
        kappa_ok = True
        reasons["kappa"] = {"pass": True, "note": "not checked (operation_type not provided)"}

    # Condition 4: τ ≃ (same logical category)
    if tau_alpha is not None and tau_beta is not None:
        tau_ok = (tau_alpha.strip().lower() == tau_beta.strip().lower())
        reasons["tau"] = {
            "tau_alpha": tau_alpha,
            "tau_beta":  tau_beta,
            "pass":      tau_ok,
        }
    else:
        tau_ok = True
        reasons["tau"] = {"pass": True, "note": "not checked (logical_category not provided)"}

    is_pair = sig_ok and acc_ok and kappa_ok and tau_ok
    return is_pair, d, reasons


# ─── Pairwise distance matrix ────────────────────────────────────────────────

def signature_distance_matrix(
        signatures: List[Dict[str, Any]],
        labels:     Optional[List[str]] = None,
        weights:    Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute pairwise ILP signature distances for a list of signatures.

    Returns
    -------
    {
      "labels":  ["task1", "task2", ...],
      "matrix":  [[0.0, d12, ...], [d21, 0.0, ...], ...],
      "pairs":   [{"i": 0, "j": 1, "label_i": "task1",
                   "label_j": "task2", "distance": 0.14}, ...]
    }
    """
    n = len(signatures)
    if labels is None:
        labels = [f"S{i}" for i in range(n)]

    matrix = [[0.0] * n for _ in range(n)]
    pairs  = []

    for i in range(n):
        for j in range(i + 1, n):
            d = ilp_signature_distance(signatures[i], signatures[j], weights)
            matrix[i][j] = d
            matrix[j][i] = d
            pairs.append({
                "i": i, "j": j,
                "label_i": labels[i],
                "label_j": labels[j],
                "distance": d,
            })

    pairs.sort(key=lambda p: p["distance"])
    return {"labels": labels, "matrix": matrix, "pairs": pairs}


# ─── Closest match search ────────────────────────────────────────────────────

def closest_match(
        query: Dict[str, Any],
        registry: List[Dict[str, Any]],
        registry_labels: Optional[List[str]] = None,
        threshold: float = 0.30,
        weights:   Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Find the most similar signature in a registry (paper §3.6 Stage 1).

    Parameters
    ----------
    query            : ILP signature to match
    registry         : list of known ILP signatures
    registry_labels  : human-readable labels (optional)
    threshold        : delta_ILP — accept only d < threshold
    weights          : weight override

    Returns
    -------
    {
      "best_idx":   int or None,
      "best_label": str or None,
      "distance":   float,
      "accepted":   bool,         # True if d < threshold
      "all_distances": [...],
    }
    """
    if not registry:
        return {"best_idx": None, "best_label": None,
                "distance": float("inf"), "accepted": False, "all_distances": []}

    if registry_labels is None:
        registry_labels = [f"S{i}" for i in range(len(registry))]

    distances = [
        (i, lbl, ilp_signature_distance(query, sig, weights))
        for i, (lbl, sig) in enumerate(zip(registry_labels, registry))
    ]
    distances.sort(key=lambda x: x[2])

    best_idx, best_label, best_d = distances[0]
    accepted = best_d < threshold

    return {
        "best_idx":      best_idx,
        "best_label":    best_label,
        "distance":      best_d,
        "accepted":      accepted,
        "threshold":     threshold,
        "all_distances": [
            {"idx": i, "label": lbl, "distance": d}
            for i, lbl, d in distances
        ],
    }


# ─── Pretty-print ─────────────────────────────────────────────────────────────

def print_distance_report(
        S1: Dict[str, Any],
        S2: Dict[str, Any],
        label1: str = "S1",
        label2: str = "S2",
        weights: Optional[Dict[str, float]] = None,
) -> None:
    """Print a formatted comparison of two signatures with per-component breakdown."""
    bd = signature_distance_breakdown(S1, S2, weights)
    print(f"\n{'='*60}")
    print(f"Signature Distance: {label1} vs {label2}")
    print(f"{'='*60}")
    print(f"  Total d(S1,S2)   = {bd['distance']:.4f}")
    print(f"  Jaccard(P1,P2)   = {bd['jaccard_similarity']:.4f}")
    print(f"  Shared predicates: {bd['shared_predicates']}")
    print(f"  Only in {label1:5s}:   {bd['only_in_S1']}")
    print(f"  Only in {label2:5s}:   {bd['only_in_S2']}")
    print(f"\n  Component contributions:")
    for name, contrib in bd["components"].items():
        raw = bd["raw_deltas"][name]
        print(f"    {name:<22} contrib={contrib:.4f}  (raw delta={raw:.4f})")
    print(f"{'='*60}")
