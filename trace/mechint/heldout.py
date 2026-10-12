"""
Three-fold held-out evaluation.

Fold i holds out split s_i of every task. For each task, tau_arch is learned
from the remaining training circuits (negatives = the other tasks' training
circuits) using the identical clause-selection procedure as the pooled
learner, then evaluated one-vs-rest on the held-out circuits only. The
structural baselines (WL kernel, attributed WL, feature-vector distance,
random forest, decision stump) are evaluated under the same folds, giving a
matched held-out comparison across methods.

Leakage note: stored ``layer3.motifs`` are a task-level union written by the
pooled ILP run, so baseline features recompute per-circuit motifs from the
circuit topology instead (via ``_features_from_circuit_dict``).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .storage.ilp.ilp_arch import (
    ArchitecturalILPLearner,
    _features_from_circuit_dict,
    _evaluate_arch_clause,
    _arch_signature,
)
from .storage.ilp.ilp_signature_distance import ilp_signature_distance
from .baselines_structural import (
    _find_ioi_key,
    _load_circuit_graph,
    extract_structural_features,
    wl_distance,
)


# ── Shared helpers ────────────────────────────────────────────────────────

def default_task_family(task: str) -> str:
    t = task.lower()
    if "ioi" in t:
        return "IOI"
    if "greater" in t or t == "gt":
        return "GreaterThan"
    if "coloredobject" in t:
        return "ColoredObjects"
    if "sva" in t:
        return "SVA"
    return "SemanticRoles"


def separation_score(matrix: Dict[str, Dict[str, float]],
                     tasks: List[str]) -> float:
    """mean(IOI-to-others) / mean(non-IOI pairwise); NaN without an IOI task."""
    ioi_key = _find_ioi_key(tasks)
    if not ioi_key:
        return float("nan")
    ioi_dists = [matrix[ioi_key][t] for t in tasks if t != ioi_key]
    other = [matrix[t1][t2]
             for i, t1 in enumerate(tasks) if t1 != ioi_key
             for t2 in tasks[i + 1:] if t2 != ioi_key]
    if not ioi_dists or not other:
        return float("nan")
    mean_other = float(np.mean(other))
    return float(np.mean(ioi_dists)) / mean_other if mean_other > 0 else float("inf")


def _extract(record: Dict[str, Any],
             model_n_layers: int = 0,
             model_n_heads: int = 0) -> Tuple[Dict, Tuple[List, List]]:
    """(aggregate feature dict, raw (nodes, edges)) for one results record."""
    c = (record.get("layer1") or {}).get("circuit", {})
    feats = _features_from_circuit_dict(
        c, model_n_layers=model_n_layers, model_n_heads=model_n_heads)
    return feats, (c.get("nodes", []), c.get("edges", []))


def _baseline_feature_vector(record: Dict[str, Any],
                             inferred_motifs: List[str]) -> Dict[str, float]:
    """
    Structural feature dict with motif indicators recomputed per circuit
    (topology-inferred) instead of read from the pooled layer3 union.
    """
    feats = extract_structural_features(record)
    motif_set = set(inferred_motifs)
    for k in list(feats):
        if k.startswith("motif_"):
            feats[k] = 1.0 if k[len("motif_"):] in motif_set else 0.0
    return feats


def _feature_distance_matrix(vecs: Dict[str, Dict[str, float]]
                             ) -> Dict[str, Dict[str, float]]:
    """Standardised euclidean distance over feature dicts, normalised to [0,1]."""
    tasks = sorted(vecs.keys())
    names = sorted(vecs[tasks[0]].keys())
    X = np.array([[vecs[t][f] for f in names] for t in tasks])
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0
    X = (X - mu) / sigma
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    if D.max() > 0:
        D = D / D.max()
    return {t1: {t2: round(float(D[i, j]), 4) for j, t2 in enumerate(tasks)}
            for i, t1 in enumerate(tasks)}


# ── Main protocol ─────────────────────────────────────────────────────────

def three_fold_heldout(
    records_by_task: Dict[str, Dict[int, Dict[str, Any]]],
    n_folds: int = 3,
    task_families: Optional[Dict[str, str]] = None,
    model_n_layers: int = 0,
    model_n_heads: int = 0,
    wl_iterations: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run the fold-by-split held-out protocol.

    Parameters
    ----------
    records_by_task : {task: {split_idx: results_json_record}}
    n_folds         : number of splits (fold i holds out split i everywhere)

    Returns a report dict with per_fold results and aggregate mean/std for:
    ILP held-out one-vs-rest F1 and coverage (per task), signature-distance
    separation per method, and RF / decision-stump held-out accuracy.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    tasks = sorted(records_by_task.keys())
    families = task_families or {t: default_task_family(t) for t in tasks}
    learner = ArchitecturalILPLearner()

    # Pre-extract features/raw once per circuit
    feats: Dict[str, Dict[int, Dict]] = {}
    raws: Dict[str, Dict[int, Tuple]] = {}
    for t, splits in records_by_task.items():
        feats[t], raws[t] = {}, {}
        for s, rec in splits.items():
            f, r = _extract(rec, model_n_layers, model_n_heads)
            feats[t][s], raws[t][s] = f, r

    per_fold: List[Dict[str, Any]] = []
    # Pooled ILP counts across folds, per task
    pooled = {t: {"tp": 0, "fn": 0, "fp": 0, "eval_folds": 0} for t in tasks}
    skipped: List[str] = []

    for fold in range(n_folds):
        train = {t: [s for s in sorted(records_by_task[t]) if s != fold]
                 for t in tasks}
        held = {t: fold for t in tasks if fold in records_by_task[t]}

        # ── ILP: learn per task on train circuits, evaluate on held-out ──
        clauses: Dict[str, str] = {}
        signatures: Dict[str, Dict] = {}
        ilp_rows: Dict[str, Dict[str, Any]] = {}

        for t in tasks:
            if len(train[t]) < 2 or t not in held:
                skipped.append(f"fold{fold}:{t} "
                               f"(train={len(train[t])}, held={'y' if t in held else 'n'})")
                continue
            pos_feats = [feats[t][s] for s in train[t]]
            pos_raw = [raws[t][s] for s in train[t]]
            neg_feats = [feats[o][s] for o in tasks if o != t for s in train[o]]
            neg_raw = [raws[o][s] for o in tasks if o != t for s in train[o]]

            clause, pool, _ = learner._select_best_clause(
                t, pos_feats, neg_feats, pos_raw, neg_raw)
            clauses[t] = clause

            pos_motifs: List[str] = []
            seen = set()
            for f in pos_feats:
                for m in f.get("motifs", []):
                    if m not in seen:
                        pos_motifs.append(m)
                        seen.add(m)
            signatures[t] = _arch_signature(clause, motifs=pos_motifs)

        for t, clause in clauses.items():
            covered = _evaluate_arch_clause(
                clause, feats[t][held[t]], raw=raws[t][held[t]])
            fp = sum(
                1 for o in tasks if o != t and o in held
                if _evaluate_arch_clause(clause, feats[o][held[o]],
                                         raw=raws[o][held[o]]))
            tp = 1 if covered else 0
            n_neg = sum(1 for o in tasks if o != t and o in held)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec_ = float(tp)  # single held-out positive
            f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0.0
            ilp_rows[t] = {
                "clause": clause,
                "covered": bool(covered),
                "fp": fp,
                "n_heldout_neg": n_neg,
                "precision": round(prec, 4),
                "f1": round(f1, 4),
            }
            pooled[t]["tp"] += tp
            pooled[t]["fn"] += 1 - tp
            pooled[t]["fp"] += fp
            pooled[t]["eval_folds"] += 1

        # ── Signature-distance separation on this fold's learned clauses ──
        sig_tasks = sorted(signatures.keys())
        ilp_matrix = {
            a: {b: (0.0 if a == b else
                    round(ilp_signature_distance(signatures[a], signatures[b]), 4))
                for b in sig_tasks}
            for a in sig_tasks
        }

        # ── Baselines on the held-out circuits only ──────────────────────
        held_tasks = sorted(held.keys())
        graphs = {t: _load_circuit_graph(records_by_task[t][held[t]])
                  for t in held_tasks}

        def _wl_matrix(attributed: bool) -> Dict[str, Dict[str, float]]:
            return {a: {b: (0.0 if a == b else
                            round(wl_distance(graphs[a], graphs[b],
                                              wl_iterations,
                                              attributed=attributed), 4))
                        for b in held_tasks}
                    for a in held_tasks}

        wl_matrix = _wl_matrix(False)
        wl_attr_matrix = _wl_matrix(True)

        held_vecs = {t: _baseline_feature_vector(
                         records_by_task[t][held[t]],
                         feats[t][held[t]].get("motifs", []))
                     for t in held_tasks}
        feat_matrix = _feature_distance_matrix(held_vecs)

        fold_sep = {
            "ilp": separation_score(ilp_matrix, sig_tasks),
            "wl": separation_score(wl_matrix, held_tasks),
            "wl_attributed": separation_score(wl_attr_matrix, held_tasks),
            "feat_euclidean": separation_score(feat_matrix, held_tasks),
        }

        # ── RF / decision stump: fit on train circuits, predict held-out ──
        feat_names = sorted(next(iter(held_vecs.values())).keys())

        def _vec(t, s):
            v = _baseline_feature_vector(records_by_task[t][s],
                                         feats[t][s].get("motifs", []))
            return [v[f] for f in feat_names]

        X_tr, y_fam_tr, y_task_tr = [], [], []
        for t in tasks:
            for s in train[t]:
                X_tr.append(_vec(t, s))
                y_fam_tr.append(families[t])
                y_task_tr.append(t)
        X_te = [_vec(t, held[t]) for t in held_tasks]
        y_fam_te = [families[t] for t in held_tasks]
        y_task_te = list(held_tasks)

        def _fit_acc(clf, y_tr, y_te):
            clf.fit(X_tr, y_tr)
            return float(np.mean(np.array(clf.predict(X_te)) == np.array(y_te)))

        # Family-level (3-class; easy with 10/12 tasks in one family) and
        # task-level (one class per task; the fair analogue of the one-vs-rest
        # clause learning)
        rf_acc = _fit_acc(RandomForestClassifier(n_estimators=100,
                                                 random_state=seed),
                          y_fam_tr, y_fam_te)
        rf_task_acc = _fit_acc(RandomForestClassifier(n_estimators=100,
                                                      random_state=seed),
                               y_task_tr, y_task_te)
        stump_acc = _fit_acc(DecisionTreeClassifier(max_depth=1,
                                                    random_state=seed),
                             y_fam_tr, y_fam_te)
        stump_task_acc = _fit_acc(DecisionTreeClassifier(max_depth=1,
                                                         random_state=seed),
                                  y_task_tr, y_task_te)

        per_fold.append({
            "fold": fold,
            "ilp": ilp_rows,
            "separation": {k: (round(v, 4) if np.isfinite(v) else None)
                           for k, v in fold_sep.items()},
            "rf_heldout_acc": round(rf_acc, 4),
            "rf_heldout_task_acc": round(rf_task_acc, 4),
            "stump_heldout_acc": round(stump_acc, 4),
            "stump_heldout_task_acc": round(stump_task_acc, 4),
        })

    # ── Aggregate ─────────────────────────────────────────────────────────
    def _mean_std(vals: List[float]) -> Dict[str, Any]:
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            return {"mean": None, "std": None, "folds": []}
        return {"mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "folds": [round(float(v), 4) for v in vals]}

    ilp_task_agg: Dict[str, Any] = {}
    for t in tasks:
        p = pooled[t]
        if p["eval_folds"] == 0:
            continue
        prec = p["tp"] / (p["tp"] + p["fp"]) if (p["tp"] + p["fp"]) else 0.0
        rec_ = p["tp"] / (p["tp"] + p["fn"]) if (p["tp"] + p["fn"]) else 0.0
        f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0.0
        ilp_task_agg[t] = {
            "coverage": round(rec_, 4),          # fraction of folds covered
            "pooled_precision": round(prec, 4),
            "pooled_f1": round(f1, 4),
            "per_fold_f1": [pf["ilp"][t]["f1"] for pf in per_fold
                            if t in pf["ilp"]],
        }

    aggregate = {
        "ilp_heldout": ilp_task_agg,
        "ilp_mean_pooled_f1": round(float(np.mean(
            [v["pooled_f1"] for v in ilp_task_agg.values()])), 4)
            if ilp_task_agg else None,
        "separation": {
            m: _mean_std([pf["separation"][m] for pf in per_fold])
            for m in ["ilp", "wl", "wl_attributed", "feat_euclidean"]
        },
        "classification": {
            "random_forest_family": _mean_std(
                [pf["rf_heldout_acc"] for pf in per_fold]),
            "random_forest_task": _mean_std(
                [pf["rf_heldout_task_acc"] for pf in per_fold]),
            "decision_stump_family": _mean_std(
                [pf["stump_heldout_acc"] for pf in per_fold]),
            "decision_stump_task": _mean_std(
                [pf["stump_heldout_task_acc"] for pf in per_fold]),
        },
        "skipped": skipped,
    }

    return {
        "protocol": "fold-by-split held-out (fold i holds out split i of every task)",
        "n_folds": n_folds,
        "tasks": tasks,
        "task_families": families,
        "per_fold": per_fold,
        "aggregate": aggregate,
    }
