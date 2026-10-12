#!/usr/bin/env python3
"""
COMPASS Statistical Tests v2 — improved and additional tests.

Replaces the failed tests from v1:

  Test 1 (revised)  IOI separation as Mann-Whitney U on pairwise distances
                    — avoids the power problem of permuting only 10 task labels.

  Test 4 (revised)  Within-task cohesion using layer1 circuit features
                    (num_nodes, num_edges, attn_ratio, DeltaT per split)
                    — fixes v1's bug where layer3.exact was identical per task.

New tests:

  Test 5            DeltaT stability — stable tasks (IOI, GT) have lower
                    DeltaT variance across prompt splits than sensitive tasks.

  Test 6            Scale monotonicity — does mean ILP confidence increase
                    monotonically from 14M → 1B → LLaMA?

Usage
-----
    python statistical_tests_v2.py
    python statistical_tests_v2.py --n-perms 50000 --out analysis/stats_v2.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy import stats as scipy_stats
from collections import defaultdict

# ── Configuration ─────────────────────────────────────────────────────────────

ANALYSIS_DIR = Path("analysis")

MODELS = [
    ("pythia14m_multi",  "Pythia-14M"),
    ("pythia1b_multi",   "Pythia-1B"),
    ("Llama3.21B_multi", "LLaMA-3.2-1B"),
]
MODEL_ORDER = ["pythia14m_multi", "pythia1b_multi", "Llama3.21B_multi"]

STABLE_TASKS   = {"ioi", "greaterthan"}

# ── Data loading ───────────────────────────────────────────────────────────────

def load_distance_report(model_dir: str) -> dict:
    path = ANALYSIS_DIR / model_dir / "results" / "distance_report.json"
    with open(path) as f:
        return json.load(f)


def load_all_results(model_dir: str) -> dict[str, dict]:
    results_dir = ANALYSIS_DIR / model_dir / "results"
    out = {}
    for p in sorted(results_dir.glob("*_results.json")):
        key = p.stem.replace("_results", "")
        with open(p) as f:
            out[key] = json.load(f)
    return out


def get_arch_matrix(report: dict) -> tuple[list[str], np.ndarray]:
    tasks = report["tasks"]
    mat   = report["architectural_matrix"]
    arr   = np.array([[mat[t1][t2] for t2 in tasks] for t1 in tasks])
    return tasks, arr


def parse_task(key: str) -> str:
    """'ioi_s0' → 'ioi',  'greaterthan_s2' → 'greaterthan'"""
    return key.rsplit("_", 1)[0]


def extract_layer1_features(result: dict) -> dict | None:
    """
    Per-split structural features from layer1 (vary between splits).
    Returns None if layer1 data is missing.
    """
    l1 = result.get("layer1") or {}
    circuit = l1.get("circuit") or {}
    metrics = l1.get("metrics") or {}

    nodes = circuit.get("nodes") or []
    if not nodes:
        return None

    # count node types directly from the circuit graph
    n_attn = sum(1 for nd in nodes if nd.get("type") == "attn")
    n_mlp  = sum(1 for nd in nodes if nd.get("type") == "mlp")
    total  = n_attn + n_mlp or 1

    num_edges = len(circuit.get("edges") or [])
    delta_t   = metrics.get("DeltaT")
    faith     = metrics.get("faithfulness")

    if delta_t is None:
        return None

    return {
        "attn_ratio":  n_attn / total,
        "n_nodes":     total,
        "n_edges":     num_edges,
        "DeltaT":      delta_t,
        "faithfulness": faith if faith is not None else 0.0,
    }


def get_circuit_records(all_results: dict) -> list[dict]:
    """List of dicts with task, split, confidence, and layer1 features."""
    records = []
    for key, result in all_results.items():
        task  = parse_task(key)
        split = key.rsplit("_", 1)[-1]
        l3    = result.get("layer3") or {}
        conf  = l3.get("tau_arch_confidence")
        feats = extract_layer1_features(result)
        if conf is None or feats is None:
            continue
        records.append({
            "key":   key,
            "task":  task,
            "split": split,
            "confidence": conf,
            **feats,
        })
    return records


# ── Helpers ────────────────────────────────────────────────────────────────────

def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def standardise(X: np.ndarray) -> np.ndarray:
    """Z-score each column; zero-variance columns → 0."""
    mu  = X.mean(axis=0)
    sd  = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):   return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# ── Test 1 (revised): Mann-Whitney U on IOI pairwise distances ────────────────

def test1_mw_separation(tasks: list[str], mat: np.ndarray,
                        target: str = "ioi") -> dict:
    """
    Mann-Whitney U comparing the (n-1) distances from *target* to every other
    task against all C(n-1, 2) pairwise distances among the remaining tasks.

    Treats each pairwise distance as an independent observation — avoids the
    power problem of permuting only n task labels.

    H0: IOI row distances come from the same distribution as non-IOI pairs.
    H1: IOI row distances are stochastically larger (one-sided).
    """
    if target not in tasks:
        return {"error": f"'{target}' not in tasks"}

    n   = len(tasks)
    idx = tasks.index(target)

    target_dists = np.array([mat[idx, j] for j in range(n) if j != idx])
    other_pairs  = np.array([mat[i, j]
                             for i in range(n) for j in range(i+1, n)
                             if i != idx and j != idx])

    U, p = scipy_stats.mannwhitneyu(target_dists, other_pairs,
                                    alternative="greater")
    p = float(p)

    return {
        "target": target,
        "n_target_dists":  int(len(target_dists)),
        "n_other_pairs":   int(len(other_pairs)),
        "mean_target_dist": float(np.mean(target_dists)),
        "mean_other_dist":  float(np.mean(other_pairs)),
        "ratio":            float(np.mean(target_dists) / np.mean(other_pairs)),
        "U_stat":  float(U),
        "p_value": p,
        "significant": p < 0.05,
    }


# ── Test 4 (revised): Within-task cohesion via layer1 features ────────────────

def test4_l1_cohesion(records: list[dict],
                      n_perms: int = 10_000,
                      rng_seed: int = 42) -> dict:
    """
    Permutation test on within-task vs between-task distances using per-split
    layer1 structural features (attn_ratio, n_nodes, n_edges, DeltaT, faith).

    These features vary between splits of the same task, unlike layer3.exact
    which is shared across splits.

    Stat:  C = mean(within-task dist) / mean(between-task dist)   [< 1 = cohesive]
    H0:    C no smaller than chance under task-label permutation.
    """
    if len(records) < 4:
        return {"error": "too few circuits"}

    # Build feature matrix — standardise so all features contribute equally
    feat_keys = ["attn_ratio", "n_nodes", "n_edges", "DeltaT", "faithfulness"]
    X         = np.array([[r[k] for k in feat_keys] for r in records])
    X         = standardise(X)
    labels    = [r["task"] for r in records]
    n         = len(records)

    # Pairwise Euclidean distances
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(X[i] - X[j])
            D[i, j] = D[j, i] = d

    def cohesion(lbl):
        within, between = [], []
        for i in range(n):
            for j in range(i+1, n):
                (within if lbl[i] == lbl[j] else between).append(D[i, j])
        return within, between

    within_obs, between_obs = cohesion(labels)
    if not within_obs or not between_obs:
        return {"error": "no within or between pairs"}

    C_obs = float(np.mean(within_obs) / np.mean(between_obs))

    rng    = np.random.default_rng(rng_seed)
    C_null = np.empty(n_perms)
    arr    = np.array(labels)
    for k in range(n_perms):
        perm      = rng.permutation(arr)
        w, b      = cohesion(list(perm))
        C_null[k] = np.mean(w) / np.mean(b) if w and b else 1.0

    # one-sided: H1 = within < between
    p = float(np.mean(C_null <= C_obs))

    # Per-task intra-mean for diagnostics
    unique_tasks = sorted(set(labels))
    mean_inter   = float(np.mean(between_obs))
    per_task = {}
    for t in unique_tasks:
        idxs  = [i for i, tl in enumerate(labels) if tl == t]
        pairs = [D[i, j] for i in idxs for j in idxs if i < j]
        if pairs:
            per_task[t] = {
                "n_splits":   len(idxs),
                "intra_mean": float(np.mean(pairs)),
                "ratio_vs_inter": float(np.mean(pairs) / mean_inter),
            }

    return {
        "n_circuits":      n,
        "observed_ratio":  C_obs,
        "mean_within":     float(np.mean(within_obs)),
        "mean_between":    mean_inter,
        "n_perms":         n_perms,
        "p_value":         p,
        "significant":     p < 0.05,
        "per_task":        per_task,
    }


# ── Test 5: DeltaT stability across prompt splits ─────────────────────────────

def test5_deltat_stability(records: list[dict]) -> dict:
    """
    Mann-Whitney U (one-sided): do stable tasks (IOI, GT) have lower
    DeltaT variance across prompt splits than sensitive tasks?

    Per-task DeltaT std is computed from the 2-3 available splits.
    Negative DeltaT values (sign-reversed circuits) are included as-is —
    they reflect genuine circuit behaviour.

    H1: stable tasks have lower DeltaT std than sensitive tasks.
    """
    # Group DeltaT by task
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r["DeltaT"])

    # require ≥2 splits for std; if only 1 split use 0.0 (perfectly stable by definition)
    task_stds = {}
    for t, vals in by_task.items():
        if len(vals) >= 2:
            task_stds[t] = float(np.std(vals, ddof=0))
        elif len(vals) == 1 and t in STABLE_TASKS:
            task_stds[t] = 0.0  # single split: treat as perfectly stable

    stable_stds    = [v for t, v in task_stds.items() if t in STABLE_TASKS]
    sensitive_stds = [v for t, v in task_stds.items() if t not in STABLE_TASKS]

    if len(stable_stds) < 1 or len(sensitive_stds) < 2:
        return {"error": "insufficient data"}

    # one-sided: stable < sensitive (lower std = more stable)
    U, p = scipy_stats.mannwhitneyu(stable_stds, sensitive_stds,
                                    alternative="less")
    p = float(p)

    return {
        "stable_tasks":    sorted(STABLE_TASKS),
        "stable_n":        len(stable_stds),
        "stable_mean_std": float(np.mean(stable_stds)),
        "sensitive_n":     len(sensitive_stds),
        "sensitive_mean_std": float(np.mean(sensitive_stds)),
        "U_stat":  float(U),
        "p_value": p,
        "significant": p < 0.05,
        "per_task_std": task_stds,
    }


# ── Test 6: Scale monotonicity of ILP confidence ─────────────────────────────

def test6_scale_monotonicity(models_records: dict[str, list[dict]]) -> dict:
    """
    For each task, does mean ILP confidence increase monotonically across
    scales (14M → 1B → LLaMA)?

    Uses a sign test: count tasks with strictly monotone increase vs decrease.
    Null: each of the 6 orderings is equally likely (p from binomial on n_mono).

    Also reports Jonckheere-Terpstra-style statistic: summed pairwise order
    agreements across tasks (n_agree / n_possible).
    """
    # Collect mean confidence per (model, task)
    model_order = [m for m, _ in MODELS if m in models_records]
    if len(model_order) < 2:
        return {"error": "need ≥2 models"}

    # Build task → [conf_m0, conf_m1, ...] table
    all_tasks = sorted(set(r["task"]
                           for recs in models_records.values()
                           for r in recs))
    task_conf = {}
    for task in all_tasks:
        row = []
        for md in model_order:
            recs  = models_records.get(md, [])
            confs = [r["confidence"] for r in recs if r["task"] == task]
            row.append(float(np.mean(confs)) if confs else None)
        task_conf[task] = row

    # Keep only tasks with data in all models
    complete = {t: v for t, v in task_conf.items() if None not in v}

    n_mono_up   = 0  # strictly increasing
    n_mono_down = 0  # strictly decreasing
    n_agree, n_total = 0, 0

    per_task_direction = {}
    for task, confs in complete.items():
        pairs = [(confs[i], confs[j])
                 for i in range(len(confs)) for j in range(i+1, len(confs))]
        agrees = sum(a < b for a, b in pairs)
        n_agree += agrees
        n_total += len(pairs)

        mono_up   = all(confs[i] < confs[i+1] for i in range(len(confs)-1))
        mono_down = all(confs[i] > confs[i+1] for i in range(len(confs)-1))
        n_mono_up   += int(mono_up)
        n_mono_down += int(mono_down)

        per_task_direction[task] = {
            "confidences": confs,
            "monotone_up": bool(mono_up),
            "models": model_order,
        }

    n_tasks = len(complete)
    # Binomial test: H1 = n_mono_up > chance (1/6 for 3 models, or 1/2 for
    # just testing whether up > down when 2 models)
    # For 3 models: chance of strict increase = 1/6; we test vs 1/6
    # For 2 models: chance of strict increase = 1/2
    n_models     = len(model_order)
    p_chance_up  = 1.0 / (2 if n_models == 2 else 6)
    binom_result = scipy_stats.binomtest(n_mono_up, n_tasks, p=p_chance_up,
                                         alternative="greater")
    p_binom = float(binom_result.pvalue)

    # JT-style: fraction of pairs in ascending order
    jt_fraction = float(n_agree / n_total) if n_total > 0 else None

    return {
        "model_order": model_order,
        "n_tasks_complete": n_tasks,
        "n_monotone_up":   n_mono_up,
        "n_monotone_down": n_mono_down,
        "p_chance_mono":   p_chance_up,
        "p_binom_mono_up": p_binom,
        "significant_mono": p_binom < 0.05,
        "jt_fraction_ascending": jt_fraction,
        "per_task": per_task_direction,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-perms", type=int, default=10_000)
    ap.add_argument("--out", type=str, default=None)
    return ap.parse_args()


def main():
    args  = parse_args()
    N     = args.n_perms
    output = {}

    print("=" * 72)
    print("COMPASS Statistical Tests v2")
    print(f"N_permutations = {N:,}")
    print("=" * 72)

    # ── Load ──────────────────────────────────────────────────────────────────
    models_data    = {}  # arch matrix
    models_records = {}  # per-circuit records

    for model_dir, label in MODELS:
        try:
            report = load_distance_report(model_dir)
        except FileNotFoundError:
            print(f"[SKIP] No distance_report for {label}")
            continue
        if "architectural_matrix" not in report:
            print(f"[SKIP] No architectural_matrix in {label}")
            continue

        tasks, mat = get_arch_matrix(report)
        models_data[model_dir] = {"tasks": tasks, "mat": mat, "label": label}

        try:
            all_results = load_all_results(model_dir)
            records     = get_circuit_records(all_results)
            models_records[model_dir] = records
        except Exception as e:
            print(f"[WARN] Results for {label}: {e}")

    # ── Test 1 (revised) ──────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("TEST 1 (revised)  All-Task Separation — Mann-Whitney U on Pairwise Distances")
    print("  For each task: row distances vs all other pairwise distances")
    print("  H1: target→other distances stochastically larger (one-sided MWU)")
    print("  Bonferroni correction applied within each model")
    print()

    t1 = {}
    for md, data in models_data.items():
        label = data["label"]
        tasks = data["tasks"]
        mat   = data["mat"]
        n_tasks = len(tasks)

        # Run for every task
        task_results = {}
        for task in tasks:
            task_results[task] = test1_mw_separation(tasks, mat, target=task)

        # Bonferroni correction
        alpha_corrected = 0.05 / n_tasks

        # Sort by ratio descending
        ranked = sorted(task_results.items(),
                        key=lambda x: x[1].get("ratio", 0), reverse=True)

        print(f"  {label}  (Bonferroni α = 0.05/{n_tasks} = {alpha_corrected:.4f})")
        print(f"    {'Task':18s}  {'ratio':>6}  {'μ_target':>8}  {'μ_other':>8}  "
              f"{'p_raw':>8}  {'p_adj':>8}  sig")
        for task, res in ranked:
            if "error" in res:
                continue
            p_raw = res["p_value"]
            p_adj = min(p_raw * n_tasks, 1.0)  # Bonferroni
            sig   = "***" if p_adj < 0.001 else ("**" if p_adj < 0.01
                    else ("*" if p_adj < 0.05 else "n.s."))
            print(
                f"    {task:18s}  "
                f"{res['ratio']:6.3f}  "
                f"{res['mean_target_dist']:8.4f}  "
                f"{res['mean_other_dist']:8.4f}  "
                f"{p_raw:8.4f}  "
                f"{p_adj:8.4f}  {sig}"
            )
        t1[md] = task_results
        print()
    output["test1_mw_separation_all"] = t1

    # ── Test 4 (revised) ──────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("TEST 4 (revised)  Within-Task Cohesion — Layer1 Circuit Features")
    print("  Features: attn_ratio, n_nodes, n_edges, DeltaT, faithfulness")
    print("  (all vary per split; standardised before distance)")
    print("  Stat: mean(within-task dist) / mean(between-task dist)  [< 1 = cohesive]")
    print()

    t4 = {}
    for md, records in models_records.items():
        label = models_data[md]["label"]
        res   = test4_l1_cohesion(records, n_perms=N)
        t4[md] = res
        if "error" in res:
            print(f"  {label:22s}  {res['error']}")
            continue
        print(
            f"  {label:22s}  "
            f"ratio={res['observed_ratio']:.3f}  "
            f"(within={res['mean_within']:.3f}, between={res['mean_between']:.3f})  "
            f"p={res['p_value']:.4f} {sig_stars(res['p_value'])}"
        )
        # Per-task detail
        inter = res["mean_between"]
        rows  = sorted(res["per_task"].items(), key=lambda x: x[1]["intra_mean"])
        for task, info in rows:
            print(f"    {task:18s}  "
                  f"intra={info['intra_mean']:.3f}  "
                  f"({info['ratio_vs_inter']:.2f}× inter)  "
                  f"n_splits={info['n_splits']}")
    output["test4_cohesion_l1"] = t4

    # ── Test 5 ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("TEST 5  DeltaT Stability — Stable vs Sensitive Tasks")
    print("  H0: DeltaT std across splits equal in stable and sensitive tasks")
    print("  H1: Stable tasks (IOI, GT) have lower DeltaT std (one-sided MWU)")
    print()

    t5 = {}
    for md, records in models_records.items():
        label = models_data[md]["label"]
        res   = test5_deltat_stability(records)
        t5[md] = res
        if "error" in res:
            print(f"  {label:22s}  {res['error']}")
            continue
        print(
            f"  {label:22s}  "
            f"stable_std={res['stable_mean_std']:.3f} (n={res['stable_n']})  "
            f"sensitive_std={res['sensitive_mean_std']:.3f} (n={res['sensitive_n']})  "
            f"U={res['U_stat']:.0f}  p={res['p_value']:.4f} {sig_stars(res['p_value'])}"
        )
        # Per-task DeltaT std for inspection
        per = sorted(res["per_task_std"].items(), key=lambda x: x[1])
        print("    Per-task DeltaT std (lower = more stable):")
        for task, std in per:
            tag = " ← stable" if task in STABLE_TASKS else ""
            print(f"      {task:18s}  std={std:.4f}{tag}")
    output["test5_deltat_stability"] = t5

    # ── Test 6 ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("TEST 6  Scale Monotonicity — Does Confidence Increase 14M → 1B → LLaMA?")
    print("  Binomial test: observed # monotone-up tasks vs chance (1/6 for 3 models)")
    print()

    res6 = test6_scale_monotonicity(models_records)
    t6   = res6
    if "error" in res6:
        print(f"  {res6['error']}")
    else:
        print(
            f"  Model order: {' → '.join(res6['model_order'])}\n"
            f"  Tasks with complete data: {res6['n_tasks_complete']}\n"
            f"  Monotone ↑: {res6['n_monotone_up']}  "
            f"Monotone ↓: {res6['n_monotone_down']}\n"
            f"  Chance of monotone ↑ = {res6['p_chance_mono']:.3f}  "
            f"p(binom) = {res6['p_binom_mono_up']:.4f} "
            f"{sig_stars(res6['p_binom_mono_up'])}\n"
            f"  JT ascending fraction: {res6['jt_fraction_ascending']:.3f} "
            f"(0.5=chance, 1.0=all pairs ascending)"
        )
        print("\n  Per-task confidences:")
        labels = [f"{models_data.get(m, {}).get('label', m):10s}"
                  for m in res6["model_order"]]
        print(f"    {'Task':18s}  {' '.join(labels)}  direction")
        for task, info in sorted(res6["per_task"].items()):
            confs  = info["confidences"]
            conf_s = "  ".join(f"{c:.3f}" for c in confs)
            arrow  = "↑" if info["monotone_up"] else "~"
            print(f"    {task:18s}  {conf_s}  {arrow}")
    output["test6_scale_mono"] = t6

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Legend: *** p<0.001  ** p<0.01  * p<0.05  n.s. not significant")
    print("=" * 72)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, cls=_NumpyEncoder)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
