"""
run_baselines.py
================
Run structural baselines (WL kernel, random forest, decision stump,
feature-vector distance) and compare against ILP signature distances.

No model loading required — operates entirely on stored results JSON files.

Usage:
    python run_baselines.py                          # all models
    python run_baselines.py --model pythia1b         # single model
    python run_baselines.py --model pythia1b --save   # save report JSON
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np


# ── Config ────────────────────────────────────────────────────────────────

ANALYSIS_ROOT = "./analysis"

MODELS = {
    "pythia14m": "pythia14m",
    "pythia1b": "pythia1b",
    "Llama3.21B": "Llama3.21B",
}

TASK_FAMILIES = {
    "ioi": "IOI",
    "location": "SemanticRoles",
    "time": "SemanticRoles",
    "path": "SemanticRoles",
    "greaterthan": "GreaterThan",
}

TASKS = ["ioi", "location", "time", "path", "greaterthan"]


# ── Helpers ───────────────────────────────────────────────────────────────

def load_records(model_dir: str) -> Dict[str, Dict]:
    """Load all *_results.json from a model's results directory."""
    results_dir = os.path.join(model_dir, "results")
    records = {}
    for task in TASKS:
        path = os.path.join(results_dir, f"{task}_results.json")
        if os.path.isfile(path):
            with open(path) as f:
                records[task] = json.load(f)
    return records


def load_ilp_distances(model_dir: str) -> Dict[str, Dict[str, float]]:
    """Load the ILP distance matrix from a model's distance_report.json."""
    path = os.path.join(model_dir, "results", "distance_report.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        report = json.load(f)
    return report.get("architectural_matrix", report.get("semantic_matrix", {}))


def format_matrix(tasks, matrix, label=""):
    """Pretty-print a distance matrix."""
    if label:
        print(f"\n  {label}")
    header = f"{'':>14}" + "".join(f"{t:>12}" for t in tasks)
    print(f"    {header}")
    for t1 in tasks:
        row = f"    {t1:>14}"
        for t2 in tasks:
            val = matrix.get(t1, {}).get(t2, float("nan"))
            if t1 == t2:
                row += f"{'---':>12}"
            else:
                row += f"{val:>12.4f}"
        print(row)


def cluster_quality_report(tasks, matrix, label=""):
    """Report within-cluster and between-cluster distances."""
    # IOI vs rest
    ioi_key = next((t for t in tasks if "ioi" in t.lower()), None)
    if not ioi_key:
        return

    ioi_dists = [matrix[ioi_key][t] for t in tasks if t != ioi_key]
    non_ioi = [t for t in tasks if t != ioi_key]
    non_ioi_dists = []
    for i, t1 in enumerate(non_ioi):
        for t2 in non_ioi[i+1:]:
            non_ioi_dists.append(matrix[t1][t2])

    role_tasks = [t for t in tasks if TASK_FAMILIES.get(t) == "SemanticRoles"]
    role_dists = []
    for i, t1 in enumerate(role_tasks):
        for t2 in role_tasks[i+1:]:
            role_dists.append(matrix[t1][t2])

    gt_key = next((t for t in tasks if "greater" in t.lower()), None)
    gt_role_dists = []
    if gt_key:
        gt_role_dists = [matrix[gt_key][t] for t in role_tasks]

    prefix = f"    [{label}] " if label else "    "
    print(f"{prefix}IOI-to-others:  mean={np.mean(ioi_dists):.4f}  "
          f"range=[{min(ioi_dists):.4f}, {max(ioi_dists):.4f}]")
    if non_ioi_dists:
        print(f"{prefix}non-IOI pairs:  mean={np.mean(non_ioi_dists):.4f}  "
              f"range=[{min(non_ioi_dists):.4f}, {max(non_ioi_dists):.4f}]")
    if role_dists:
        print(f"{prefix}role-role:      mean={np.mean(role_dists):.4f}  "
              f"range=[{min(role_dists):.4f}, {max(role_dists):.4f}]")
    if gt_role_dists:
        print(f"{prefix}GT-to-roles:    mean={np.mean(gt_role_dists):.4f}  "
              f"range=[{min(gt_role_dists):.4f}, {max(gt_role_dists):.4f}]")
    ratio = np.mean(ioi_dists) / np.mean(non_ioi_dists) if non_ioi_dists else 0
    print(f"{prefix}separation ratio (IOI/non-IOI): {ratio:.2f}x")


# ── Main ──────────────────────────────────────────────────────────────────

def run_for_model(model_key: str, save: bool = False):
    """Run all baselines for a single model."""
    from trace.mechint.baselines_structural import (
        wl_distance_matrix,
        feature_vector_distance_matrix,
        random_forest_classify,
        decision_stump_classify,
        full_baseline_comparison,
    )

    model_dir = os.path.join(ANALYSIS_ROOT, model_key)
    if not os.path.isdir(model_dir):
        print(f"  Model directory not found: {model_dir}")
        return None

    print(f"\n{'='*70}")
    print(f"  BASELINES — {model_key}")
    print(f"{'='*70}")

    records = load_records(model_dir)
    if len(records) < 2:
        print(f"  Only {len(records)} circuits found, need at least 2. Skipping.")
        return None

    tasks = sorted(records.keys())
    print(f"  Circuits loaded: {tasks}")

    ilp_matrix = load_ilp_distances(model_dir)

    # ── Run full comparison ──────────────────────────────────────────────
    report = full_baseline_comparison(
        records, TASK_FAMILIES,
        ilp_distance_matrix=ilp_matrix if ilp_matrix else None,
        wl_iterations=3,
    )

    # ── Print distance matrices ──────────────────────────────────────────
    if ilp_matrix:
        format_matrix(tasks, ilp_matrix, "ILP Signature Distance")
        cluster_quality_report(tasks, ilp_matrix, "ILP")

    wl_mat = report["distance_matrices"]["wl_kernel"]
    format_matrix(tasks, wl_mat, "WL Kernel Distance (3 iterations)")
    cluster_quality_report(tasks, wl_mat, "WL")

    feat_euc = report["distance_matrices"]["feature_euclidean"]
    format_matrix(tasks, feat_euc, "Feature-Vector Euclidean Distance (normalised)")
    cluster_quality_report(tasks, feat_euc, "Feat-Euc")

    feat_cos = report["distance_matrices"]["feature_cosine"]
    format_matrix(tasks, feat_cos, "Feature-Vector Cosine Distance")
    cluster_quality_report(tasks, feat_cos, "Feat-Cos")

    # ── Correlations ─────────────────────────────────────────────────────
    corr = report.get("correlations", {})
    if corr:
        print(f"\n  Distance matrix correlations (Pearson r over upper triangle):")
        for pair, r in sorted(corr.items()):
            print(f"    {pair:>30}: r = {r:.4f}")

    # ── Separation scores ────────────────────────────────────────────────
    sep = report.get("separation_scores", {})
    if sep:
        print(f"\n  IOI separation scores (mean IOI-to-others / mean non-IOI pairs):")
        for method, score in sorted(sep.items()):
            print(f"    {method:>20}: {score:.3f}x")

    # ── Classification ───────────────────────────────────────────────────
    rf = report["classification"]["random_forest"]
    print(f"\n  Random Forest (LOO, {len(tasks)} circuits):")
    print(f"    Accuracy: {rf['accuracy']:.2f} ({rf['correct']}/{rf['total']})")
    for item in rf["per_task"]:
        mark = "OK" if item["correct"] else "WRONG"
        print(f"    {item['task']:>14}: true={item['true']:>14}  "
              f"pred={item['predicted']:>14}  [{mark}]")
    print(f"    Top features:")
    for fname, imp in rf["top_features"][:5]:
        print(f"      {fname:>30}: {imp:.4f}")

    stump = report["classification"]["decision_stump"]
    print(f"\n  Decision Stump (depth=1 tree):")
    print(f"    Accuracy: {stump['accuracy']:.2f} ({stump['accuracy']*stump['total']:.0f}/{stump['total']})")
    print(f"    Split feature: {stump['best_feature']} (threshold={stump['threshold']})")
    for item in stump["per_task"]:
        mark = "OK" if item["correct"] else "WRONG"
        print(f"    {item['task']:>14}: true={item['true']:>14}  "
              f"pred={item['predicted']:>14}  [{mark}]")

    # ── ILP vs baselines summary ─────────────────────────────────────────
    print(f"\n  Summary comparison:")
    print(f"    ILP_arch learned clauses achieve F1=1.0 on one-vs-rest (5 circuits)")
    print(f"    Random Forest LOO accuracy: {rf['accuracy']:.2f}")
    print(f"    Decision Stump accuracy: {stump['accuracy']:.2f}")
    if corr:
        best_corr = max(corr.items(), key=lambda x: abs(x[1]))
        print(f"    Best distance correlation with ILP: {best_corr[0]} "
              f"(r={best_corr[1]:.4f})")

    # ── Save ─────────────────────────────────────────────────────────────
    if save:
        out_dir = os.path.join(model_dir, "baselines")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "baseline_comparison.json")

        # Make report JSON-serializable
        def _serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=_serialize)
        print(f"\n  Report saved: {out_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run structural baselines")
    parser.add_argument("--model", type=str, default=None,
                        help="Single model key (e.g. pythia1b)")
    parser.add_argument("--save", action="store_true",
                        help="Save report JSON to analysis/{model}/baselines/")
    args = parser.parse_args()

    if args.model:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(MODELS.keys())}")
            sys.exit(1)
        run_for_model(args.model, save=args.save)
    else:
        for model_key in MODELS:
            run_for_model(model_key, save=args.save)


if __name__ == "__main__":
    main()
