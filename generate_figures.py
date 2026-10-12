"""
generate_figures.py
===================
Generate all paper figures from stored results.

Produces:
  - 5×5 distance heatmap (main paper, original 5 tasks)
  - Full N×N distance heatmap (appendix, all tasks)
  - Baseline comparison bar chart
  - Within-task variance analysis (appendix)
  - δ_ILP sensitivity curve
  - ILP confidence comparison (5-task vs multi-task)

Usage:
    python generate_figures.py                         # all models
    python generate_figures.py --model pythia1b        # single model
    python generate_figures.py --model pythia1b --only distance
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ── Config ────────────────────────────────────────────────────────────────

ANALYSIS_ROOT = "./analysis"
FIGURES_DIR = f"{ANALYSIS_ROOT}/figures/paper"

MODELS = ["pythia14m", "pythia1b", "Llama3.21B"]

ORIGINAL_5 = ["ioi", "location", "path", "time", "greaterthan"]
ORIGINAL_5_LABELS = ["IOI", "Location", "Path", "Time", "GT"]

TASK_FAMILIES = {
    "ioi": "IOI", "location": "SemanticRoles", "time": "SemanticRoles",
    "path": "SemanticRoles", "greaterthan": "GreaterThan",
    "beneficiary": "SemanticRoles", "goal": "SemanticRoles",
    "instrument": "SemanticRoles", "source": "SemanticRoles",
    "topic": "SemanticRoles",
}

FAMILY_COLORS = {
    "IOI": "#e74c3c",
    "SemanticRoles": "#3498db",
    "GreaterThan": "#2ecc71",
}

# ── CFS / DLA figure constants ─────────────────────────────────────────────

TASK_COLORS = {
    "IOI":      "#c0392b",
    "Location": "#2980b9",
    "Time":     "#8e44ad",
    "Path":     "#16a085",
    "GT":       "#27ae60",
}

LINGUISTIC_ROLES_ORDER = ["subject", "object", "verb", "function", "entity", "other"]
LING_ROLE_COLORS = {
    "subject":  "#2196F3",   # blue
    "object":   "#43A047",   # green
    "verb":     "#E53935",   # red
    "function": "#8E24AA",   # purple
    "entity":   "#FB8C00",   # amber
    "other":    "#78909C",   # blue-grey
}

TASK_ROLE_PALETTE = [
    "#1976D2", "#388E3C", "#D32F2F", "#7B1FA2",
    "#F57C00", "#0097A7", "#5D4037", "#455A64",
]


# ── Helpers ───────────────────────────────────────────────────────────────

def load_distance_matrix(model, multi=False):
    suffix = "_multi" if multi else ""
    path = f"{ANALYSIS_ROOT}/{model}{suffix}/results/distance_report.json"
    if not os.path.isfile(path):
        return None, None
    with open(path) as f:
        dr = json.load(f)
    # The ILP signature distance is the architectural matrix; older reports
    # only carried it under the (duplicated) "semantic_matrix" key.
    return dr["tasks"], dr.get("architectural_matrix", dr.get("semantic_matrix"))


def load_baselines(model):
    path = f"{ANALYSIS_ROOT}/{model}/baselines/baseline_comparison.json"
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_sensitivity(model, multi=False):
    suffix = "_multi" if multi else ""
    path = f"{ANALYSIS_ROOT}/{model}{suffix}/baselines/delta_ilp_sensitivity.json"
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Figure 1: 5×5 Distance Heatmap (main paper) ──────────────────────────

def plot_distance_heatmap_5x5(model, out_dir):
    """5×5 heatmap for the original 5 tasks — main paper figure."""
    tasks, matrix = load_distance_matrix(model, multi=False)
    if not matrix:
        tasks, matrix = load_distance_matrix(model, multi=True)
    if not matrix:
        print(f"  [SKIP] No distance matrix for {model}")
        return

    # Filter to original 5
    avail = [t for t in ORIGINAL_5 if t in matrix]
    if len(avail) < 5:
        print(f"  [SKIP] Only {len(avail)} of 5 tasks available for {model}")
        return

    labels = [ORIGINAL_5_LABELS[ORIGINAL_5.index(t)] for t in avail]
    n = len(avail)
    D = np.zeros((n, n))
    for i, t1 in enumerate(avail):
        for j, t2 in enumerate(avail):
            D[i, j] = matrix[t1][t2]

    fig, ax = plt.subplots(figsize=(5, 4))
    cmap = LinearSegmentedColormap.from_list("ilp", ["#ffffff", "#3498db", "#1a252f"])
    im = ax.imshow(D, cmap=cmap, vmin=0, vmax=0.5)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_yticklabels(labels, fontsize=14)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i != j:
                color = "white" if D[i, j] > 0.25 else "black"
                ax.text(j, i, f"{D[i,j]:.2f}", ha="center", va="center",
                        fontsize=13, color=color)

    plt.colorbar(im, ax=ax, label="ILP Signature Distance", shrink=0.8)
    ax.set_title(f"ILP Signature Distance ({model})", fontsize=11, pad=10)
    plt.tight_layout()

    path = os.path.join(out_dir, f"rq2_distance_5x5_{model}.pdf")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {path}")


# ── Figure 2: Full N×N Distance Heatmap (appendix) ───────────────────────

def plot_distance_heatmap_full(model, out_dir):
    """Full N×N heatmap for all tasks — appendix figure."""
    tasks, matrix = load_distance_matrix(model, multi=True)
    if not matrix:
        print(f"  [SKIP] No multi distance matrix for {model}")
        return

    # Sort: IOI first, GT second, roles alphabetically
    def _sort_key(t):
        if t == "ioi": return (0, t)
        if t == "greaterthan": return (1, t)
        return (2, t)
    tasks_sorted = sorted(tasks, key=_sort_key)

    n = len(tasks_sorted)
    D = np.zeros((n, n))
    for i, t1 in enumerate(tasks_sorted):
        for j, t2 in enumerate(tasks_sorted):
            D[i, j] = matrix[t1][t2]

    _label_map = {
        "ioi": "IOI", "greaterthan": "GT", "beneficiary": "Beneficiary",
        "goal": "Goal", "instrument": "Instrument", "location": "Location",
        "path": "Path", "source": "Source", "time": "Time", "topic": "Topic",
        "manner": "Manner", "material": "Material",
    }
    labels = [_label_map.get(t, t.title()) for t in tasks_sorted]

    cell_size = 0.85
    fig_w = max(8, n * cell_size + 2.5)
    fig_h = max(7, n * cell_size + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = LinearSegmentedColormap.from_list("ilp", ["#ffffff", "#3498db", "#1a252f"])
    im = ax.imshow(D, cmap=cmap, vmin=0, vmax=max(0.35, D.max()))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=12, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=12)

    for i in range(n):
        for j in range(n):
            if i != j:
                color = "white" if D[i, j] > 0.15 else "black"
                ax.text(j, i, f"{D[i,j]:.2f}", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("ILP Signature Distance", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    ax.set_title(f"ILP Signature Distance — All Tasks ({model})", fontsize=14, pad=12)
    ax.tick_params(axis='both', which='major', pad=4)
    plt.tight_layout()

    path = os.path.join(out_dir, f"appendix_distance_full_{model}.pdf")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  {path}")


# ── Figure 3: Baseline Comparison ────────────────────────────────────────

def plot_baseline_comparison(model, out_dir):
    """Bar chart comparing separation scores across methods."""
    bl = load_baselines(model)
    if not bl:
        print(f"  [SKIP] No baselines for {model}")
        return

    sep = bl.get("separation_scores", {})
    if not sep:
        return

    methods = ["ilp", "wl", "feat_euclidean", "feat_cosine"]
    labels = ["ILP", "WL Kernel", "Feature (Euc)", "Feature (Cos)"]
    scores = [sep.get(m, 0) for m in methods]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(range(len(methods)), scores, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("IOI Separation Score", fontsize=10)
    ax.set_title(f"Structural Separation — {model}", fontsize=11)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="No separation")
    ax.legend(fontsize=8)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{score:.2f}x", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, f"rq2_baseline_comparison_{model}.pdf")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {path}")


# ── Figure 4: δ_ILP Sensitivity Curve ────────────────────────────────────

def plot_sensitivity(model, out_dir):
    """Sensitivity curve for δ_ILP threshold."""
    sweep = load_sensitivity(model, multi=True)
    if not sweep:
        sweep = load_sensitivity(model, multi=False)
    if not sweep:
        print(f"  [SKIP] No sensitivity data for {model}")
        return

    deltas = [r["delta"] for r in sweep["thresholds"]]
    f1s = [r["f1"] for r in sweep["thresholds"]]
    accs = [r["accuracy"] for r in sweep["thresholds"]]
    precs = [r["precision"] for r in sweep["thresholds"]]
    recalls = [r["recall"] for r in sweep["thresholds"]]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(deltas, f1s, "o-", color="#e74c3c", label="F1", linewidth=2)
    ax.plot(deltas, precs, "s--", color="#3498db", label="Precision", linewidth=1.5)
    ax.plot(deltas, recalls, "^--", color="#2ecc71", label="Recall", linewidth=1.5)
    ax.axvline(sweep["best_threshold"], color="gray", linestyle=":", linewidth=1,
               label=f"Best δ={sweep['best_threshold']}")
    ax.set_xlabel("δ_ILP threshold", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(f"δ_ILP Sensitivity — {model}", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 0.55)
    plt.tight_layout()

    path = os.path.join(out_dir, f"appendix_sensitivity_{model}.pdf")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {path}")


# ── Figure 5: Within-Task Variance (appendix) ────────────────────────────

def plot_within_task_variance(model, out_dir):
    """
    For each task with 3 splits, compute pairwise WL distance between
    splits of the SAME task vs splits of DIFFERENT tasks.

    Shows whether prompt subsets produce structurally different circuits.
    """
    from trace.mechint.baselines_structural import (
        wl_distance, _load_circuit_graph, extract_structural_features,
    )

    results_dir = f"{ANALYSIS_ROOT}/{model}_multi/results"
    if not os.path.isdir(results_dir):
        print(f"  [SKIP] No multi results for {model}")
        return

    # Load all split records
    import glob
    records = {}
    for fp in sorted(glob.glob(f"{results_dir}/*_s*_results.json")):
        key = os.path.basename(fp).replace("_results.json", "")
        with open(fp) as f:
            records[key] = json.load(f)

    # Group by task
    from collections import defaultdict
    task_splits = defaultdict(list)
    for key in records:
        task = key.rsplit("_s", 1)[0]
        if task in ("manner", "material"):
            continue
        task_splits[task].append(key)

    # Compute within-task and between-task WL distances
    within_dists = {}  # task -> list of pairwise WL distances among splits
    between_dists = {}  # task -> list of WL distances to other tasks' splits

    for task, keys in sorted(task_splits.items()):
        if len(keys) < 2:
            continue

        graphs = {k: _load_circuit_graph(records[k]) for k in keys}

        # Within-task: pairwise among splits
        w_dists = []
        for i, k1 in enumerate(keys):
            for k2 in keys[i+1:]:
                d = wl_distance(graphs[k1], graphs[k2])
                w_dists.append(d)
        within_dists[task] = w_dists

        # Between-task: distance to first split of each other task
        b_dists = []
        for other_task, other_keys in task_splits.items():
            if other_task == task:
                continue
            other_graph = _load_circuit_graph(records[other_keys[0]])
            d = wl_distance(graphs[keys[0]], other_graph)
            b_dists.append(d)
        between_dists[task] = b_dists

    if not within_dists:
        print(f"  [SKIP] No multi-split tasks for {model}")
        return

    # Plot
    tasks_sorted = sorted(within_dists.keys())
    n = len(tasks_sorted)

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(n)
    w_means = [np.mean(within_dists[t]) for t in tasks_sorted]
    w_stds = [np.std(within_dists[t]) if len(within_dists[t]) > 1 else 0 for t in tasks_sorted]
    b_means = [np.mean(between_dists[t]) for t in tasks_sorted]
    b_stds = [np.std(between_dists[t]) if len(between_dists[t]) > 1 else 0 for t in tasks_sorted]

    width = 0.35
    bars1 = ax.bar(x - width/2, w_means, width, yerr=w_stds,
                   label="Within-task (same role, diff prompts)",
                   color="#3498db", alpha=0.8, capsize=3)
    bars2 = ax.bar(x + width/2, b_means, width, yerr=b_stds,
                   label="Between-task (different roles)",
                   color="#e74c3c", alpha=0.8, capsize=3)

    ax.set_xticks(x)
    _var_label_map = {
        "ioi": "IOI", "greaterthan": "GT", "beneficiary": "Beneficiary",
        "goal": "Goal", "instrument": "Instrument", "location": "Location",
        "path": "Path", "source": "Source", "time": "Time", "topic": "Topic",
    }
    ax.set_xticklabels([_var_label_map.get(t, t.title()) for t in tasks_sorted],
                        fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("WL Kernel Distance", fontsize=10)
    ax.set_title(f"Within-Task vs Between-Task Structural Variance — {model}", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(max(b_means) + 0.1, 0.4))
    plt.tight_layout()

    path = os.path.join(out_dir, f"appendix_within_task_variance_{model}.pdf")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {path}")

    # Also save numeric report
    report = {
        "model": model,
        "tasks": tasks_sorted,
        "within_task_mean": {t: round(np.mean(within_dists[t]), 4) for t in tasks_sorted},
        "between_task_mean": {t: round(np.mean(between_dists[t]), 4) for t in tasks_sorted},
        "ratio": {t: round(np.mean(between_dists[t]) / max(np.mean(within_dists[t]), 0.001), 2)
                  for t in tasks_sorted},
    }
    report_path = os.path.join(out_dir, f"appendix_within_task_variance_{model}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"    {'Task':>12} {'Within':>8} {'Between':>8} {'Ratio':>6}")
    for t in tasks_sorted:
        w = np.mean(within_dists[t])
        b = np.mean(between_dists[t])
        r = b / max(w, 0.001)
        print(f"    {t:>12} {w:8.4f} {b:8.4f} {r:6.1f}x")


# ── Figure 6: ILP Confidence Comparison ──────────────────────────────────

def plot_confidence_comparison(out_dir):
    """Compare ILP F1 across models and task counts."""
    data = {}
    for model in MODELS:
        # 5-task confidence
        confs_5 = {}
        for task in ORIGINAL_5:
            path = f"{ANALYSIS_ROOT}/{model}/results/{task}_results.json"
            if os.path.isfile(path):
                with open(path) as f:
                    rec = json.load(f)
                c = rec.get("layer3", {}).get("tau_arch_confidence")
                if c is not None:
                    confs_5[task] = c

        # Multi-task confidence
        confs_multi = {}
        import glob
        for fp in glob.glob(f"{ANALYSIS_ROOT}/{model}_multi/results/*_s0_results.json"):
            with open(fp) as f:
                rec = json.load(f)
            task = rec.get("meta", {}).get("task", "").lower()
            if task in ("manner", "material"):
                continue
            c = rec.get("layer3", {}).get("tau_arch_confidence")
            if c is not None:
                confs_multi[task] = c

        if confs_5 or confs_multi:
            data[model] = {"5_task": confs_5, "multi_task": confs_multi}

    if not data:
        print("  [SKIP] No confidence data")
        return

    fig, axes = plt.subplots(1, len(data), figsize=(4 * len(data), 4), sharey=True)
    if len(data) == 1:
        axes = [axes]

    for ax, (model, d) in zip(axes, data.items()):
        tasks_5 = sorted(d["5_task"].keys())
        tasks_m = sorted(d["multi_task"].keys())
        all_tasks = sorted(set(tasks_5) | set(tasks_m))

        x = np.arange(len(all_tasks))
        vals_5 = [d["5_task"].get(t, 0) for t in all_tasks]
        vals_m = [d["multi_task"].get(t, 0) for t in all_tasks]

        width = 0.35
        ax.bar(x - width/2, vals_5, width, label="5-task (n=1 pos)",
               color="#2ecc71", alpha=0.8)
        ax.bar(x + width/2, vals_m, width, label="10-task (n=3 pos)",
               color="#e74c3c", alpha=0.8)

        ax.set_xticks(x)
        _conf_labels = {
            "ioi": "IOI", "greaterthan": "GT", "beneficiary": "Benef.",
            "goal": "Goal", "instrument": "Instr.", "location": "Locat.",
            "path": "Path", "source": "Source", "time": "Time", "topic": "Topic",
        }
        ax.set_xticklabels([_conf_labels.get(t, t[:6].title()) for t in all_tasks],
                            fontsize=7, rotation=45, ha="right")
        ax.set_title(model, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
        if ax == axes[0]:
            ax.set_ylabel("ILP F1 (confidence)", fontsize=9)
        ax.legend(fontsize=7)

    plt.suptitle("ILP Confidence: 5-task vs Multi-task Evaluation", fontsize=11, y=1.02)
    plt.tight_layout()

    path = os.path.join(out_dir, "rq2_confidence_comparison.pdf")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {path}")


# ── Figure: DLA Scatter (attn vs MLP) ────────────────────────────────────

_CFS_TASKS  = ["ioi", "location", "time", "path", "greaterthan"]
_CFS_LABELS = ["IOI", "Location", "Time", "Path", "GT"]


def _load_cfs_task_data(model, tasks, labels):
    """Load CFS component data for all tasks.

    Returns (attn_abs_mean, mlp_abs_mean, ling_profiles, task_profiles) dicts
    keyed by task label.  Skips tasks whose result file is missing.
    """
    base = f"{ANALYSIS_ROOT}/{model}/results"
    attn_abs_mean, mlp_abs_mean, ling_profiles, task_profiles = {}, {}, {}, {}

    for task, label in zip(tasks, labels):
        path = f"{base}/{task}_results.json"
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            d = json.load(f)
        comps = d.get("layer2", {}).get("cfs", {}).get("components", {})
        if not comps:
            continue

        # DLA magnitude means
        attn_abs = [abs(v["dla_score"]) for k, v in comps.items() if k.startswith("a")]
        mlp_abs  = [abs(v["dla_score"]) for k, v in comps.items() if k.startswith("m")]
        attn_abs_mean[label] = np.mean(attn_abs) if attn_abs else 0.0
        mlp_abs_mean[label]  = np.mean(mlp_abs)  if mlp_abs  else 0.0

        # Linguistic profiles — prefer causal heads, fall back to all attn heads
        ling_totals = {r: 0.0 for r in LINGUISTIC_ROLES_ORDER}
        ling_count = 0
        for marginal_ok in (False, True):
            for k, v in comps.items():
                if not k.startswith("a"):
                    continue
                if not marginal_ok and v.get("causally_marginal", True):
                    continue
                prof = v.get("attention_profile", {})
                if sum(prof.values()) > 0.01:
                    for r in LINGUISTIC_ROLES_ORDER:
                        ling_totals[r] += prof.get(r, 0.0)
                    ling_count += 1
            if ling_count > 0:
                break
        if ling_count > 0:
            for r in LINGUISTIC_ROLES_ORDER:
                ling_totals[r] /= ling_count
            ling_profiles[label] = ling_totals

        # Task-specific profiles
        task_totals = {}
        task_count = 0
        for k, v in comps.items():
            if k.startswith("a") and not v.get("causally_marginal", True):
                tp = v.get("task_attention_profile", {})
                if tp and sum(tp.values()) > 0.01:
                    for r, val in tp.items():
                        task_totals[r] = task_totals.get(r, 0.0) + val
                    task_count += 1
        if task_count > 0:
            for r in task_totals:
                task_totals[r] /= task_count
            task_profiles[label] = task_totals

    return attn_abs_mean, mlp_abs_mean, ling_profiles, task_profiles


def print_ling_profile_table(model):
    """Print dominant linguistic role per task from actual mean profiles (for table)."""
    _, _, ling_profiles, _ = _load_cfs_task_data(model, _CFS_TASKS, _CFS_LABELS)
    print(f"\n  Linguistic profile means ({model}):")
    print(f"  {'Task':>10}  {'Dom.Role':>10}  {'%':>6}  {'subj':>6} {'obj':>6} {'verb':>6} {'fn':>6} {'ent':>6} {'other':>6}")
    for label in _CFS_LABELS:
        if label not in ling_profiles:
            print(f"  {label:>10}  {'[no data]':>10}")
            continue
        p = ling_profiles[label]
        dom = max(p, key=p.get)
        pct = p[dom] * 100
        print(f"  {label:>10}  {dom:>10}  {pct:5.1f}%"
              f"  {p.get('subject',0):6.3f} {p.get('object',0):6.3f}"
              f" {p.get('verb',0):6.3f} {p.get('function',0):6.3f}"
              f" {p.get('entity',0):6.3f} {p.get('other',0):6.3f}")


def plot_dla_scatter(model, out_dir):
    """Scatter: mean |δ_v| for attention heads vs MLP layers per task."""
    attn_abs_mean, mlp_abs_mean, _, _ = _load_cfs_task_data(
        model, _CFS_TASKS, _CFS_LABELS)
    available = [l for l in _CFS_LABELS if l in attn_abs_mean]
    if not available:
        print(f"  [SKIP] No CFS data for {model}")
        return

    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    max_val = max(
        max(attn_abs_mean[l] for l in available),
        max(mlp_abs_mean[l] for l in available),
    ) * 1.15
    ax.plot([0, max_val], [0, max_val], color="#cccccc", linewidth=1.0,
            linestyle="--", zorder=1)
    ax.text(max_val * 0.62, max_val * 0.72, "attn = MLP",
            color="#aaaaaa", fontsize=8, rotation=42, ha="center")

    offsets = {"IOI": (0.04, -0.10), "GT": (0.04, 0.05),
               "Location": (0.04, 0.04), "Time": (-0.28, 0.04), "Path": (0.04, -0.10)}
    for label in available:
        x, y = attn_abs_mean[label], mlp_abs_mean[label]
        c = TASK_COLORS[label]
        ax.scatter(x, y, s=80, color=c, zorder=4, edgecolors="white", linewidths=0.8)
        dx, dy = offsets.get(label, (0.04, 0.04))
        ax.text(x + dx, y + dy, label, fontsize=10, color=c, fontweight="bold")

    ax.set_xlabel("Mean $|\\delta_v|$ — attention heads", fontsize=10)
    ax.set_ylabel("Mean $|\\delta_v|$ — MLP layers", fontsize=10)
    ax.set_xlim(-0.1, max_val)
    ax.set_ylim(-0.1, max_val)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.18, linewidth=0.7)
    ax.set_title(f"DLA magnitude per component type ({model})",
                 fontsize=11, fontweight="bold", pad=6)

    plt.tight_layout()
    path = os.path.join(out_dir, f"fig_dla_scatter_{model}.pdf")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  {path}")


# ── Figure: Linguistic Attention Role Profile ─────────────────────────────

def plot_linguistic_profile(model, out_dir):
    """Horizontal stacked bars: all 6 linguistic roles per task."""
    _, _, ling_profiles, _ = _load_cfs_task_data(model, _CFS_TASKS, _CFS_LABELS)
    available = [l for l in _CFS_LABELS if l in ling_profiles]
    if not available:
        print(f"  [SKIP] No linguistic profile data for {model}")
        return

    fig, ax = plt.subplots(figsize=(7, max(2.5, len(available) * 0.7)))

    y_pos = list(range(len(available)))
    lefts = np.zeros(len(available))

    for role in LINGUISTIC_ROLES_ORDER:
        vals = np.array([ling_profiles[l].get(role, 0.0) for l in available])
        if not any(v > 0.005 for v in vals):
            continue
        ax.barh(y_pos, vals, left=lefts, color=LING_ROLE_COLORS[role],
                label=role.capitalize(), height=0.55, alpha=0.88)
        lefts += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(available, fontsize=11)
    for yi, label in enumerate(available):
        ax.get_yticklabels()[yi].set_color(TASK_COLORS.get(label, "#333333"))
        ax.get_yticklabels()[yi].set_fontweight("bold")

    ax.set_xlabel("Attention weight (mean over causal heads)", fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_title(f"Linguistic attention role profile ({model})",
                 fontsize=11, fontweight="bold", pad=6)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, f"fig_linguistic_profile_{model}.pdf")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  {path}")


# ── Figure: Task-Specific Attention Role Profile ──────────────────────────

def plot_taskspecific_profile(model, out_dir):
    """Subplots: task-specific role composition, one subplot per task."""
    _, _, _, task_profiles = _load_cfs_task_data(model, _CFS_TASKS, _CFS_LABELS)
    available = [(l, task_profiles[l]) for l in _CFS_LABELS if l in task_profiles]
    if not available:
        print(f"  [SKIP] No task-specific profile data for {model}")
        return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(7, max(3, 1.8 * n)))
    if n == 1:
        axes = [axes]

    for ax, (label, profile) in zip(axes, available):
        # Sort by contribution descending; "other" always last
        content_roles = sorted(
            [r for r in profile if r != "other"],
            key=lambda r: -profile[r],
        )
        roles_sorted = content_roles + (["other"] if "other" in profile else [])

        left = 0.0
        for i, r in enumerate(roles_sorted):
            v = profile.get(r, 0.0)
            if v <= 0.005:
                continue
            color = "#78909C" if r == "other" else TASK_ROLE_PALETTE[i % len(TASK_ROLE_PALETTE)]
            ax.barh([0], [v], left=left, color=color,
                    label=r.replace("_", " "), height=0.5, alpha=0.88)
            left += v

        ax.set_yticks([0])
        ax.set_yticklabels([label], fontsize=10,
                           color=TASK_COLORS.get(label, "#333333"),
                           fontweight="bold")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Attention weight" if ax is axes[-1] else "", fontsize=9)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.6, ncol=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)

    fig.suptitle(f"Task-specific attention role profile ({model})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"fig_taskspecific_profile_{model}.pdf")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--model", default=None)
    parser.add_argument("--only", default=None,
                        help="Only generate specific figure type")
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    models = [args.model] if args.model else MODELS

    print(f"Generating figures -> {FIGURES_DIR}/")
    print(f"Models: {models}")

    for model in models:
        print(f"\n{'='*60}")
        print(f"  {model}")
        print(f"{'='*60}")

        if not args.only or args.only == "distance":
            plot_distance_heatmap_5x5(model, FIGURES_DIR)
            plot_distance_heatmap_full(model, FIGURES_DIR)

        if not args.only or args.only == "baseline":
            plot_baseline_comparison(model, FIGURES_DIR)

        if not args.only or args.only == "sensitivity":
            plot_sensitivity(model, FIGURES_DIR)

        if not args.only or args.only == "variance":
            plot_within_task_variance(model, FIGURES_DIR)

        if not args.only or args.only == "dla":
            print_ling_profile_table(model)
            plot_dla_scatter(model, FIGURES_DIR)
            plot_linguistic_profile(model, FIGURES_DIR)
            plot_taskspecific_profile(model, FIGURES_DIR)

    # Cross-model figures
    if not args.only or args.only == "confidence":
        print(f"\n{'='*60}")
        print("  Cross-model")
        print(f"{'='*60}")
        plot_confidence_comparison(FIGURES_DIR)

    print(f"\nDone. Figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
