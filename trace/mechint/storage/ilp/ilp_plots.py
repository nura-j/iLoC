"""
ilp_plots.py
============
Publication-quality figures for the ILP pipeline results.

Style: clean serif, warm neutral palette — readable in both paper and slides.
All functions return the figure object; call fig.savefig(...) to export.

Usage:
    from ilp_plots import (
        plot_distance_matrices,
        plot_transfer_candidates,
        plot_circuit_fingerprints,
        plot_motif_overlap,
        save_all,
    )
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np


# ── Palette ───────────────────────────────────────────────────────────────────
# Warm, restrained — readable at journal scale, distinctive at poster scale

_C = {
    "coral":   "#D4693A",   # orange-red — high attention
    "slate":   "#3D4852",   # dark neutral text / axes
    "warm_bg": "#F7F5F2",   # warm off-white background
    "mid":     "#C8C4BE",   # mid neutral (gridlines, spines)
    "sand":    "#8B6F5C",   # muted warm brown (secondary)
    "teal":    "#3A7D7B",   # cool accent for second series
    "cream":   "#EDE8E2",   # very light warm (cell backgrounds)
}

_TASK_COLORS = {
    "location": "#D4693A",
    "time":     "#3A7D7B",
    "path":     "#8B6F5C",
    "ioi":      "#3D4852",
    "unknown":  "#C8C4BE",
}


def _base_rcparams():
    """Apply consistent rcParams for all figures."""
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Georgia", "DejaVu Serif", "Times New Roman"],
        "font.size":         10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  True,
        "axes.spines.bottom":True,
        "axes.linewidth":    0.8,
        "axes.edgecolor":    _C["mid"],
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "text.color":        _C["slate"],
        "axes.labelcolor":   _C["slate"],
        "xtick.color":       _C["slate"],
        "ytick.color":       _C["slate"],
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.frameon":    False,
        "legend.fontsize":   9,
    })


def _cmap(hi_color: str) -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "custom", ["white", hi_color], N=256
    )


# ── 1. Distance Matrix Heatmaps ───────────────────────────────────────────────

def plot_distance_matrices(
    sem_matrix:  Dict[str, Dict[str, float]],
    arch_matrix: Dict[str, Dict[str, float]],
    tasks:       Optional[List[str]] = None,
    model_label: str = "",
) -> plt.Figure:
    """
    Side-by-side heatmaps of semantic and architectural ILP signature distances.

    Parameters
    ----------
    sem_matrix  : {task_a: {task_b: float}} — semantic distances
    arch_matrix : {task_a: {task_b: float}} — architectural distances
    tasks       : task order (defaults to sorted keys of sem_matrix)
    model_label : e.g. "Pythia-14M" — used in suptitle
    """
    _base_rcparams()

    tasks = tasks or sorted(sem_matrix.keys())
    n     = len(tasks)
    labels = [t.upper() if t == "ioi" else t.capitalize() for t in tasks]

    def _to_array(mat):
        return np.array([[mat.get(a, {}).get(b, 0.0) for b in tasks]
                         for a in tasks])

    sem_arr  = _to_array(sem_matrix)
    arch_arr = _to_array(arch_matrix)
    vmax     = max(sem_arr.max(), arch_arr.max(), 0.1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("white")

    configs = [
        (axes[0], sem_arr,  "Semantic distance  $d_{sem}$",  _C["coral"]),
        (axes[1], arch_arr, "Architectural distance  $d_{arch}$", _C["slate"]),
    ]

    for ax, arr, title, hi in configs:
        cmap = _cmap(hi)
        im   = ax.imshow(arr, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")

        # Cell annotations
        for i in range(n):
            for j in range(n):
                v = arr[i, j]
                text_color = "white" if v > vmax * 0.55 else _C["slate"]
                weight     = "bold"  if i == j         else "normal"
                ax.text(j, i, f"{v:.3f}",
                        ha="center", va="center",
                        fontsize=9, color=text_color, fontweight=weight)

        # Axes
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11, pad=10,
                     color=_C["slate"], fontweight="bold")
        ax.tick_params(length=0)

        # Thin cell separators
        for k in range(-1, n):
            ax.axhline(k + 0.5, color=_C["cream"], lw=0.7, zorder=0)
            ax.axvline(k + 0.5, color=_C["cream"], lw=0.7, zorder=0)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)

        # Colourbar
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8, colors=_C["slate"])
        cb.outline.set_edgecolor(_C["mid"])

    title_parts = ["ILP Signature Distance Matrices"]
    if model_label:
        title_parts.append(f"({model_label})")
    fig.suptitle("  ".join(title_parts),
                 fontsize=12, y=1.02, color=_C["slate"], fontweight="bold")
    plt.tight_layout()
    return fig


# ── 2. Transfer Candidate Rankings ───────────────────────────────────────────

def plot_transfer_candidates(
    transfer_reports: Dict[str, Any],
    source_tasks:     Optional[List[str]] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of fingerprint similarity scores for each source task.
    Shows which candidate tasks score highest as transfer targets.
    Bars are coloured by task; struct_pass candidates are solid, others faded.
    """
    _base_rcparams()

    source_tasks = source_tasks or list(transfer_reports.keys())
    n_sources    = len(source_tasks)
    if not n_sources:
        return plt.figure()

    fig, axes = plt.subplots(
        1, n_sources,
        figsize=(3.5 * n_sources, 3.5),
        sharey=False,
    )
    if n_sources == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, src_task in zip(axes, source_tasks):
        report = transfer_reports.get(src_task)
        if report is None:
            ax.set_visible(False)
            continue

        candidates = report.candidates
        if not candidates:
            ax.text(0.5, 0.5, "no candidates", ha="center", va="center",
                    transform=ax.transAxes, color=_C["mid"])
            continue

        # Sort by fingerprint score desc (fall back to arch_score)
        candidates = sorted(
            candidates,
            key=lambda c: -(c.fingerprint_score if c.fingerprint_score is not None
                            else c.arch_score),
        )

        names  = [c.task_name or c.circuit_uri or "?" for c in candidates]
        scores = [c.fingerprint_score if c.fingerprint_score is not None
                  else c.arch_score for c in candidates]
        passed = [c.struct_pass for c in candidates]

        y = np.arange(len(names))
        bar_colors = [
            _TASK_COLORS.get(n.lower().split("/")[-1], _C["sand"])
            for n in names
        ]
        alphas = [1.0 if p else 0.35 for p in passed]

        for i, (yi, sc, col, al) in enumerate(zip(y, scores, bar_colors, alphas)):
            ax.barh(yi, sc, color=col, alpha=al, height=0.6)
            ax.text(sc + 0.01, yi, f"{sc:.2f}",
                    va="center", fontsize=8, color=_C["slate"])

        ax.set_xlim(0, 1.05)
        ax.set_yticks(y)
        _labels = [n.split("/")[-1].capitalize() for n in names]
        ax.set_yticklabels(_labels, fontsize=9)
        ax.set_xlabel("Fingerprint similarity", fontsize=9)
        ax.set_title(f"Source: {src_task.capitalize()}",
                     fontsize=10, fontweight="bold", color=_C["slate"])
        ax.axvline(0.7, color=_C["mid"], lw=0.8, ls="--", zorder=0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)

    fig.suptitle("Transfer Candidate Ranking  (circuit fingerprint similarity)",
                 fontsize=11, y=1.02, color=_C["slate"], fontweight="bold")
    plt.tight_layout()
    return fig


# ── 3. Circuit Fingerprint Radar / Envelope ───────────────────────────────────

def plot_circuit_fingerprints(
    fingerprints: Dict[str, Dict[str, float]],
) -> plt.Figure:
    """
    Overlapping layer-depth envelopes for each circuit, plus component
    composition bars.  Two-panel layout.

    Parameters
    ----------
    fingerprints : {task: fingerprint_dict} from _circuit_fingerprint()
    """
    _base_rcparams()
    if not fingerprints:
        return plt.figure()

    tasks = list(fingerprints.keys())
    fig, (ax_env, ax_comp) = plt.subplots(1, 2, figsize=(11, 4),
                                          gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("white")

    # ── Left: depth envelope (horizontal band per circuit) ────────────────────
    for i, task in enumerate(tasks):
        fp    = fingerprints[task]
        lo    = fp.get("depth_min",  0.0)
        hi    = fp.get("depth_max",  1.0)
        mean  = fp.get("depth_mean", (lo + hi) / 2)
        color = _TASK_COLORS.get(task, _C["sand"])

        ax_env.barh(i, hi - lo, left=lo, height=0.5,
                    color=color, alpha=0.35, zorder=2)
        ax_env.barh(i, 0.01, left=mean, height=0.5,
                    color=color, alpha=0.9, zorder=3)  # mean tick
        ax_env.text(hi + 0.02, i, f"{lo:.2f}–{hi:.2f}",
                    va="center", fontsize=8, color=_C["slate"])

    ax_env.set_xlim(0, 1.15)
    ax_env.set_ylim(-0.5, len(tasks) - 0.5)
    ax_env.set_yticks(range(len(tasks)))
    ax_env.set_yticklabels(
        [t.upper() if t == "ioi" else t.capitalize() for t in tasks],
        fontsize=9)
    ax_env.set_xlabel("Relative layer position  [0 = first, 1 = last]", fontsize=9)
    ax_env.set_title("Layer depth envelope", fontsize=10,
                     fontweight="bold", color=_C["slate"])
    ax_env.axvline(0.0, color=_C["mid"], lw=0.5)
    ax_env.axvline(1.0, color=_C["mid"], lw=0.5)
    ax_env.spines["left"].set_visible(False)
    ax_env.tick_params(left=False)

    # ── Right: attention ratio bars ───────────────────────────────────────────
    attn_ratios = [fingerprints[t].get("attn_ratio", 0.8) for t in tasks]
    rel_sizes   = [fingerprints[t].get("rel_size",   0.1) for t in tasks]
    x = np.arange(len(tasks))
    colors = [_TASK_COLORS.get(t, _C["sand"]) for t in tasks]

    bars = ax_comp.bar(x, attn_ratios, color=colors, alpha=0.85, width=0.5,
                       label="Attn ratio")
    ax_comp.set_ylim(0, 1.15)
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(
        [t.upper() if t == "ioi" else t.capitalize() for t in tasks],
        fontsize=9)
    ax_comp.set_ylabel("Attention head fraction", fontsize=9)
    ax_comp.set_title("Component composition", fontsize=10,
                       fontweight="bold", color=_C["slate"])

    # Overlay rel_size as dots on a twin axis
    ax2 = ax_comp.twinx()
    ax2.scatter(x, rel_sizes, color=colors, s=60, zorder=5, marker="D",
                edgecolors=_C["slate"], linewidths=0.6)
    ax2.set_ylim(0, max(rel_sizes) * 3 or 0.5)
    ax2.set_ylabel("Relative size  (circuit / model)", fontsize=8,
                   color=_C["sand"])
    ax2.tick_params(axis="y", colors=_C["sand"], labelsize=8)
    ax2.spines["right"].set_edgecolor(_C["mid"])

    # Legend
    dot_patch = mpatches.Patch(color=_C["sand"], alpha=0.7, label="rel size ◆")
    bar_patch  = mpatches.Patch(color=_C["slate"], alpha=0.4, label="attn ratio")
    ax_comp.legend(handles=[bar_patch, dot_patch], loc="upper right", fontsize=8)

    fig.suptitle("Circuit Structural Fingerprints",
                 fontsize=12, y=1.02, color=_C["slate"], fontweight="bold")
    plt.tight_layout()
    return fig


# ── 4. Motif Overlap Matrix ───────────────────────────────────────────────────

def plot_motif_overlap(
    motif_sets: Dict[str, List[str]],
) -> plt.Figure:
    """
    Heatmap of pairwise Jaccard motif overlap between circuits.
    Rows/columns are tasks; annotations show shared motif count.
    """
    _base_rcparams()
    if not motif_sets:
        return plt.figure()

    tasks   = list(motif_sets.keys())
    n       = len(tasks)
    labels  = [t.upper() if t == "ioi" else t.capitalize() for t in tasks]
    sets    = [set(motif_sets[t]) for t in tasks]

    jaccard_mat = np.zeros((n, n))
    shared_mat  = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            jaccard_mat[i, j] = inter / union if union else 1.0
            shared_mat[i, j]  = inter

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("white")

    cmap = _cmap(_C["teal"])
    im   = ax.imshow(jaccard_mat, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    for i in range(n):
        for j in range(n):
            jac   = jaccard_mat[i, j]
            sh    = shared_mat[i, j]
            tc    = "white" if jac > 0.55 else _C["slate"]
            label = f"{jac:.2f}\n({sh})"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=8, color=tc,
                    fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Motif Jaccard overlap\n(Jaccard score / shared motif count)",
                 fontsize=10, fontweight="bold", color=_C["slate"])
    ax.tick_params(length=0)

    for k in range(-1, n):
        ax.axhline(k + 0.5, color=_C["cream"], lw=0.7, zorder=0)
        ax.axvline(k + 0.5, color=_C["cream"], lw=0.7, zorder=0)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8, colors=_C["slate"])
    cb.set_label("Jaccard similarity", fontsize=8, color=_C["slate"])
    cb.outline.set_edgecolor(_C["mid"])

    plt.tight_layout()
    return fig


# ── 5. Cross-Scale Transfer Summary ──────────────────────────────────────────

def plot_cross_scale_transfer(
    cross_scale_reports: List[tuple],   # [("loc: 14m→1b", report), ...]
) -> plt.Figure:
    """
    Summary bar chart of cross-scale transfer fingerprint scores.
    One bar per (source_task, source_model → target_model) pair.
    """
    _base_rcparams()
    if not cross_scale_reports:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No cross-scale transfer data\n(requires ≥2 models in registry)",
                ha="center", va="center", transform=ax.transAxes,
                color=_C["mid"], fontsize=10)
        ax.set_axis_off()
        fig.suptitle("Cross-Scale Transfer", fontsize=11,
                     color=_C["slate"], fontweight="bold")
        return fig

    labels, scores, passed_flags = [], [], []
    for key, report in cross_scale_reports:
        best = max(
            (c for c in report.candidates),
            key=lambda c: c.fingerprint_score or c.arch_score,
            default=None,
        )
        if best is None:
            continue
        labels.append(key)
        scores.append(best.fingerprint_score if best.fingerprint_score is not None
                      else best.arch_score)
        passed_flags.append(best.struct_pass)

    if not labels:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No candidates found", ha="center", va="center",
                transform=ax.transAxes, color=_C["mid"], fontsize=10)
        fig.suptitle("Cross-Scale Transfer", fontsize=11,
                     color=_C["slate"], fontweight="bold")
        return fig

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.8), 4))
    fig.patch.set_facecolor("white")
    x = np.arange(len(labels))
    colors = [_C["coral"] if p else _C["mid"] for p in passed_flags]

    ax.bar(x, scores, color=colors, alpha=0.85, width=0.6)
    ax.axhline(0.7, color=_C["slate"], lw=0.8, ls="--", zorder=0,
               label="similarity threshold 0.7")

    for xi, sc, p in zip(x, scores, passed_flags):
        mark = "✓" if p else "✗"
        ax.text(xi, sc + 0.02, f"{mark} {sc:.2f}",
                ha="center", fontsize=8, color=_C["slate"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Best fingerprint similarity", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.set_title("Cross-Scale Transfer  (best candidate per source)",
                 fontsize=10, fontweight="bold", color=_C["slate"])

    plt.tight_layout()
    return fig


# ── Convenience: save all ─────────────────────────────────────────────────────

def save_all(
    output_dir:           str,
    sem_matrix:           Optional[Dict]  = None,
    arch_matrix:          Optional[Dict]  = None,
    tasks:                Optional[List]  = None,
    model_label:          str             = "",
    transfer_reports:     Optional[Dict]  = None,
    fingerprints:         Optional[Dict]  = None,
    motif_sets:           Optional[Dict]  = None,
    cross_scale_reports:  Optional[List]  = None,
) -> List[str]:
    """
    Render and save all available figures to output_dir.
    Returns list of saved paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    def _save(fig: plt.Figure, name: str) -> str:
        path = os.path.join(output_dir, name)
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    if sem_matrix and arch_matrix:
        saved.append(_save(
            plot_distance_matrices(sem_matrix, arch_matrix, tasks, model_label),
            "distance_matrices.pdf",
        ))

    if transfer_reports:
        saved.append(_save(
            plot_transfer_candidates(transfer_reports),
            "transfer_candidates.pdf",
        ))

    if fingerprints:
        saved.append(_save(
            plot_circuit_fingerprints(fingerprints),
            "circuit_fingerprints.pdf",
        ))

    if motif_sets:
        saved.append(_save(
            plot_motif_overlap(motif_sets),
            "motif_overlap.pdf",
        ))

    if cross_scale_reports is not None:
        saved.append(_save(
            plot_cross_scale_transfer(cross_scale_reports),
            "cross_scale_transfer.pdf",
        ))

    return saved
