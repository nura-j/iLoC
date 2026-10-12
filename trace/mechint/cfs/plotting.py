"""
CFS visualisation — DLA scores, attention profiles, and combined views.

Produces separate figures:
    layer2_{role}_dla.png           — DLA bar chart per component
    layer2_{role}_attention.png     — attention profile heatmap per component
    layer2_{role}_cfs_summary.png   — combined CFS overview
    layer2_dla_comparison.png       — DLA across circuits

Usage:
    from trace.mechint.cfs.plotting import plot_dla, plot_attention_profiles, plot_cfs_summary
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .cfs import CFSResult
from .dla import LogitLensResult
from .attention_profile import TOKEN_ROLES

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "positive":  "#10B981",   # green
    "negative":  "#EF4444",   # red
    "marginal":  "#CBD5E1",   # slate
    "bg":        "#F8FAFC",
}

ROLE_COLORS = {
    "subject": "#3B82F6",
    "object":  "#F59E0B",
    "verb":    "#8B5CF6",
    "entity":  "#10B981",
    "other":   "#94A3B8",
}

TASK_COLORS = {
    "location": "#3B82F6",
    "time":     "#10B981",
    "path":     "#8B5CF6",
    "ioi":      "#F97316",
}


# ===========================================================================
# Plot 1: DLA scores per component
# ===========================================================================

def plot_dla(
    cfs: CFSResult,
    role: str,
    out_path: str,
    title: Optional[str] = None,
):
    """Bar chart of DLA scores per component."""
    if not cfs.components:
        return

    # Sort by |δ_v| descending
    sorted_comps = sorted(
        cfs.components.values(),
        key=lambda c: abs(c.dla_score),
        reverse=True,
    )
    names = [c.name for c in sorted_comps]
    scores = [c.dla_score for c in sorted_comps]
    colors = []
    for c in sorted_comps:
        if c.causally_marginal:
            colors.append(COLORS["marginal"])
        elif c.dla_score >= 0:
            colors.append(COLORS["positive"])
        else:
            colors.append(COLORS["negative"])

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    x = np.arange(len(names))
    bars = ax.bar(x, scores, color=colors, edgecolor="#1E293B", linewidth=0.8)

    # Threshold lines
    ax.axhline(y=cfs.epsilon_dla, color="#EF4444", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=-cfs.epsilon_dla, color="#EF4444", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=0, color="#64748B", linewidth=0.5)

    # Labels
    for bar, score in zip(bars, scores):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                y + (0.002 if y >= 0 else -0.004),
                f"{score:.4f}", ha="center", va="bottom" if y >= 0 else "top",
                fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("DLA Score (δ_v)")
    ax.set_title(
        title or f"Layer 2 — DLA Scores: {role.upper()} (pythia-14m)\n"
                 f"ε_DLA={cfs.epsilon_dla}, n={cfs.num_examples} examples",
        fontsize=13, fontweight="bold",
    )

    # Legend
    legend = [
        mpatches.Patch(color=COLORS["positive"], label="Positive DLA"),
        mpatches.Patch(color=COLORS["negative"], label="Negative DLA"),
        mpatches.Patch(color=COLORS["marginal"], label=f"Marginal (|δ| < {cfs.epsilon_dla})"),
    ]
    ax.legend(handles=legend, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Plot 2: Attention profiles per component
# ===========================================================================

def plot_attention_profiles(
    cfs: CFSResult,
    role: str,
    out_path: str,
    title: Optional[str] = None,
):
    """Heatmap of attention profiles: components x roles."""
    # Only non-marginal components
    causal = cfs.causal_components()
    if not causal:
        return

    names = [c.name for c in causal]
    roles = list(TOKEN_ROLES)
    data = np.array([
        [c.attention_profile.get(r, 0.0) for r in roles]
        for c in causal
    ])

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.6)), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels(roles, fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)

    # Annotate cells
    for i in range(len(names)):
        for j in range(len(roles)):
            val = data[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="π_v(role)")
    ax.set_title(
        title or f"Layer 2 — Attention Profiles: {role.upper()} (pythia-14m)\n"
                 f"Attribution-weighted attention over token roles",
        fontsize=13, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Plot 3: CFS summary (DLA + dominant role per component)
# ===========================================================================

def plot_cfs_summary(
    cfs: CFSResult,
    role: str,
    out_path: str,
    title: Optional[str] = None,
):
    """Combined view: DLA bars coloured by dominant attention role."""
    if not cfs.components:
        return

    sorted_comps = sorted(
        cfs.components.values(),
        key=lambda c: abs(c.dla_score),
        reverse=True,
    )
    names = [c.name for c in sorted_comps]
    scores = [c.dla_score for c in sorted_comps]
    colors = []
    for c in sorted_comps:
        if c.causally_marginal:
            colors.append(COLORS["marginal"])
        else:
            dom = c.dominant_role() or "other"
            colors.append(ROLE_COLORS.get(dom, ROLE_COLORS["other"]))

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    x = np.arange(len(names))
    ax.bar(x, scores, color=colors, edgecolor="#1E293B", linewidth=0.8)
    ax.axhline(y=0, color="#64748B", linewidth=0.5)
    ax.axhline(y=cfs.epsilon_dla, color="#EF4444", linestyle="--", linewidth=1, alpha=0.4)
    ax.axhline(y=-cfs.epsilon_dla, color="#EF4444", linestyle="--", linewidth=1, alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("DLA Score (δ_v)")
    ax.set_title(
        title or f"Layer 2 — CFS Summary: {role.upper()} (pythia-14m)\n"
                 f"Bar colour = dominant token role",
        fontsize=13, fontweight="bold",
    )

    # Legend for roles
    legend = [mpatches.Patch(color=c, label=r) for r, c in ROLE_COLORS.items()]
    legend.append(mpatches.Patch(color=COLORS["marginal"], label="marginal"))
    ax.legend(handles=legend, fontsize=8, loc="upper right", ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Plot 4: DLA comparison across circuits
# ===========================================================================

def plot_dla_comparison(
    cfs_results: Dict[str, CFSResult],
    out_path: str,
):
    """Compare DLA score distributions across circuits."""
    if not cfs_results:
        return

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    names = list(cfs_results.keys())
    positions = np.arange(len(names))

    for i, (role, cfs) in enumerate(cfs_results.items()):
        scores = sorted(
            [abs(c.dla_score) for c in cfs.components.values()],
            reverse=True,
        )
        color = TASK_COLORS.get(role.lower(), "#94A3B8")
        x = [i] * len(scores)
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(scores))
        ax.scatter(np.array(x) + jitter, scores, color=color, s=80,
                   edgecolors="#1E293B", linewidths=0.8, alpha=0.8, zorder=5)
        if scores:
            ax.bar(i, np.mean(scores), width=0.4, color=color, alpha=0.2)

    ax.axhline(y=0.05, color="#EF4444", linestyle="--", linewidth=1,
               alpha=0.5, label="ε_DLA=0.05")
    ax.set_xticks(positions)
    ax.set_xticklabels([r.upper() for r in names], fontsize=11)
    ax.set_ylabel("|δ_v| (absolute DLA)")
    ax.set_title("Layer 2 — DLA Comparison Across Circuits (pythia-14m)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Plot 5: Logit Lens — per-layer logit difference
# ===========================================================================

def plot_logit_lens(
    ll: LogitLensResult,
    role: str,
    out_path: str,
    title: Optional[str] = None,
):
    """Logit lens curve + DLA breakdown (attn vs MLP per layer)."""
    if not ll.per_layer:
        return

    layers = np.arange(ll.n_layers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=COLORS["bg"])

    # --- Left: Logit lens curve ---
    ax = axes[0]
    ax.set_facecolor(COLORS["bg"])
    color = TASK_COLORS.get(role.lower(), "#3B82F6")

    ax.plot(layers, ll.per_layer, color=color, lw=2.5, label=role.upper())

    if ll.curves:
        arr = np.array(ll.curves)
        std = np.std(arr, axis=0) / np.sqrt(len(arr))
        mean = np.array(ll.per_layer)
        ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.15)

    ax.axhline(0, color="black", ls="--", lw=1, alpha=0.4)

    cl = ll.crossing_layer()
    if cl is not None:
        ax.axvline(cl, color=color, ls=":", lw=1.5, alpha=0.7)
        ax.text(cl + 0.3, ax.get_ylim()[0] * 0.85, f"L{cl}", color=color, fontsize=9)

    ax.set_xlabel("Layer")
    ax.set_ylabel("logit(target) − mean(logit(foils))")
    ax.set_title(f"Logit Lens: {role.upper()} (n={ll.num_examples})")
    ax.grid(True, alpha=0.3)

    # --- Right: DLA breakdown (attn vs MLP per layer) ---
    ax = axes[1]
    ax.set_facecolor(COLORS["bg"])

    if ll.per_layer_dla:
        attn_vals = [d.get("attn", 0) for d in ll.per_layer_dla]
        mlp_vals = [d.get("mlp", 0) for d in ll.per_layer_dla]
        bar_w = 0.4

        ax.bar(layers - bar_w / 2, [max(v, 0) for v in attn_vals], bar_w,
               color="#6366F1", alpha=0.85, label="Attn (+)")
        ax.bar(layers - bar_w / 2, [min(v, 0) for v in attn_vals], bar_w,
               color="#6366F1", alpha=0.35)
        ax.bar(layers + bar_w / 2, [max(v, 0) for v in mlp_vals], bar_w,
               color="#F59E0B", alpha=0.85, label="MLP (+)")
        ax.bar(layers + bar_w / 2, [min(v, 0) for v in mlp_vals], bar_w,
               color="#F59E0B", alpha=0.35)

    ax.axhline(0, color="black", ls="--", lw=1, alpha=0.4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("DLA contribution")
    ax.set_title(f"DLA Breakdown: {role.upper()}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        title or f"Layer 2 — Logit Lens & DLA: {role.upper()} (pythia-14m)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Plot 6: Logit Lens overlay — all circuits on one plot
# ===========================================================================

def plot_logit_lens_overlay(
    ll_results: Dict[str, LogitLensResult],
    out_path: str,
):
    """Overlay logit lens curves for all circuits."""
    if not ll_results:
        return

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    for role, ll in ll_results.items():
        if not ll.per_layer:
            continue
        layers = np.arange(ll.n_layers)
        color = TASK_COLORS.get(role.lower(), "#94A3B8")
        ax.plot(layers, ll.per_layer, lw=2.5, color=color, label=role.upper())

        cl = ll.crossing_layer()
        if cl is not None:
            ax.axvline(cl, color=color, ls=":", lw=1.2, alpha=0.5)

    ax.axhline(0, color="black", ls="--", lw=1, alpha=0.4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("logit(target) − mean(logit(foils))")
    ax.set_title("Layer 2 — Logit Lens: All Circuits (pythia-14m)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  Saved: {out_path}")
