"""
plot_layer1_circuits.py
=======================
Visualise Layer 1 structural data from stored TTL files.

Produces one figure PER CIRCUIT plus a comparison figure:
  ./figures/layer1_{role}_topology.png   — circuit graph
  ./figures/layer1_comparison.png        — cross-circuit comparison

Usage:
    python plot_layer1_circuits.py
"""

import os
import sys
import glob
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from rdflib import Graph as RDFGraph, Namespace, RDF

# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from trace.mechint.ontology.schema import MECHINT

from test_layer1_structural import (
    _discover_ttl_files, _parse_ttl_id, _load_rdf,
    _find_circuit_uri, extract_layer1,
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "attn":    "#6366F1",
    "mlp":     "#F59E0B",
    "input":   "#94A3B8",
    "output":  "#94A3B8",
    "unknown": "#CBD5E1",
    "edge_hi": "#EF4444",
    "edge_lo": "#CBD5E1",
    "bg":      "#F8FAFC",
}

TASK_COLORS = {
    "location": "#3B82F6",
    "time":     "#10B981",
    "path":     "#8B5CF6",
    "ioi":      "#F97316",
}


def _build_nx_graph(l1: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in l1["nodes"]:
        addr = node.get("address", "?")
        G.add_node(addr,
                   component_type=node["component_type"],
                   layer=node.get("layer", -1),
                   head=node.get("head", -1),
                   relative_depth=node.get("relative_depth", 0.0))
    for edge in l1["edges"]:
        src = edge.get("source", "?")
        tgt = edge.get("target", "?")
        score = edge.get("score", 0.0)
        G.add_edge(src, tgt, score=abs(score) if score else 0.0, raw_score=score or 0.0)
    return G


def _node_colors(G: nx.DiGraph) -> list:
    return [COLORS.get(G.nodes[n].get("component_type", "unknown"), COLORS["unknown"])
            for n in G.nodes()]


def _edge_widths(G: nx.DiGraph, min_w=0.5, max_w=4.0) -> list:
    scores = [G.edges[e].get("score", 0.0) for e in G.edges()]
    if not scores or max(scores) == 0:
        return [min_w] * len(scores)
    mx = max(scores)
    return [min_w + (s / mx) * (max_w - min_w) for s in scores]


def _edge_colors(G: nx.DiGraph) -> list:
    scores = [G.edges[e].get("score", 0.0) for e in G.edges()]
    if not scores or max(scores) == 0:
        return [COLORS["edge_lo"]] * len(scores)
    mx = max(scores)
    return [COLORS["edge_hi"] if s > 0.5 * mx else COLORS["edge_lo"] for s in scores]


def _layout(G: nx.DiGraph) -> dict:
    """Layer-based layout: x = layer, y = spread within layer."""
    layers = {}
    for n, data in G.nodes(data=True):
        layer = data.get("layer", -1)
        layers.setdefault(layer, []).append(n)
    pos = {}
    sorted_layers = sorted(layers.keys())
    for x_idx, layer in enumerate(sorted_layers):
        nodes_in_layer = layers[layer]
        n = len(nodes_in_layer)
        for y_idx, node in enumerate(nodes_in_layer):
            y = (y_idx - (n - 1) / 2) * 1.2
            pos[node] = (x_idx * 2, y)
    return pos


# ===========================================================================
# Plot: Individual circuit topology (one figure per circuit)
# ===========================================================================

def plot_single_circuit(info: dict, l1: dict, out_path: str):
    """One figure for a single circuit's topology."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    G = _build_nx_graph(l1)

    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, f"{info['role_raw'].upper()}\n(empty circuit)",
                ha="center", va="center", fontsize=18, color="#EF4444")
        ax.set_title(info["role_raw"].upper(), fontsize=16, fontweight="bold")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        plt.close(fig)
        print(f"  Saved: {out_path}")
        return

    pos = _layout(G)
    node_colors = _node_colors(G)
    edge_widths = _edge_widths(G)
    edge_colors = _edge_colors(G)

    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=edge_widths,
                           edge_color=edge_colors,
                           alpha=0.7,
                           arrows=True,
                           arrowsize=18,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=900,
                           edgecolors="#1E293B",
                           linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            font_size=9,
                            font_weight="bold",
                            font_color="white")

    # Edge score labels
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        s = data.get("raw_score", 0)
        if abs(s) > 0:
            edge_labels[(u, v)] = f"{s:.4f}"
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax,
                                  font_size=7, font_color="#64748B")

    # Title with metrics
    f = l1["metrics"].get("faithfulness", 0)
    n_attn = len([n for n in l1["nodes"] if n["component_type"] == "attn"])
    n_mlp = len([n for n in l1["nodes"] if n["component_type"] == "mlp"])
    n_edges = len(l1["edges"])
    ratio = l1["aggregates"].get("componentRatio", "?")
    d_min = l1["aggregates"].get("depthMin", 0)
    d_max = l1["aggregates"].get("depthMax", 0)

    title = f"Layer 1 — {info['role_raw'].upper()} Circuit Topology (pythia-14m)"
    subtitle = (f"attn={n_attn}  mlp={n_mlp}  edges={n_edges}  "
                f"ratio={ratio}  depth=[{d_min:.2f}, {d_max:.2f}]  "
                f"faithfulness={f:.3f}")
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold")

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLORS["attn"], label="Attention Head"),
        mpatches.Patch(color=COLORS["mlp"], label="MLP Block"),
        mpatches.Patch(color=COLORS["input"], label="Input / Output"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10, frameon=True)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Plot: Cross-circuit structural comparison
# ===========================================================================

def plot_component_counts(circuits: list, out_path: str):
    """Bar chart of component & edge counts across circuits."""
    if not circuits:
        return
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    names = [c[0]["role_raw"].upper() for c in circuits]
    x = np.arange(len(names))
    attn_counts = [len([n for n in c[1]["nodes"] if n["component_type"] == "attn"]) for c in circuits]
    mlp_counts = [len([n for n in c[1]["nodes"] if n["component_type"] == "mlp"]) for c in circuits]
    edge_counts = [len(c[1]["edges"]) for c in circuits]
    w = 0.25
    ax.bar(x - w, attn_counts, w, label="Attn Heads", color=COLORS["attn"])
    ax.bar(x, mlp_counts, w, label="MLP Blocks", color=COLORS["mlp"])
    ax.bar(x + w, edge_counts, w, label="Edges", color="#64748B")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title("Layer 1 — Component & Edge Counts (pythia-14m)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_depth_profile(circuits: list, out_path: str):
    """Horizontal bars showing depth range per circuit."""
    if not circuits:
        return
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    names = [c[0]["role_raw"].upper() for c in circuits]
    colors = [TASK_COLORS.get(c[0]["role_raw"], "#94A3B8") for c in circuits]
    for i, (info, l1) in enumerate(circuits):
        d_min = float(l1["aggregates"].get("depthMin", 0))
        d_max = float(l1["aggregates"].get("depthMax", 0))
        d_mean = float(l1["aggregates"].get("depthMean", (d_min + d_max) / 2))
        ax.barh(i, d_max - d_min, left=d_min, height=0.5,
                color=colors[i], alpha=0.6, edgecolor=colors[i])
        ax.plot(d_mean, i, 'D', color=colors[i], markersize=10, zorder=5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Relative Depth ρ(ℓ) = ℓ/L")
    ax.set_xlim(-0.05, 1.05)
    ax.set_title("Layer 1 — Depth Profile (pythia-14m)\n◆ = mean depth",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_faithfulness(circuits: list, out_path: str):
    """Bar chart of faithfulness per circuit."""
    if not circuits:
        return
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    names = [c[0]["role_raw"].upper() for c in circuits]
    colors = [TASK_COLORS.get(c[0]["role_raw"], "#94A3B8") for c in circuits]
    x = np.arange(len(names))
    faiths = [c[1]["metrics"].get("faithfulness", 0) for c in circuits]
    bars = ax.bar(x, faiths, color=colors, edgecolor="#1E293B", linewidth=1)
    for bar, f in zip(bars, faiths):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{f:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Faithfulness")
    ax.set_title("Layer 1 — Circuit Faithfulness (pythia-14m)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    print(f"  Saved: {out_path}")
    plt.close(fig)


def _build_table_rows(circuits: list) -> tuple:
    """Build table rows shared by the PNG and LaTeX exporters."""
    col_labels = ["Circuit", "Attn", "MLP", "Edges", "Depth", "Ratio", "Faith.", "$\\varepsilon$-min"]
    rows = []
    for info, l1 in circuits:
        n_attn = len([n for n in l1["nodes"] if n["component_type"] == "attn"])
        n_mlp = len([n for n in l1["nodes"] if n["component_type"] == "mlp"])
        n_edges = len(l1["edges"])
        d_min = l1["aggregates"].get("depthMin", 0)
        d_max = l1["aggregates"].get("depthMax", 0)
        depth = f"{d_min:.2f}--{d_max:.2f}" if isinstance(d_min, float) else "---"
        ratio = l1["aggregates"].get("componentRatio", "?")
        faith = f'{l1["metrics"].get("faithfulness", 0):.3f}'
        eps = l1["metrics"].get("epsilonMinimal", "N/A")
        if isinstance(eps, float):
            eps = "Yes" if eps > 0.5 else "No"
        rows.append([info["role_raw"].upper(), n_attn, n_mlp, n_edges,
                     depth, ratio, faith, eps])
    return col_labels, rows


def plot_summary_table(circuits: list, out_path: str):
    """Summary table of structural properties (PNG)."""
    if not circuits:
        return
    col_labels, rows = _build_table_rows(circuits)
    # For matplotlib, use plain text labels
    col_labels_plain = [c.replace("$\\varepsilon$", "ε") for c in col_labels]

    fig, ax = plt.subplots(figsize=(10, 3), facecolor=COLORS["bg"])
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=col_labels_plain,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    for j in range(len(col_labels_plain)):
        table[0, j].set_facecolor("#1E293B")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Layer 1 — Structural Summary (pythia-14m)",
                 fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    print(f"  Saved: {out_path}")
    plt.close(fig)


def export_summary_latex(circuits: list, out_path: str):
    """Export the structural summary as a LaTeX table."""
    if not circuits:
        return
    col_labels, rows = _build_table_rows(circuits)

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Layer~1 structural summary for circuits discovered in Pythia-14M.}",
        r"  \label{tab:layer1-structural}",
        r"  \begin{tabular}{l r r r c c r c}",
        r"    \toprule",
        "    " + " & ".join(col_labels) + r" \\",
        r"    \midrule",
    ]
    for row in rows:
        cells = [str(c) for c in row]
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    out_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(out_dir, exist_ok=True)

    ttl_files = _discover_ttl_files()
    circuits = []
    for ttl_path in ttl_files:
        info = _parse_ttl_id(ttl_path)
        rdf = _load_rdf(ttl_path)
        circuit_uri = _find_circuit_uri(rdf)
        l1 = extract_layer1(rdf, circuit_uri)
        circuits.append((info, l1))

        n_attn = len([n for n in l1["nodes"] if n["component_type"] == "attn"])
        n_mlp = len([n for n in l1["nodes"] if n["component_type"] == "mlp"])
        n_edges = len(l1["edges"])
        print(f"  {info['stem']}: {n_attn} attn, {n_mlp} mlp, {n_edges} edges")

        # Individual circuit figure
        plot_single_circuit(
            info, l1,
            os.path.join(out_dir, f"layer1_{info['role_raw']}_topology.png")
        )

    # Separate comparison figures
    plot_component_counts(circuits, os.path.join(out_dir, "layer1_component_counts.png"))
    plot_depth_profile(circuits, os.path.join(out_dir, "layer1_depth_profile.png"))
    plot_faithfulness(circuits, os.path.join(out_dir, "layer1_faithfulness.png"))
    plot_summary_table(circuits, os.path.join(out_dir, "layer1_summary_table.png"))
    export_summary_latex(circuits, os.path.join(out_dir, "layer1_summary_table.tex"))

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
