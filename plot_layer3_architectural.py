"""
plot_layer3_architectural.py
============================
Visualise Layer 3 architectural signature data.

Produces separate figures:
    layer3_{role}_clause.png        — learned τ_arch clause card
    layer3_motif_heatmap.png        — motif distribution across circuits
    layer3_distance_matrix.png      — pairwise signature distance (semantic + arch)
    layer3_component_ratios.png     — attn:mlp ratios across circuits
    layer3_summary_table.png/.tex   — summary table

Usage:
    python plot_layer3_architectural.py
"""

import os
import sys
import glob
import json
import re
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Default paths — overridden when imported by run_experiment.py
# Scan analysis/{model}/ilp and analysis/{model}/results for any model
_ANALYSIS_ROOT = os.path.join(PROJECT_ROOT, "analysis")
ILP_DIR = None
RESULTS_DIR = None
FIGURES_DIR = os.path.join(_ANALYSIS_ROOT, "figures", "layer3")
# Find first model subdirectory that has ilp/ and results/
if os.path.isdir(_ANALYSIS_ROOT):
    for _d in sorted(os.listdir(_ANALYSIS_ROOT)):
        _md = os.path.join(_ANALYSIS_ROOT, _d)
        if os.path.isdir(os.path.join(_md, "ilp")) and ILP_DIR is None:
            ILP_DIR = os.path.join(_md, "ilp")
        if os.path.isdir(os.path.join(_md, "results")) and RESULTS_DIR is None:
            RESULTS_DIR = os.path.join(_md, "results")
# Fall back to legacy paths
if not ILP_DIR or not os.path.isdir(ILP_DIR):
    ILP_DIR = os.path.join(PROJECT_ROOT, "data", "ilp")
if not RESULTS_DIR or not os.path.isdir(RESULTS_DIR):
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results_json")
if not os.path.isdir(FIGURES_DIR):
    FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {
    "bg":      "#F8FAFC",
    "attn":    "#6366F1",
    "mlp":     "#F59E0B",
}
TASK_COLORS = {
    "location": "#3B82F6",
    "time":     "#10B981",
    "path":     "#8B5CF6",
    "ioi":      "#F97316",
}


# ---------------------------------------------------------------------------
# Parse arch background .pl files
# ---------------------------------------------------------------------------

def _parse_arch_pl(path: str) -> dict:
    """Extract learned clause, confidence, motifs, and circuit stats from .pl file."""
    with open(path) as f:
        text = f.read()

    role = os.path.basename(path).replace("_arch_background.pl", "")

    # Learned clause from header
    clause_match = re.search(r"%\s*Learned clause:\n%\s*(.*)", text)
    clause = clause_match.group(1).strip() if clause_match else "—"

    # Confidence from header
    conf_match = re.search(r"%\s*Confidence:\s*([\d.]+)", text)
    confidence = float(conf_match.group(1)) if conf_match else 0.0

    # Motifs from positive circuit
    motifs = sorted(set(re.findall(r"has_motif\(C_pos\d+,\s*(\w+)\)\.", text)))

    # All motifs across all circuits
    all_motifs = sorted(set(re.findall(r"has_motif\(C_\w+,\s*(\w+)\)\.", text)))

    # Positive circuit stats
    pos_size = re.search(r"size\(C_pos1,\s*(\d+)\)", text)
    pos_attn = re.search(r"num_components\(C_pos1,\s*attention,\s*(\d+)\)", text)
    pos_mlp = re.search(r"num_components\(C_pos1,\s*mlp,\s*(\d+)\)", text)

    # S_ILP from the learned clause
    from trace.mechint.storage.ilp.ilp_arch import _arch_signature
    sig = _arch_signature(clause, motifs)

    return {
        "role": role,
        "clause": clause,
        "confidence": confidence,
        "motifs": motifs,
        "all_motifs": all_motifs,
        "size": int(pos_size.group(1)) if pos_size else 0,
        "n_attn": int(pos_attn.group(1)) if pos_attn else 0,
        "n_mlp": int(pos_mlp.group(1)) if pos_mlp else 0,
        "s_ilp": sig,
    }


def _load_all_arch() -> list:
    files = sorted(glob.glob(os.path.join(ILP_DIR, "*_arch_background.pl")))
    return [_parse_arch_pl(f) for f in files]


# ===========================================================================
# Plot 1: Learned clause cards
# ===========================================================================

def plot_clause_card(arch: dict, out_path: str):
    """A formatted card showing the learned τ_arch clause."""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.axis("off")

    role = arch["role"].upper()
    clause = arch["clause"]
    conf = arch["confidence"]
    sig = arch["s_ilp"]
    motifs = arch["motifs"]

    # Format clause for display (wrap long lines)
    if ":-" in clause:
        head, body = clause.split(":-", 1)
        body_parts = []
        depth = 0
        current = ""
        for ch in body:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                body_parts.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            body_parts.append(current.strip().rstrip("."))
        clause_display = f"{head.strip()} :-\n    " + ",\n    ".join(body_parts) + "."
    else:
        clause_display = clause

    text_lines = [
        f"τ_arch for {role}",
        "",
        clause_display,
        "",
        f"Confidence: {conf:.3f}",
        f"S_ILP: n_clauses={sig['n_clauses']}, "
        f"literals={sig['avg_length']}, "
        f"depth={sig['avg_depth']}, "
        f"complexity={sig['complexity']:.2f}",
        f"Predicates: {', '.join(sig['predicates'])}",
        f"Motifs: {', '.join(motifs[:5])}" + (f" +{len(motifs)-5} more" if len(motifs) > 5 else ""),
    ]

    color = TASK_COLORS.get(arch["role"], "#94A3B8")
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96,
                                fill=True, facecolor="white",
                                edgecolor=color, linewidth=3,
                                transform=ax.transAxes))

    y = 0.88
    for i, line in enumerate(text_lines):
        if i == 0:
            ax.text(0.5, y, line, transform=ax.transAxes,
                    fontsize=16, fontweight="bold", ha="center", color=color)
        elif ":-" in line or line.startswith("    "):
            ax.text(0.08, y, line, transform=ax.transAxes,
                    fontsize=10, fontfamily="monospace", va="top")
        else:
            ax.text(0.08, y, line, transform=ax.transAxes,
                    fontsize=10, va="top")
        y -= 0.11

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Plot 2: Motif heatmap across circuits
# ===========================================================================

def plot_motif_heatmap(archs: list, out_path: str):
    """Binary heatmap: circuits x motifs."""
    # Collect all motifs across all positive circuits
    all_motifs = sorted(set(m for a in archs for m in a["motifs"]))
    roles = [a["role"].upper() for a in archs]

    if not all_motifs:
        return

    data = np.zeros((len(archs), len(all_motifs)))
    for i, a in enumerate(archs):
        for j, m in enumerate(all_motifs):
            if m in a["motifs"]:
                data[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(max(8, len(all_motifs) * 0.7), max(3, len(archs) * 0.8)),
                            facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(all_motifs)))
    ax.set_xticklabels(all_motifs, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(roles)))
    ax.set_yticklabels(roles, fontsize=11)

    for i in range(len(roles)):
        for j in range(len(all_motifs)):
            ax.text(j, i, "+" if data[i, j] else "",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if data[i, j] else "#CBD5E1")

    ax.set_title("Layer 3 — Motif Distribution (positive circuits, pythia-14m)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Plot 3: Signature distance matrix
# ===========================================================================

def _plot_single_matrix(ax, matrix: dict, tasks: list, title: str, cmap: str = "YlOrRd"):
    """Helper: draw a single distance matrix heatmap on the given axes."""
    ax.set_facecolor(COLORS["bg"])
    data = np.array([[matrix.get(t1, {}).get(t2, 0) for t2 in tasks] for t1 in tasks])
    vmax = max(0.5, data.max()) if data.size else 0.5
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t.upper() for t in tasks], fontsize=10)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t.upper() for t in tasks], fontsize=10)
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            color = "white" if data[i, j] > 0.3 else "black"
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
    ax.set_title(title, fontsize=11, fontweight="bold")
    return im


def _build_circuit_distance_matrix(tasks: list) -> dict:
    """Compute d_circuit pairwise from results JSON files."""
    from trace.mechint.storage.ilp.circuit_distance import circuit_distance

    circuits = {}
    for task in tasks:
        # Try multiple naming patterns
        for pattern in [f"{task}_results.json", f"pythia14m_{task}_results.json"]:
            path = os.path.join(RESULTS_DIR, pattern)
            if os.path.isfile(path):
                with open(path) as f:
                    record = json.load(f)
                circuit_data = record.get("layer1", {}).get("circuit", {})
                n_layers = circuit_data.get("n_layers", 6)
                if circuit_data.get("nodes"):
                    circuits[task] = (circuit_data, n_layers)
                break

    matrix = {}
    for t1 in tasks:
        matrix[t1] = {}
        for t2 in tasks:
            if t1 == t2:
                matrix[t1][t2] = 0.0
            elif t1 in circuits and t2 in circuits:
                c1, nl1 = circuits[t1]
                c2, nl2 = circuits[t2]
                matrix[t1][t2] = circuit_distance(c1, c2, max(nl1, nl2))
            else:
                matrix[t1][t2] = 1.0  # missing data
    return matrix


def plot_distance_matrix_signature(out_path: str):
    """Pairwise d_signature heatmap (semantic + architectural)."""
    report_path = os.path.join(RESULTS_DIR, "distance_report.json")
    if not os.path.isfile(report_path):
        print(f"  Skipped: {report_path} not found")
        return

    with open(report_path) as f:
        report = json.load(f)

    tasks = report.get("tasks", [])
    if not tasks:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=COLORS["bg"])
    im1 = _plot_single_matrix(ax1, report["semantic_matrix"], tasks,
                               "d_signature (semantic)", "YlOrRd")
    im2 = _plot_single_matrix(ax2, report["architectural_matrix"], tasks,
                               "d_signature (architectural)", "YlOrRd")
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle("Signature Distance — learned theory comparison (pythia-14m)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_distance_matrix_circuit(out_path: str):
    """Pairwise d_circuit heatmap."""
    report_path = os.path.join(RESULTS_DIR, "distance_report.json")
    if not os.path.isfile(report_path):
        print(f"  Skipped: {report_path} not found")
        return

    with open(report_path) as f:
        report = json.load(f)

    tasks = report.get("tasks", [])
    if not tasks:
        return

    circuit_matrix = _build_circuit_distance_matrix(tasks)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=COLORS["bg"])
    im = _plot_single_matrix(ax, circuit_matrix, tasks,
                              "d_circuit (graph topology)", "PuBu")
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Circuit Distance — structural graph comparison (pythia-14m)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_distance_comparison(out_path: str):
    """All three distances side by side: d_sem, d_arch, d_circuit."""
    report_path = os.path.join(RESULTS_DIR, "distance_report.json")
    if not os.path.isfile(report_path):
        print(f"  Skipped: {report_path} not found")
        return

    with open(report_path) as f:
        report = json.load(f)

    tasks = report.get("tasks", [])
    if not tasks:
        return

    circuit_matrix = _build_circuit_distance_matrix(tasks)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5), facecolor=COLORS["bg"])

    im1 = _plot_single_matrix(ax1, report["semantic_matrix"], tasks,
                               "d_signature (semantic)\nlearned predicates", "YlOrRd")
    im2 = _plot_single_matrix(ax2, report["architectural_matrix"], tasks,
                               "d_signature (architectural)\nlearned clauses + motifs", "YlOrRd")
    im3 = _plot_single_matrix(ax3, circuit_matrix, tasks,
                               "d_circuit (structural)\ngraph topology", "PuBu")

    for im, ax in [(im1, ax1), (im2, ax2), (im3, ax3)]:
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Three Distance Views — theory vs. structure (pythia-14m)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Plot 4: Component ratios
# ===========================================================================

def plot_component_ratios(archs: list, out_path: str):
    """Stacked bar chart of attn vs MLP per circuit."""
    roles = [a["role"].upper() for a in archs]
    attn = [a["n_attn"] for a in archs]
    mlp = [a["n_mlp"] for a in archs]
    x = np.arange(len(roles))

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.bar(x, attn, label="Attention Heads", color=COLORS["attn"])
    ax.bar(x, mlp, bottom=attn, label="MLP Blocks", color=COLORS["mlp"])

    for i, (a, m) in enumerate(zip(attn, mlp)):
        total = a + m
        if total > 0:
            ax.text(i, total + 0.3, f"{a}:{m}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(roles, fontsize=11)
    ax.set_ylabel("Component Count")
    ax.set_title("Layer 3 — Component Composition (positive circuits, pythia-14m)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Plot 5: Summary table + LaTeX
# ===========================================================================

def plot_summary_table(archs: list, out_path: str):
    """Summary table as PNG."""
    col_labels = ["Circuit", "τ_arch clause", "Conf.", "Pred.", "Motifs", "Cmplx."]
    rows = []
    for a in archs:
        clause_short = a["clause"]
        if len(clause_short) > 50:
            clause_short = clause_short[:47] + "..."
        rows.append([
            a["role"].upper(),
            clause_short,
            f"{a['confidence']:.2f}",
            ", ".join(a["s_ilp"]["predicates"]),
            str(len(a["motifs"])),
            f"{a['s_ilp']['complexity']:.2f}",
        ])

    fig, ax = plt.subplots(figsize=(14, max(2, len(archs) * 0.7 + 1)), facecolor=COLORS["bg"])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1E293B")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Make clause column wider
    for i in range(len(rows) + 1):
        table[i, 1].set_width(0.35)
    ax.set_title("Layer 3 — Architectural Signature Summary (pythia-14m)",
                 fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  Saved: {out_path}")


def export_summary_latex(archs: list, out_path: str):
    """LaTeX table of architectural signatures."""
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Layer~3 architectural signatures learned by ILP$_{\text{arch}}$ (Pythia-14M).}",
        r"  \label{tab:layer3-arch}",
        r"  \begin{tabular}{l p{6cm} r l r r}",
        r"    \toprule",
        r"    Circuit & $\tau_{\text{arch}}$ clause & Conf. & Predicates & Motifs & Cmplx. \\",
        r"    \midrule",
    ]
    for a in archs:
        clause_tex = a["clause"].replace("_", r"\_").replace(">", r"$>$").replace("<", r"$<$")
        if len(clause_tex) > 80:
            clause_tex = clause_tex[:77] + "..."
        preds = ", ".join(a["s_ilp"]["predicates"]).replace("_", r"\_")
        lines.append(
            f"    {a['role'].upper()} & \\texttt{{{clause_tex}}} & "
            f"{a['confidence']:.2f} & {preds} & "
            f"{len(a['motifs'])} & {a['s_ilp']['complexity']:.2f} \\\\"
        )
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
    archs = _load_all_arch()
    print(f"Loaded {len(archs)} architectural signatures:")
    for a in archs:
        print(f"  {a['role']}: conf={a['confidence']:.3f}, "
              f"motifs={len(a['motifs'])}, "
              f"complexity={a['s_ilp']['complexity']:.2f}")

    # Per-circuit clause cards
    for a in archs:
        plot_clause_card(a, os.path.join(FIGURES_DIR, f"layer3_{a['role']}_clause.png"))

    # Cross-circuit comparisons
    plot_motif_heatmap(archs, os.path.join(FIGURES_DIR, "layer3_motif_heatmap.png"))
    plot_distance_matrix_signature(os.path.join(FIGURES_DIR, "layer3_distance_signature.png"))
    plot_distance_matrix_circuit(os.path.join(FIGURES_DIR, "layer3_distance_circuit.png"))
    plot_distance_comparison(os.path.join(FIGURES_DIR, "layer3_distance_comparison.png"))
    plot_component_ratios(archs, os.path.join(FIGURES_DIR, "layer3_component_ratios.png"))
    plot_summary_table(archs, os.path.join(FIGURES_DIR, "layer3_summary_table.png"))
    export_summary_latex(archs, os.path.join(FIGURES_DIR, "layer3_summary_table.tex"))

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
