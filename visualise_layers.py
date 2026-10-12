"""
visualise_layers.py
===================
Publication-ready visualisations for the four-layer CKG ontology.

Generates 5 PNG figures:
    layer0_provenance.png     — provenance metadata table (L0)
    layer1_structural.png     — circuit topology + metrics heatmap (L1)
    layer2_semantic.png       — ILP_sem specification + predicate graph (L2)
    layer3_architectural.png  — ILP_arch clause + radar profile (L3)
    summary_table.png         — full 4-layer comparison table (all)

Data sources (in priority order):
    1. Real TTL exports in `ttl_dir` (via SPARQL)
    2. Real JSON results in `json_dir` (from results_export.py)
    3. Mock data as fallback for development

Public API
----------
    viz_data = load_circuit_data(task, ttl_dir, json_dir)
    plot_layer0(viz_data, out_dir)
    plot_layer1(viz_data, out_dir)
    plot_layer2(viz_data, out_dir)
    plot_layer3(viz_data, out_dir)
    plot_summary_table(viz_data, out_dir)

    # Or one-shot:
    visualise_all(ttl_dir, json_dir, out_dir)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import numpy as np

# ── Design tokens ─────────────────────────────────────────────────────────
_FONT_FAMILY = "DejaVu Sans"

_PALETTE = {
    "bg":          "#0d1117",
    "panel":       "#161b22",
    "border":      "#30363d",
    "text":        "#e6edf3",
    "muted":       "#8b949e",
    "accent":      "#58a6ff",
    "green":       "#3fb950",
    "yellow":      "#d29922",
    "orange":      "#f0883e",
    "red":         "#f85149",
    "purple":      "#bc8cff",
    "attn":        "#238636",
    "mlp":         "#1f6feb",
    "L0":          "#0d1117",
    "L1":          "#0d1117",
    "L2":          "#0d1117",
    "L3":          "#0d1117",
}

_LAYER_ACCENT = {
    0: "#58a6ff",   # blue
    1: "#3fb950",   # green
    2: "#bc8cff",   # purple
    3: "#f0883e",   # orange
}

_LAYER_NAME = {
    0: "Provenance",
    1: "Structural",
    2: "Semantic",
    3: "Architectural",
}

plt.rcParams.update({
    "font.family":         _FONT_FAMILY,
    "text.color":          _PALETTE["text"],
    "axes.facecolor":      _PALETTE["panel"],
    "axes.edgecolor":      _PALETTE["border"],
    "axes.labelcolor":     _PALETTE["text"],
    "xtick.color":         _PALETTE["muted"],
    "ytick.color":         _PALETTE["muted"],
    "figure.facecolor":    _PALETTE["bg"],
    "grid.color":          _PALETTE["border"],
    "grid.alpha":          0.4,
})


# ============================================================
# Data loading
# ============================================================

def _load_from_json(task: str, json_dir: str) -> Optional[Dict]:
    """Load circuit data from a per-task JSON results file."""
    path = os.path.join(json_dir, f"{task.lower()}_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _load_from_ttl(task: str, ttl_dir: str) -> Optional[Dict]:
    """Load circuit data from TTL files via rdflib."""
    try:
        from rdflib import Graph, Namespace
        import glob

        # Find matching TTL file
        patterns = [
            f"{ttl_dir}/*{task.lower()}*pythia*.ttl",
            f"{ttl_dir}/*{task.lower()}*.ttl",
            f"{ttl_dir}/*semanticroles*{task.lower()}*.ttl",
        ]
        matches = []
        for pat in patterns:
            matches.extend(glob.glob(pat))
        if not matches:
            return None

        ttl_path = sorted(matches)[-1]
        g = Graph()
        g.parse(ttl_path, format="turtle")
        ns = Namespace("http://purl.org/mechint#")

        # Find circuit URI
        q_uri = "SELECT ?c WHERE { ?c a <http://purl.org/mechint#Circuit> . } LIMIT 1"
        rows = list(g.query(q_uri))
        if not rows:
            return None
        circuit_uri = str(rows[0][0])

        # Extract basic structural data
        agg_q = f"""
        PREFIX mechint: <{ns}>
        SELECT ?dMin ?dMax ?dMean ?faith ?numAttn ?numMlp ?ratio WHERE {{
            <{circuit_uri}> mechint:depthMin  ?dMin  .
            <{circuit_uri}> mechint:depthMax  ?dMax  .
            <{circuit_uri}> mechint:depthMean ?dMean .
            OPTIONAL {{ <{circuit_uri}> mechint:faithfulness        ?faith   }}
            OPTIONAL {{ <{circuit_uri}> mechint:numAttentionHeads   ?numAttn }}
            OPTIONAL {{ <{circuit_uri}> mechint:numMLPBlocks        ?numMlp  }}
            OPTIONAL {{ <{circuit_uri}> mechint:componentRatio      ?ratio   }}
        }}
        """
        agg_rows = list(g.query(agg_q))

        nodes_q = f"""
        PREFIX mechint: <{ns}>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?comp ?ctype ?layer WHERE {{
            <{circuit_uri}> mechint:hasComponent ?comp .
            ?comp rdf:type ?ctype .
            OPTIONAL {{ ?comp mechint:inLayer ?layer }}
        }}
        """
        nodes = []
        for row in g.query(nodes_q):
            ctype = str(row.ctype).split("#")[-1]
            ntype = "attn" if "AttentionHead" in ctype else "mlp"
            name  = str(row.comp).split("#")[-1]
            layer = int(row.layer) if row.layer is not None else 0
            nodes.append({"name": name, "layer": layer, "type": ntype})

        # Construct dict compatible with JSON results format
        data: Dict[str, Any] = {
            "meta":   {"task": task, "model_name": "pythia-14m", "owl_path": ttl_path},
            "layer0": {"task_family": task, "role": task, "model_name": "pythia-14m",
                       "attribution_method": "EAP-IG-inputs", "top_k": 50},
            "layer1": {
                "circuit": {"nodes": nodes, "num_nodes": len(nodes)},
                "metrics": {},
                "validation": {},
            },
            "layer2": None,
            "layer3": None,
        }

        if agg_rows:
            r = agg_rows[0]
            data["layer1"]["metrics"] = {
                "faithfulness":        float(r.faith)   if r.faith   else None,
                "num_attn_heads":      int(r.numAttn)   if r.numAttn else None,
                "num_mlp_blocks":      int(r.numMlp)    if r.numMlp  else None,
                "depth_mean":          float(r.dMean)   if r.dMean   else None,
            }

        return data

    except Exception as e:
        print(f"  [visualise] TTL load failed for {task}: {e}")
        return None


def _mock_data(task: str) -> Dict:
    """Canonical mock data — used when neither TTL nor JSON is available."""
    is_ioi = task.upper() == "IOI"
    return {
        "meta": {
            "task":        task,
            "model_name":  "pythia-14m",
            "owl_path":    f"./data/experiments/{task.lower()}.ttl",
        },
        "layer0": {
            "task_family":        "IOI" if is_ioi else "SemanticRoles",
            "role":               task,
            "model_name":         "pythia-14m",
            "attribution_method": "EAP-IG-inputs",
            "top_k":              50,
        },
        "layer1": {
            "circuit": {
                "num_nodes": 12 if is_ioi else 9,
                "num_edges": 50,
                "n_layers":  6,
                "n_heads":   4,
                "nodes": [
                    {"name": "a0.h0", "layer": 0, "type": "attn", "score": 0.41},
                    {"name": "a0.h3", "layer": 0, "type": "attn", "score": 0.35},
                    {"name": "a1.h1", "layer": 1, "type": "attn", "score": 0.28},
                    {"name": "m2",    "layer": 2, "type": "mlp",  "score": 0.47},
                    {"name": "a3.h0", "layer": 3, "type": "attn", "score": 0.62},
                    {"name": "a3.h2", "layer": 3, "type": "attn", "score": 0.55},
                    {"name": "a4.h1", "layer": 4, "type": "attn", "score": 0.71},
                    {"name": "m4",    "layer": 4, "type": "mlp",  "score": 0.38},
                    {"name": "a5.h0", "layer": 5, "type": "attn", "score": 0.83},
                ],
            },
            "metrics": {
                "faithfulness":          0.829 if is_ioi else 0.812,
                "faithfulness_simple":   0.856 if is_ioi else 0.843,
                "baseline_performance":  0.912,
                "circuit_performance":   0.781,
                "ablated_performance":   0.081,
                "performance_drop":      0.131,
                "num_nodes_in_circuit":  12 if is_ioi else 9,
                "num_edges_in_circuit":  50,
            },
            "validation": {
                "epsilon_minimal":           True,
                "permutation_pvalue":        0.01,
                "statistically_significant": True,
            },
        },
        "layer2": {
            "spec_text":  (
                "indirect_object(C, N) :- is_name(N, _), not_repeated(N)."
                if is_ioi else
                "location_entity(C, E) :- role(E, location)."
            ),
            "predicates": (
                ["is_name", "not_repeated"] if is_ioi else ["role"]
            ),
            "confidence":  0.87 if is_ioi else 0.91,
            "ilp_signature": {
                "n_clauses":  1,
                "avg_depth":  1.0,
                "avg_length": 2.0 if is_ioi else 1.0,
                "complexity": 0.20,
                "predicates": ["is_name", "not_repeated"] if is_ioi else ["role"],
            },
        },
        "layer3": {
            "depth_mean":          0.44 if is_ioi else 0.51,
            "depth_min":           0.0,
            "depth_max":           0.83,
            "layer_span":          0.83,
            "component_ratio":     "10:2" if is_ioi else "7:2",
            "num_attn_heads":      10 if is_ioi else 7,
            "num_mlp_blocks":      2,
            "motifs":              (
                ["induction_head", "token_copy"] if is_ioi
                else ["scaffold_event_entity"]
            ),
            "tau_arch_text": (
                "arch_ioi(C) :- layer_span(C, Lmin, Lmax), Lmin >= 0.0, Lmax =< 0.83, "
                "component_ratio(C, attention, R), R > 0.71, size(C, N), N < 17."
                if is_ioi else
                "arch_location(C) :- layer_span(C, Lmin, Lmax), Lmin >= 0.0, Lmax =< 0.83, "
                "has_motif(C, scaffold_event_entity), size(C, N), N < 15."
            ),
            "tau_arch_confidence": 0.82 if is_ioi else 0.78,
        },
    }


def load_circuit_data(
        task:    str,
        ttl_dir: str  = "./data/experiments",
        json_dir: str = "./data/results_json",
) -> Dict:
    """
    Load circuit data for a task in priority order:
      1. JSON results file (most complete — includes all 4 layers)
      2. TTL file via SPARQL (structural data only)
      3. Canonical mock data (for development / CI)

    The returned dict has the shape of the per-task JSON schema.
    """
    # Try JSON first (has L2/L3 if ILP has run)
    data = _load_from_json(task, json_dir)
    if data is not None:
        print(f"  {task:<10} <- JSON  (L0-L3 complete)")
        return data

    # Try TTL
    data = _load_from_ttl(task, ttl_dir)
    if data is not None:
        print(f"  {task:<10} <- TTL  (L0-L1 only)")
        return data

    # Fallback
    print(f"  {task:<10} <- MOCK (no files found)")
    return _mock_data(task)


# ============================================================
# Helper drawing utilities
# ============================================================

def _fig_ax(w: float = 14, h: float = 7) -> tuple:
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(_PALETTE["bg"])
    ax.set_facecolor(_PALETTE["panel"])
    ax.tick_params(colors=_PALETTE["muted"])
    for spine in ax.spines.values():
        spine.set_edgecolor(_PALETTE["border"])
    return fig, ax


def _layer_header(fig, layer_idx: int, title: str, subtitle: str = "") -> None:
    accent  = _LAYER_ACCENT[layer_idx]
    lname   = _LAYER_NAME[layer_idx]
    fig.text(
        0.012, 0.97,
        f"Layer {layer_idx}  ·  {lname}",
        fontsize=8, color=accent, ha="left", va="top",
        fontweight="bold",
        transform=fig.transFigure,
    )
    fig.text(
        0.012, 0.93,
        title,
        fontsize=15, color=_PALETTE["text"], ha="left", va="top",
        fontweight="bold",
        transform=fig.transFigure,
    )
    if subtitle:
        fig.text(
            0.012, 0.88,
            subtitle,
            fontsize=9, color=_PALETTE["muted"], ha="left", va="top",
            transform=fig.transFigure,
        )


def _badge(ax, x: float, y: float, text: str, color: str, size: float = 8.5):
    ax.text(
        x, y, text,
        fontsize=size, color=_PALETTE["bg"],
        ha="center", va="center", fontweight="bold",
        bbox=dict(facecolor=color, edgecolor="none",
                  boxstyle="round,pad=0.3", alpha=0.9),
        transform=ax.transAxes,
    )


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=_PALETTE["bg"])
    plt.close(fig)
    print(f"  -> {path}")
    return path


# ============================================================
# Layer 0: Provenance
# ============================================================

def plot_layer0(viz_data: Dict, out_dir: str) -> str:
    """
    Provenance panel — shows L0 metadata for each task side by side.
    """
    tasks  = list(viz_data.keys())
    n      = len(tasks)
    fig    = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor(_PALETTE["bg"])
    _layer_header(fig, 0,
                  "Circuit Registry — Provenance",
                  "Model · Task · Attribution method · Output")

    gs  = gridspec.GridSpec(1, n, figure=fig,
                            left=0.02, right=0.98, top=0.80, bottom=0.08,
                            wspace=0.04)

    fields = [
        ("Task family",        lambda d: d.get("task_family", "N/A")),
        ("Role",               lambda d: d.get("role",         "N/A")),
        ("Model",              lambda d: d.get("model_name",   "N/A")),
        ("Attribution method", lambda d: d.get("attribution_method", "N/A")),
        ("Top-k edges",        lambda d: str(d.get("top_k",   "N/A"))),
        ("OWL path",           lambda d: Path(str(d.get("owl_path",""))).name[:36]),
    ]

    for col, task in enumerate(tasks):
        ax = fig.add_subplot(gs[col])
        ax.set_facecolor(_PALETTE["panel"])
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.axis("off")

        # Task name banner
        accent = _LAYER_ACCENT[0]
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.01, 0.90), 0.98, 0.09,
            boxstyle="round,pad=0.01",
            facecolor=accent, edgecolor="none", alpha=0.2,
            transform=ax.transAxes, clip_on=False
        ))
        ax.text(0.5, 0.945, task, fontsize=13, color=accent,
                ha="center", va="center", fontweight="bold",
                transform=ax.transAxes)

        l0 = viz_data[task].get("layer0", {})
        y  = 0.80
        for label, extractor in fields:
            val = extractor(l0)
            ax.text(0.06, y, label + ":",
                    fontsize=8.5, color=_PALETTE["muted"],
                    ha="left", va="top", transform=ax.transAxes)
            ax.text(0.94, y, val,
                    fontsize=8.5, color=_PALETTE["text"],
                    ha="right", va="top", transform=ax.transAxes,
                    fontweight="bold")
            y -= 0.14
            ax.axhline(y + 0.06, xmin=0.05, xmax=0.95,
                       color=_PALETTE["border"], linewidth=0.5)

        # Bottom accent bar
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.0, 0.0), 1.0, 0.03,
            facecolor=accent, edgecolor="none", alpha=0.15,
            transform=ax.transAxes, clip_on=False
        ))

    return _save(fig, out_dir, "layer0_provenance")


# ============================================================
# Layer 1: Structural
# ============================================================

def plot_layer1(viz_data: Dict, out_dir: str) -> str:
    """
    Structural panel — node-layer heatmap + metrics table.
    """
    tasks   = list(viz_data.keys())
    fig     = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor(_PALETTE["bg"])
    _layer_header(fig, 1,
                  "Circuit Structure — EAP-IG Attribution",
                  "Node distribution across layers · Faithfulness · ε-minimality")

    n_tasks = len(tasks)
    gs = gridspec.GridSpec(
        2, n_tasks,
        figure=fig,
        left=0.04, right=0.97,
        top=0.82, bottom=0.08,
        hspace=0.38, wspace=0.10,
        height_ratios=[2.2, 1],
    )

    for col, task in enumerate(tasks):
        d      = viz_data[task]
        l1     = d.get("layer1", {})
        circ   = l1.get("circuit", {})
        nodes  = circ.get("nodes", [])
        n_lay  = circ.get("n_layers", 6)
        accent = _LAYER_ACCENT[1]

        # ── Top: node distribution heatmap by (layer × type) ──
        ax_top = fig.add_subplot(gs[0, col])
        ax_top.set_facecolor(_PALETTE["panel"])

        attn_counts = [0] * n_lay
        mlp_counts  = [0] * n_lay
        for nd in nodes:
            lyr = min(nd.get("layer", 0), n_lay - 1)
            if nd.get("type") == "attn":
                attn_counts[lyr] += 1
            else:
                mlp_counts[lyr]  += 1

        xs = np.arange(n_lay)
        ax_top.bar(xs - 0.2, attn_counts, width=0.38,
                   color=_PALETTE["attn"], alpha=0.85, label="Attn")
        ax_top.bar(xs + 0.2, mlp_counts,  width=0.38,
                   color=_PALETTE["mlp"],  alpha=0.85, label="MLP")
        ax_top.set_xticks(xs)
        ax_top.set_xticklabels([f"L{i}" for i in xs], fontsize=7.5)
        ax_top.set_ylabel("Nodes", fontsize=7.5)
        ax_top.tick_params(labelsize=7.5)
        ax_top.grid(axis="y", alpha=0.25)
        ax_top.set_title(task, fontsize=10, color=accent, fontweight="bold", pad=4)
        ax_top.legend(fontsize=7, loc="upper right",
                      facecolor=_PALETTE["bg"], edgecolor=_PALETTE["border"])

        # ── Bottom: metrics ──
        ax_bot = fig.add_subplot(gs[1, col])
        ax_bot.set_facecolor(_PALETTE["panel"])
        ax_bot.axis("off")

        met = l1.get("metrics",    {})
        val = l1.get("validation", {})

        def _fv(v, fmt=".4f"):
            return f"{v:{fmt}}" if isinstance(v, (int, float)) else str(v)

        rows_data = [
            ("Faithfulness",    _fv(met.get("faithfulness")),         accent),
            ("Baseline perf.",  _fv(met.get("baseline_performance")), _PALETTE["muted"]),
            ("Circuit perf.",   _fv(met.get("circuit_performance")),  _PALETTE["muted"]),
            ("ε-minimal",
             "✓" if val.get("epsilon_minimal") else "✗",
             _PALETTE["green"] if val.get("epsilon_minimal") else _PALETTE["red"]),
            ("p-value",         _fv(met.get("performance_drop", val.get("permutation_pvalue"))), _PALETTE["muted"]),
        ]

        for ri, (label, value, color) in enumerate(rows_data):
            y_pos = 0.88 - ri * 0.20
            ax_bot.text(0.03, y_pos, label,
                        fontsize=7.5, color=_PALETTE["muted"],
                        transform=ax_bot.transAxes, va="top")
            ax_bot.text(0.97, y_pos, value,
                        fontsize=7.5, color=color, fontweight="bold",
                        ha="right", transform=ax_bot.transAxes, va="top")

    return _save(fig, out_dir, "layer1_structural")


# ============================================================
# Layer 2: Semantic (ILP_sem results)
# ============================================================

def plot_layer2(viz_data: Dict, out_dir: str) -> str:
    """
    Semantic panel — ILP_sem learned specification, predicates, confidence.
    """
    tasks  = list(viz_data.keys())
    n      = len(tasks)
    fig    = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor(_PALETTE["bg"])
    _layer_header(fig, 2,
                  "Semantic Specification — ILP_sem",
                  "Learned FOL clauses · Predicate inventory · Confidence (Eq. 2)")

    gs = gridspec.GridSpec(
        1, n,
        figure=fig,
        left=0.03, right=0.97,
        top=0.80, bottom=0.08,
        wspace=0.05,
    )

    for col, task in enumerate(tasks):
        ax = fig.add_subplot(gs[col])
        ax.set_facecolor(_PALETTE["panel"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        accent = _LAYER_ACCENT[2]
        l2     = viz_data[task].get("layer2")

        # Banner
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.0, 0.91), 1.0, 0.08,
            facecolor=accent, edgecolor="none", alpha=0.18,
            transform=ax.transAxes, clip_on=False,
            boxstyle="round,pad=0.01",
        ))
        ax.text(0.5, 0.952, task,
                fontsize=12, color=accent, ha="center", va="center",
                fontweight="bold", transform=ax.transAxes)

        if l2 is None:
            ax.text(0.5, 0.5, "ILP_sem pending\n(ILP_sem step not run)",
                    fontsize=9, color=_PALETTE["yellow"], ha="center", va="center",
                    transform=ax.transAxes, style="italic",
                    bbox=dict(facecolor=_PALETTE["bg"], edgecolor=_PALETTE["border"],
                              boxstyle="round,pad=0.4"))
            continue

        # Specification clause (wrapped)
        spec = l2.get("spec_text", "N/A")
        y    = 0.83
        ax.text(0.04, y, "φ  (specification)",
                fontsize=7.5, color=_PALETTE["muted"],
                transform=ax.transAxes, va="top", fontweight="bold")
        y -= 0.07
        # Word-wrap the clause
        words = spec.replace(",", ",\n    ").replace(":-", ":-\n  ")
        ax.text(0.04, y, words,
                fontsize=7.5, color=_PALETTE["text"],
                transform=ax.transAxes, va="top",
                fontfamily="DejaVu Sans Mono",
                wrap=True,
                bbox=dict(facecolor="#0d1117", edgecolor=_PALETTE["border"],
                          boxstyle="round,pad=0.4", alpha=0.6))
        y -= 0.32

        # Confidence gauge
        conf = l2.get("confidence", 0.0)
        ax.text(0.04, y, "Confidence (Eq. 2)",
                fontsize=7.5, color=_PALETTE["muted"],
                transform=ax.transAxes, va="top")
        y -= 0.06
        bar_w = conf * 0.92
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.04, y - 0.04), 0.92, 0.05,
            facecolor=_PALETTE["border"], edgecolor="none",
            transform=ax.transAxes, clip_on=True,
            boxstyle="round,pad=0.01"
        ))
        conf_color = (_PALETTE["green"] if conf > 0.75 else
                      _PALETTE["yellow"] if conf > 0.5 else _PALETTE["red"])
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.04, y - 0.04), bar_w, 0.05,
            facecolor=conf_color, edgecolor="none",
            transform=ax.transAxes, clip_on=True,
            boxstyle="round,pad=0.01"
        ))
        ax.text(0.5, y - 0.015, f"{conf:.3f}",
                fontsize=8, color=_PALETTE["bg"], ha="center", va="center",
                fontweight="bold", transform=ax.transAxes)
        y -= 0.10

        # Predicates
        preds = l2.get("predicates", [])
        ax.text(0.04, y, "Predicate inventory P_H (Eq. 3)",
                fontsize=7.5, color=_PALETTE["muted"],
                transform=ax.transAxes, va="top")
        y -= 0.06
        for pi, pred in enumerate(preds[:6]):
            px = 0.04 + (pi % 2) * 0.48
            py = y - (pi // 2) * 0.08
            ax.add_patch(mpatches.FancyBboxPatch(
                (px, py - 0.05), 0.42, 0.055,
                facecolor=accent, edgecolor="none", alpha=0.22,
                transform=ax.transAxes, clip_on=True,
                boxstyle="round,pad=0.01"
            ))
            ax.text(px + 0.21, py - 0.02, pred,
                    fontsize=7.5, color=accent, ha="center", va="center",
                    fontweight="bold", transform=ax.transAxes)
        y -= max((len(preds) // 2 + 1) * 0.09, 0.18)

        # Signature table
        sig = l2.get("ilp_signature", {})
        if sig:
            ax.text(0.04, y, "S_ILP_sem (Eq. 3)",
                    fontsize=7.5, color=_PALETTE["muted"],
                    transform=ax.transAxes, va="top")
            y -= 0.06
            for field, key in [
                ("n_clauses",  "n_clauses"),
                ("avg_depth",  "avg_depth"),
                ("avg_length", "avg_length"),
                ("complexity", "complexity"),
            ]:
                val_str = str(sig.get(key, "N/A"))
                ax.text(0.04, y, field + ":",
                        fontsize=7.5, color=_PALETTE["muted"],
                        transform=ax.transAxes, va="top")
                ax.text(0.95, y, val_str,
                        fontsize=7.5, color=_PALETTE["text"],
                        ha="right", fontweight="bold",
                        transform=ax.transAxes, va="top")
                y -= 0.065

    return _save(fig, out_dir, "layer2_semantic")


# ============================================================
# Layer 3: Architectural (ILP_arch results)
# ============================================================

def plot_layer3(viz_data: Dict, out_dir: str) -> str:
    """
    Architectural panel — ILP_arch clause, structural radar, motifs.
    """
    tasks  = list(viz_data.keys())
    n      = len(tasks)
    fig    = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(_PALETTE["bg"])
    _layer_header(fig, 3,
                  "Architectural Signature — ILP_arch",
                  "Learned structural clause (Eq. 4) · Component radar · Motifs")

    gs = gridspec.GridSpec(
        1, n * 2,
        figure=fig,
        left=0.03, right=0.97,
        top=0.80, bottom=0.06,
        wspace=0.04,
    )

    for col, task in enumerate(tasks):
        ax_l = fig.add_subplot(gs[col * 2])
        ax_r = fig.add_subplot(gs[col * 2 + 1], polar=True)

        for ax in [ax_l]:
            ax.set_facecolor(_PALETTE["panel"])
            ax.axis("off")

        ax_r.set_facecolor(_PALETTE["panel"])
        ax_r.spines["polar"].set_color(_PALETTE["border"])
        ax_r.tick_params(colors=_PALETTE["muted"], labelsize=6.5)
        ax_r.grid(color=_PALETTE["border"], alpha=0.4)

        accent = _LAYER_ACCENT[3]
        l3     = viz_data[task].get("layer3")

        # Banner for left panel
        ax_l.add_patch(mpatches.FancyBboxPatch(
            (0.0, 0.91), 1.0, 0.08,
            facecolor=accent, edgecolor="none", alpha=0.18,
            transform=ax_l.transAxes, clip_on=False,
            boxstyle="round,pad=0.01",
        ))
        ax_l.text(0.5, 0.952, task,
                  fontsize=12, color=accent, ha="center", va="center",
                  fontweight="bold", transform=ax_l.transAxes)

        if l3 is None:
            ax_l.text(0.5, 0.55, "ILP_arch pending\n(ILP_arch step not run)",
                      fontsize=9, color=_PALETTE["yellow"], ha="center",
                      va="center", transform=ax_l.transAxes, style="italic",
                      bbox=dict(facecolor=_PALETTE["bg"],
                                edgecolor=_PALETTE["border"],
                                boxstyle="round,pad=0.4"))
            ax_r.set_title("", pad=0)
            continue

        # Learned clause
        clause = l3.get("tau_arch_text", "N/A")
        y      = 0.82
        ax_l.text(0.04, y, "τ_arch clause (Eq. 4)",
                  fontsize=7.5, color=_PALETTE["muted"],
                  transform=ax_l.transAxes, va="top", fontweight="bold")
        y -= 0.07
        clause_fmt = clause.replace(", ", ",\n  ").replace(" :-", " :-\n  ")
        ax_l.text(0.04, y, clause_fmt,
                  fontsize=6.5, color=_PALETTE["text"],
                  transform=ax_l.transAxes, va="top",
                  fontfamily="DejaVu Sans Mono",
                  bbox=dict(facecolor="#0d1117", edgecolor=_PALETTE["border"],
                            boxstyle="round,pad=0.4", alpha=0.6))
        y -= 0.32

        # Structural statistics
        conf = l3.get("tau_arch_confidence", 0.0)
        ax_l.text(0.04, y, f"Confidence_arch: {conf:.3f}",
                  fontsize=8, color=accent, fontweight="bold",
                  transform=ax_l.transAxes, va="top")
        y -= 0.09

        stats = [
            ("Depth mean",        f"{l3.get('depth_mean', 0):.3f}"),
            ("Layer span",        f"[{l3.get('depth_min',0):.2f}, {l3.get('depth_max',0):.2f}]"),
            ("Component ratio",   l3.get("component_ratio", "N/A")),
            ("Attn heads",        str(l3.get("num_attn_heads", "N/A"))),
            ("MLP blocks",        str(l3.get("num_mlp_blocks", "N/A"))),
        ]
        for label, value in stats:
            ax_l.text(0.04, y, label + ":",
                      fontsize=7.5, color=_PALETTE["muted"],
                      transform=ax_l.transAxes, va="top")
            ax_l.text(0.95, y, value,
                      fontsize=7.5, color=_PALETTE["text"], fontweight="bold",
                      ha="right", transform=ax_l.transAxes, va="top")
            y -= 0.07

        # Motifs
        motifs = l3.get("motifs", [])
        if motifs:
            y -= 0.04
            ax_l.text(0.04, y, "Motifs:",
                      fontsize=7.5, color=_PALETTE["muted"],
                      transform=ax_l.transAxes, va="top")
            y -= 0.065
            for mi, m in enumerate(motifs[:4]):
                mx = 0.04 + (mi % 2) * 0.48
                my = y - (mi // 2) * 0.07
                ax_l.add_patch(mpatches.FancyBboxPatch(
                    (mx, my - 0.045), 0.42, 0.050,
                    facecolor=accent, edgecolor="none", alpha=0.22,
                    transform=ax_l.transAxes, clip_on=True,
                    boxstyle="round,pad=0.01"
                ))
                ax_l.text(mx + 0.21, my - 0.018, m.replace("_", " "),
                          fontsize=6.5, color=accent, ha="center", va="center",
                          fontweight="bold", transform=ax_l.transAxes)

        # ── Polar radar chart ──
        n_attn    = l3.get("num_attn_heads",      0)
        n_mlp     = l3.get("num_mlp_blocks",       0)
        d_mean    = l3.get("depth_mean",           0.5)
        d_span    = l3.get("layer_span",           0.5)
        arch_conf = l3.get("tau_arch_confidence",  0.5)

        categories = ["Attn\nratio", "MLP\nratio", "Depth\nmean",
                      "Layer\nspan", "τ_arch\nconf"]
        total = max(n_attn + n_mlp, 1)
        values = [
            n_attn  / total,
            n_mlp   / total,
            d_mean,
            d_span,
            arch_conf,
        ]
        N      = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles      = angles + [angles[0]]

        ax_r.plot(angles, values_plot, color=accent, linewidth=1.5, alpha=0.9)
        ax_r.fill(angles, values_plot, color=accent, alpha=0.18)
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(categories, fontsize=6, color=_PALETTE["muted"])
        ax_r.set_ylim(0, 1)
        ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_r.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=5.5,
                              color=_PALETTE["muted"])
        ax_r.set_title(f"{task}\nprofile", fontsize=8,
                       color=accent, pad=8, fontweight="bold")

    return _save(fig, out_dir, "layer3_architectural")


# ============================================================
# Summary table: full 4-layer comparison
# ============================================================

def plot_summary_table(viz_data: Dict, out_dir: str) -> str:
    """
    Full comparison table across all four layers.
    One row per (layer, field), one column per task.
    """
    tasks   = list(viz_data.keys())
    fig     = plt.figure(figsize=(14, 16))
    fig.patch.set_facecolor(_PALETTE["bg"])
    fig.text(0.04, 0.98, "Registry Entry E — Full 4-Layer Comparison",
             fontsize=14, color=_PALETTE["text"], ha="left", va="top",
             fontweight="bold", transform=fig.transFigure)
    fig.text(0.04, 0.965, "Eq. 7  ·  S_ILP_sem and S_ILP_arch columns populated after ILP runs",
             fontsize=8.5, color=_PALETTE["muted"], ha="left", va="top",
             transform=fig.transFigure)

    ax = fig.add_axes([0.02, 0.02, 0.96, 0.93])
    ax.set_facecolor(_PALETTE["bg"])
    ax.axis("off")

    # Build rows
    def _g(d, *keys, default="N/A"):
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k, None)
            if d is None:
                return default
        return d

    def _fmt(v):
        if v is None or v == "N/A":
            return "N/A"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            return f"{v:.4f}"
        s = str(v)
        return s[:32] + "…" if len(s) > 33 else s

    ROWS = []

    # L0
    for label, key in [
        ("Model",        "model_name"),
        ("Task family",  "task_family"),
        ("Role",         "role"),
        ("Method",       "attribution_method"),
        ("Top-k",        "top_k"),
    ]:
        row = {"layer": 0, "field": label}
        for t in tasks:
            row[t] = _fmt(_g(viz_data[t], "layer0", key))
        ROWS.append(row)

    # L1
    for label, keys in [
        ("Nodes",           ("layer1", "metrics", "num_nodes_in_circuit")),
        ("Edges",           ("layer1", "metrics", "num_edges_in_circuit")),
        ("Faithfulness",    ("layer1", "metrics", "faithfulness")),
        ("Baseline perf.",  ("layer1", "metrics", "baseline_performance")),
        ("Circuit perf.",   ("layer1", "metrics", "circuit_performance")),
        ("ε-minimal",       ("layer1", "validation", "epsilon_minimal")),
        ("p-value",         ("layer1", "validation", "permutation_pvalue")),
        ("Significant",     ("layer1", "validation", "statistically_significant")),
    ]:
        row = {"layer": 1, "field": label}
        for t in tasks:
            row[t] = _fmt(_g(viz_data[t], *keys))
        ROWS.append(row)

    # L2
    for label, keys in [
        ("Specification φ", ("layer2", "spec_text")),
        ("Predicates P_H",  ("layer2", "predicates")),
        ("Confidence_sem",  ("layer2", "confidence")),
        ("n_clauses",       ("layer2", "ilp_signature", "n_clauses")),
        ("avg_depth",       ("layer2", "ilp_signature", "avg_depth")),
        ("complexity",      ("layer2", "ilp_signature", "complexity")),
    ]:
        row = {"layer": 2, "field": label}
        for t in tasks:
            v = _g(viz_data[t], *keys, default="pending ILP_sem")
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            row[t] = _fmt(v)
        ROWS.append(row)

    # L3
    for label, keys in [
        ("Depth mean",      ("layer3", "depth_mean")),
        ("Layer span",      ("layer3", "layer_span")),
        ("Comp. ratio",     ("layer3", "component_ratio")),
        ("Attn heads",      ("layer3", "num_attn_heads")),
        ("MLP blocks",      ("layer3", "num_mlp_blocks")),
        ("Motifs",          ("layer3", "motifs")),
        ("τ_arch conf.",    ("layer3", "tau_arch_confidence")),
    ]:
        row = {"layer": 3, "field": label}
        for t in tasks:
            v = _g(viz_data[t], *keys, default="pending ILP_arch")
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            row[t] = _fmt(v)
        ROWS.append(row)

    # Draw table
    cols   = ["Layer", "Field"] + tasks
    n_rows = len(ROWS)
    n_cols = len(cols)

    col_widths = [0.07, 0.22] + [0.71 / max(len(tasks), 1)] * len(tasks)
    x_starts   = []
    cx = 0.0
    for w in col_widths:
        x_starts.append(cx)
        cx += w

    row_height = 0.93 / (n_rows + 1.5)
    y_top      = 0.93

    # Header
    for ci, (col, xs) in enumerate(zip(cols, x_starts)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (xs + 0.002, y_top - row_height * 0.9),
            col_widths[ci] - 0.004, row_height * 0.85,
            facecolor=_PALETTE["border"], edgecolor="none",
            transform=ax.transAxes, clip_on=True,
            boxstyle="round,pad=0.005"
        ))
        ax.text(xs + col_widths[ci] / 2, y_top - row_height * 0.45,
                col, fontsize=8, color=_PALETTE["text"], fontweight="bold",
                ha="center", va="center", transform=ax.transAxes)

    # Data rows
    prev_layer = -1
    ri = 0
    for row in ROWS:
        layer  = row["layer"]
        y_cell = y_top - (ri + 1.5) * row_height
        accent = _LAYER_ACCENT[layer]

        # Layer separator / background
        if layer != prev_layer:
            # Draw a thin accent line above the first row of each layer
            ax.axhline(y_cell + row_height * 0.95,
                       xmin=0.01, xmax=0.99,
                       color=accent, linewidth=1.2, alpha=0.5)
            prev_layer = layer

        # Alternating row background
        bg = _PALETTE["panel"] if ri % 2 == 0 else "#1a2030"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.005, y_cell - row_height * 0.05),
            0.99, row_height * 0.92,
            facecolor=bg, edgecolor="none",
            transform=ax.transAxes, clip_on=True,
            boxstyle="round,pad=0.002"
        ))

        # Layer badge (only on first row of each layer block)
        if row == ROWS[0] or row["layer"] != ROWS[ROWS.index(row) - 1]["layer"]:
            ax.text(x_starts[0] + col_widths[0] / 2,
                    y_cell + row_height * 0.42,
                    f"L{layer}",
                    fontsize=7.5, color=accent, ha="center", va="center",
                    fontweight="bold", transform=ax.transAxes,
                    bbox=dict(facecolor=accent, edgecolor="none",
                              boxstyle="round,pad=0.2", alpha=0.18))

        # Field name
        ax.text(x_starts[1] + 0.008,
                y_cell + row_height * 0.42,
                row["field"],
                fontsize=7.5, color=_PALETTE["muted"], ha="left", va="center",
                transform=ax.transAxes)

        # Task values
        for ci, task in enumerate(tasks, 2):
            val     = row.get(task, "N/A")
            pending = "pending" in str(val).lower()
            vcolor  = _PALETTE["yellow"] if pending else _PALETTE["text"]
            ax.text(x_starts[ci] + col_widths[ci] / 2,
                    y_cell + row_height * 0.42,
                    val,
                    fontsize=7.0, color=vcolor, ha="center", va="center",
                    transform=ax.transAxes,
                    fontweight="bold" if not pending else "normal",
                    style="italic" if pending else "normal")

        ri += 1

    return _save(fig, out_dir, "summary_table")


# ============================================================
# One-shot convenience function
# ============================================================

def visualise_all(
        ttl_dir:  str = "./data/experiments",
        json_dir: str = "./data/results_json",
        out_dir:  str = "./data/analysis/layers",
        tasks:    Optional[List[str]] = None,
) -> List[str]:
    """
    Load data for all tasks and generate all 5 figures.

    Returns list of written PNG paths.
    """
    if tasks is None:
        tasks = ["Location", "IOI"]

    print(f"\nLoading circuit data...")
    viz_data = {t: load_circuit_data(t, ttl_dir, json_dir) for t in tasks}

    print(f"\nGenerating figures -> {out_dir}/")
    paths = [
        plot_layer0(viz_data, out_dir),
        plot_layer1(viz_data, out_dir),
        plot_layer2(viz_data, out_dir),
        plot_layer3(viz_data, out_dir),
        plot_summary_table(viz_data, out_dir),
    ]
    return paths


# ============================================================
# CLI / standalone
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualise CKG layers")
    parser.add_argument("--ttl-dir",  default="./data/experiments")
    parser.add_argument("--json-dir", default="./data/results_json")
    parser.add_argument("--out-dir",  default="./data/analysis/layers")
    parser.add_argument("--tasks",    nargs="+", default=["Location", "IOI"])
    args = parser.parse_args()

    paths = visualise_all(
        ttl_dir  = args.ttl_dir,
        json_dir = args.json_dir,
        out_dir  = args.out_dir,
        tasks    = args.tasks,
    )
    print(f"\nDone — {len(paths)} figures written.")

