"""
results_export.py
=================
JSON serialisation for MechInt pipeline results.

Imported by run_experiment.py and usable standalone:

    from trace.mechint.storage.results_export import export_task_json, export_comparison_table

Schema design
-------------
Each task produces one JSON file that mirrors the four-layer CKG ontology
from the paper (§3.5, Eq. 7):

    {task}_results.json
    {
      "schema_version": "1.0",
      "timestamp":      "...",
      "meta":    { task, task_family, role, model_name, owl_path },
      "layer0":  { provenance fields },
      "layer1":  { circuit graph, metrics, validation kappa },
      "layer2":  null,   <- populated by add_ilp_semantic()  after ILP_sem
      "layer3":  null,   <- populated by add_ilp_architectural() after ILP_arch
    }

The comparison_table.json has one row per (layer, field) pair and one
column per task, so it is easy to load into pandas or a paper table.

Public API
----------
    export_task_json(task, task_family, role, model_name,
                     results, graph, owl_path, out_dir) -> str

    export_comparison_table(tasks_records, out_dir) -> str

    add_ilp_semantic(json_path, spec_text, predicates,
                     confidence, ilp_signature) -> str

    add_ilp_architectural(json_path, depth_mean, depth_min,
                          depth_max, component_ratio, num_attn,
                          num_mlp, motifs, tau_arch_text,
                          tau_arch_confidence) -> str
"""

import datetime
import json
import os
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    """UTC timestamp string, ISO-8601."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _safe(v: Any) -> Any:
    """
    Recursively convert a value to a JSON-safe primitive.
    Handles: bool, int, float, str, None, torch/numpy scalar, list, dict.
    """
    if v is None:
        return None
    if isinstance(v, bool):           # must come before int (bool IS int)
        return bool(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return float(v)
    if isinstance(v, str):
        return v
    if hasattr(v, "item"):            # torch.Tensor / numpy scalar
        return _safe(v.item())
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _safe(vv) for k, vv in v.items()}
    return str(v)


def _results_to_safe(results: Dict[str, Any]) -> Dict[str, Any]:
    """Convert full results dict to JSON-safe primitives."""
    return {k: _safe(v) for k, v in results.items()}


def _graph_to_dict(graph) -> Dict[str, Any]:
    """
    Serialise a TransformerLens EAP Graph to a JSON-safe dict.

    Returns
    -------
    {
      "num_nodes": int,
      "num_edges": int,
      "n_layers":  int,
      "n_heads":   int,
      "nodes": [
        { "name": str, "layer": int, "type": "attn"|"mlp", "score": float },
        ...                                   # sorted by layer, then name
      ],
      "edges": [
        { "name": str, "source": str, "target": str, "score": float },
        ...                                   # sorted by score desc
      ]
    }
    """
    nodes: List[Dict[str, Any]] = []
    for name, node in graph.nodes.items():
        if not node.in_graph:
            continue
        node_name = getattr(node, "name", name)
        if node_name == "logits":
            continue
        # Naming convention: a{layer}.h{head} = attention, m{layer} = MLP
        ntype = "attn" if (node_name.startswith("a") and ".h" in node_name) else "mlp"
        entry: Dict[str, Any] = {
            "name":  node_name,
            "layer": int(node.layer),
            "type":  ntype,
        }
        if node.score is not None:
            try:
                entry["score"] = float(node.score)
            except (TypeError, ValueError):
                entry["score"] = None
        nodes.append(entry)

    edges: List[Dict[str, Any]] = []
    for edge_name, edge in graph.edges.items():
        if not edge.in_graph:
            continue
        try:
            score = float(edge.score)
        except (TypeError, ValueError):
            score = None
        edges.append({
            "name":   edge_name,
            "source": edge.parent.name,
            "target": edge.child.name,
            "score":  score,
        })

    nodes.sort(key=lambda n: (n["layer"], n["name"]))
    edges.sort(key=lambda e: (-(e["score"] or 0.0)))

    # cfg is a dict-like HookedTransformerConfig or a plain dict
    cfg = graph.cfg
    n_layers = int(cfg.get("n_layers", 0)) if isinstance(cfg, dict) else int(getattr(cfg, "n_layers", 0))
    n_heads  = int(cfg.get("n_heads",  0)) if isinstance(cfg, dict) else int(getattr(cfg, "n_heads",  0))

    return {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "n_layers":  n_layers,
        "n_heads":   n_heads,
        "nodes":     nodes,
        "edges":     edges,
    }


# ---------------------------------------------------------------------------
# Public API — per-task export
# ---------------------------------------------------------------------------

def export_task_json(
    task:             str,
    task_family:      str,
    role:             str,
    model_name:       str,
    results:          Dict[str, Any],
    graph:            Any,
    owl_path:         Optional[str],
    out_dir:          str,
    step:             int = 0,
    operation_type:   Optional[str] = None,   # Gap 1: κ — e.g. "binding", "selection"
    logical_category: Optional[str] = None,   # Gap 2: τ ∈ {PREDICATE, ROLE, OPERATOR}
    filename_override: Optional[str] = None,  # custom filename for multi-split circuits
) -> str:
    """
    Write a single task's complete results to JSON.

    Parameters
    ----------
    task        : short name used as file stem, e.g. "IOI" or "Location"
    task_family : e.g. "IOI" or "SemanticRoles"
    role        : e.g. "IOI" or "Location"
    model_name  : e.g. "pythia-14m"
    results     : dict returned by analyze_and_export()
    graph       : EAP Graph object
    owl_path    : path to the exported TTL file (or None)
    out_dir     : directory to write JSON into
    step        : model checkpoint step (default 0)

    Returns
    -------
    Path to the written JSON file.

    JSON schema (§3.5 registry entry E)
    ------------------------------------
    layer0  — provenance:  model, task, role, method, top_k, owl_path
    layer1  — structural:  circuit graph (nodes + edges), metrics, validation kappa
    layer2  — semantic:    null until add_ilp_semantic() is called
    layer3  — architectural: null until add_ilp_architectural() is called
    """
    safe  = _results_to_safe(results)
    graph_dict = _graph_to_dict(graph)

    # ── Preserve existing layer2/layer3 if the JSON already exists ────────
    # CFS (layer2) and ILP (layer3) are written by separate pipeline steps.
    # Without this, re-running EAP would clobber previously-stored results.
    existing_layer2 = None
    existing_layer3 = None
    _fname = filename_override or f"{task.lower()}_results.json"
    out_path = os.path.join(out_dir, _fname)
    if os.path.isfile(out_path):
        try:
            with open(out_path) as _fh:
                _prev = json.load(_fh)
            existing_layer2 = _prev.get("layer2")
            existing_layer3 = _prev.get("layer3")
        except (json.JSONDecodeError, OSError):
            pass  # corrupt file — start fresh

    record = {
        "schema_version": "1.0",
        "timestamp": _ts(),

        # ── meta ─────────────────────────────────────────────────────────
        # Gap 10: globally unique registry ID per Eq. 7
        "meta": {
            "task":              task,
            "task_family":       task_family,
            "role":              role,
            "model_name":        model_name,
            "owl_path":          owl_path,
            "registry_id":       f"{task.lower()}_{model_name}_{step}_{_ts().replace(':','-')}",
            "operation_type":    operation_type,    # Gap 1: κ (e.g. "binding")
            "logical_category":  logical_category,  # Gap 2: τ (PREDICATE|ROLE|OPERATOR)
        },

        # ── Layer 0: Provenance (§3.1) ────────────────────────────────────
        "layer0": {
            "task_family":       task_family,
            "role":              role,
            "model_name":        model_name,
            "step":              step,
            "attribution_method":safe.get("method"),
            "top_k":             safe.get("top_k"),
            "owl_path":          owl_path,
            "operation_type":    operation_type,    # Gap 1: κ ∈ L(C) = ⟨σ,κ,φ,τ⟩
            "logical_category":  logical_category,  # Gap 2: τ ∈ {PREDICATE, ROLE, OPERATOR}
        },

        # ── Layer 1: Structural (§3.4) ────────────────────────────────────
        "layer1": {

            # Full circuit graph — every in-circuit node and edge
            "circuit": graph_dict,

            # Performance metrics (Eq. 5)
            "metrics": {
                **{k: safe[k]
                   for k in [
                       "faithfulness",
                       "faithfulness_simple",
                       "baseline_performance",
                       "circuit_performance",
                       "ablated_performance",
                       "performance_drop",
                       "performance_drop_pct",
                       "num_nodes_in_circuit",
                       "num_edges_in_circuit",
                       "k_sweep",
                   ]
                   if k in safe},
                # Gap 9: explicit AccT(C,φ) alias per Eq. 5
                "AccT":  safe.get("circuit_performance"),
                # Eq. 6: ΔT(C,φ) = AccT(M,φ) − AccT(M^ablate,C,φ)
                # = baseline_performance − ablated_performance
                # NOTE: performance_drop = baseline − circuit ≠ Eq.6
                "DeltaT": (
                    safe.get("baseline_performance", 0.0) -
                    safe.get("ablated_performance",  0.0)
                ),
            },

            # Validation kappa (Def. 4 — epsilon-minimality + permutation test)
            "validation": {
                "epsilon_minimal":        safe.get("epsilon_minimal"),
                "permutation_pvalue":     safe.get("permutation_pvalue"),
                "statistically_significant": safe.get("statistically_significant"),
            },
        },

        # ── Layer 2: Semantic — preserve existing CFS/ILP_sem data ────────
        "layer2": existing_layer2,

        # ── Layer 3: Architectural — preserve existing ILP_arch data ──────
        "layer3": existing_layer3,
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, _fname)
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)

    return path


# ---------------------------------------------------------------------------
# Public API — comparison table
# ---------------------------------------------------------------------------

def export_comparison_table(
    tasks_records: Dict[str, Dict[str, Any]],
    out_dir: str,
) -> str:
    """
    Write a side-by-side comparison table covering all four layers.

    Parameters
    ----------
    tasks_records : {task_name: json_record_dict, ...}
                    Records should be the dicts loaded from the per-task JSON files.
    out_dir       : directory to write comparison_table.json into

    Returns
    -------
    Path to comparison_table.json

    JSON schema
    -----------
    {
      "schema_version": "1.0",
      "timestamp":      "...",
      "description":    "...",
      "columns":        ["layer", "field", "symbol", "Location", "IOI", ...],
      "table": [
        { "layer": "L0", "field": "Model", "symbol": "M",
          "Location": "pythia-14m", "IOI": "pythia-14m" },
        ...
      ]
    }
    """
    task_names = list(tasks_records.keys())

    def _get(d: Any, *keys: str, default: str = "N/A") -> Any:
        """Safe nested dict access."""
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k)
            if d is None:
                return default
        return d

    def _fmt(v: Any) -> str:
        if v is None or v == "N/A":
            return "N/A"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    rows: List[Dict[str, Any]] = []

    def row(layer: str, field: str, symbol: str, **per_task: Any) -> None:
        rows.append({
            "layer":  layer,
            "field":  field,
            "symbol": symbol,
            **{t: _fmt(per_task.get(t, "N/A")) for t in task_names},
        })

    # ── Layer 0: Provenance ───────────────────────────────────────────────
    row("L0", "Model",             "M",
        **{t: _get(tasks_records[t], "layer0", "model_name")        for t in task_names})
    row("L0", "Task family",       "T",
        **{t: _get(tasks_records[t], "layer0", "task_family")       for t in task_names})
    row("L0", "Role",              "tau (categorical)",
        **{t: _get(tasks_records[t], "layer0", "role")              for t in task_names})
    row("L0", "Attribution method","EAP method",
        **{t: _get(tasks_records[t], "layer0", "attribution_method")for t in task_names})
    row("L0", "Top-k edges",       "top_k",
        **{t: _get(tasks_records[t], "layer0", "top_k")             for t in task_names})

    # ── Layer 1: Structural ───────────────────────────────────────────────
    row("L1", "Nodes in circuit",  "|V_C|",
        **{t: _get(tasks_records[t], "layer1", "metrics", "num_nodes_in_circuit") for t in task_names})
    row("L1", "Edges in circuit",  "|E_C|",
        **{t: _get(tasks_records[t], "layer1", "metrics", "num_edges_in_circuit") for t in task_names})
    row("L1", "AccT(C,phi)",          "Acc_T(C,phi)  Eq.5",
        **{t: _get(tasks_records[t], "layer1", "metrics", "AccT")                  for t in task_names})
    row("L1", "Faithfulness (norm.)", "EAP normalised (not Eq.5)",
        **{t: _get(tasks_records[t], "layer1", "metrics", "faithfulness")          for t in task_names})
    row("L1", "Faithfulness simple","circuit / baseline",
        **{t: _get(tasks_records[t], "layer1", "metrics", "faithfulness_simple")   for t in task_names})
    row("L1", "Baseline perf.",    "full model",
        **{t: _get(tasks_records[t], "layer1", "metrics", "baseline_performance")  for t in task_names})
    row("L1", "Circuit perf.",     "pruned model",
        **{t: _get(tasks_records[t], "layer1", "metrics", "circuit_performance")   for t in task_names})
    row("L1", "Ablated perf.",     "empty circuit",
        **{t: _get(tasks_records[t], "layer1", "metrics", "ablated_performance")   for t in task_names})
    row("L1", "DeltaT(C,phi)",      "Delta_T  Eq.6  baseline-ablated",
        **{t: _get(tasks_records[t], "layer1", "metrics", "DeltaT")             for t in task_names})
    row("L1", "Performance drop",  "baseline - circuit (diagnostic)",
        **{t: _get(tasks_records[t], "layer1", "metrics", "performance_drop")      for t in task_names})
    row("L1", "epsilon-minimal",   "Def. 4",
        **{t: _get(tasks_records[t], "layer1", "validation", "epsilon_minimal")    for t in task_names})
    row("L1", "Permutation p-value","alpha=0.05",
        **{t: _get(tasks_records[t], "layer1", "validation", "permutation_pvalue") for t in task_names})
    row("L1", "Statistically significant","permutation test",
        **{t: _get(tasks_records[t], "layer1", "validation", "statistically_significant") for t in task_names})

    # ── Layer 2: Semantic — pending ILP_sem run ───────────────────────────
    row("L2", "Specification phi", "phi  (ILP_sem)",
        **{t: _get(tasks_records[t], "layer2", "spec_text",  default="pending ILP_sem") for t in task_names})
    row("L2", "Predicates P_H",    "P_H  Eq.3",
        **{t: _get(tasks_records[t], "layer2", "predicates", default="pending ILP_sem") for t in task_names})
    row("L2", "Confidence sem",    "conf_sem  Eq.2",
        **{t: _get(tasks_records[t], "layer2", "confidence", default="pending ILP_sem") for t in task_names})
    row("L2", "n_clauses",         "S_ILP n_clauses",
        **{t: _get(tasks_records[t], "layer2", "ilp_signature", "n_clauses", default="pending ILP_sem") for t in task_names})
    row("L2", "avg_clause_depth",  "S_ILP d-bar  Eq.3",
        **{t: _get(tasks_records[t], "layer2", "ilp_signature", "avg_depth", default="pending ILP_sem") for t in task_names})
    row("L2", "complexity",        "S_ILP complexity",
        **{t: _get(tasks_records[t], "layer2", "ilp_signature", "complexity", default="pending ILP_sem") for t in task_names})

    # ── Layer 3: Architectural — pending ILP_arch run ─────────────────────
    row("L3", "Depth mean",        "d-bar  Sec.3.3",
        **{t: _get(tasks_records[t], "layer3", "depth_mean",       default="pending ILP_arch") for t in task_names})
    row("L3", "Layer span",        "[d_min, d_max]",
        **{t: _get(tasks_records[t], "layer3", "layer_span",       default="pending ILP_arch") for t in task_names})
    row("L3", "Component ratio",   "attn:mlp",
        **{t: _get(tasks_records[t], "layer3", "component_ratio",  default="pending ILP_arch") for t in task_names})
    row("L3", "Num attn heads",    "num_attn",
        **{t: _get(tasks_records[t], "layer3", "num_attn_heads",   default="pending ILP_arch") for t in task_names})
    row("L3", "Num MLP blocks",    "num_mlp",
        **{t: _get(tasks_records[t], "layer3", "num_mlp_blocks",   default="pending ILP_arch") for t in task_names})
    row("L3", "Motifs",            "has_motif  Eq.4",
        **{t: _get(tasks_records[t], "layer3", "motifs",           default="pending ILP_arch") for t in task_names})
    row("L3", "tau_arch confidence","conf_arch",
        **{t: _get(tasks_records[t], "layer3", "tau_arch_confidence", default="pending ILP_arch") for t in task_names})

    doc = {
        "schema_version": "1.0",
        "timestamp":      _ts(),
        "description":    (
            "Side-by-side comparison of all tasks. "
            "Mirrors registry entry E (Sec. 3.5, Eq. 7). "
            "L2/L3 rows show 'pending ILP_sem/arch' until ILP runs are completed."
        ),
        "columns":  ["layer", "field", "symbol"] + task_names,
        "table":    rows,
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "comparison_table.json")
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)

    return path


# ---------------------------------------------------------------------------
# Public API — ILP write-back  (call after ilp_sem / ilp_arch runs)
# ---------------------------------------------------------------------------

def add_ilp_semantic(
    json_path:     str,
    spec_text:     str,
    predicates:    List[str],
    confidence:    float,
    ilp_signature: Dict[str, Any],
    best_lambda:   Optional[float] = None,    # Gap 5: λ for H* from LFI (None if unavailable)
    lfi_weights:   Optional[Dict[str, float]] = None,  # Gap 5: full λ vector (empty if unavailable)
    lfi_available: bool = False,              # True = real EM weights; False = uniform fallback
) -> str:
    """
    Populate layer2 in an existing task JSON after ILP_sem has run.

    Parameters
    ----------
    json_path     : path returned by export_task_json()
    spec_text     : full FOL clause string, e.g. "location(C,E,L) :- ..."
    predicates    : list of predicate names, e.g. ["scaffold_for_role", "token_at"]
    confidence    : float in [0,1]   (Eq. 2)
    ilp_signature : dict with keys:
                      n_clauses, avg_depth, avg_length, complexity
                      (matches S_ILP_sem from Eq. 3)
    best_lambda   : Gap 5 — λ for H* (paper §3.2 H=(L,λ)); None when lfi_available=False
    lfi_weights   : Gap 5 — full dict of LFI-learned weights (empty dict when unavailable)
    lfi_available : True when ProbLog LFI converged; False = fallback uniform weights used

    Returns
    -------
    Path to the updated JSON file (same as json_path).
    """
    with open(json_path) as fh:
        record = json.load(fh)

    # Gap 3: store both Eq.2 pure coverage and adjusted formula
    # ilp_signature may carry confidence_eq2 (pure tp/|Ep|) separately
    _confidence_eq2  = _safe(ilp_signature.get("confidence_eq2",  confidence))
    _confidence_adj  = _safe(ilp_signature.get("confidence_adj",  confidence))

    record["layer2"] = {
        "spec_text":        spec_text,
        "predicates":       predicates,
        "confidence":       _safe(confidence),     # primary (adjusted, for compatibility)
        "confidence_eq2":   _confidence_eq2,       # Gap 3: pure Eq.2 = tp / |Ep|
        "confidence_adj":   _confidence_adj,       # Gap 3: adjusted = coverage×precision×simplicity
        # Gap 5: H = (L, λ) — probabilistic weights learned by LFI EM (paper §3.2)
        "best_lambda":      _safe(best_lambda),         # λ for H* (None if lfi_available=False)
        "lfi_weights":      _safe(lfi_weights or {}),   # full λ vector (empty if unavailable)
        "lfi_available":    bool(lfi_available),        # False = fallback; don't trust lambda
        "ilp_signature": {
            "n_clauses":    _safe(ilp_signature.get("n_clauses")),
            "avg_depth":    _safe(ilp_signature.get("avg_depth")),
            "avg_length":   _safe(ilp_signature.get("avg_length")),
            "complexity":   _safe(ilp_signature.get("complexity")),
            "predicates":   _safe(ilp_signature.get("predicates", [])),
            "lfi_weights":  _safe(ilp_signature.get("lfi_weights", {})),
            "best_lambda":  _safe(ilp_signature.get("best_lambda")),
            "lfi_available": bool(ilp_signature.get("lfi_available", lfi_available)),
        },
        "learned_by":  "ILP_sem",
        "updated_at":  _ts(),
    }
    record["timestamp"] = _ts()

    with open(json_path, "w") as fh:
        json.dump(record, fh, indent=2)

    return json_path


def add_ilp_architectural(
    json_path:            str,
    depth_mean:           float,
    depth_min:            float,
    depth_max:            float,
    component_ratio:      str,
    num_attn_heads:       int,
    num_mlp_blocks:       int,
    motifs:               List[str],
    tau_arch_text:        str,
    tau_arch_confidence:  float,
    ilp_signature:        Optional[Dict[str, Any]] = None,   # Gap 6: S_ILP_arch (Eq. 20)
    exact:                Optional[Dict[str, Any]] = None,   # raw absolute values
    loo_f1:               Optional[float] = None,            # out-of-sample F1
) -> str:
    """
    Populate layer3 in an existing task JSON after ILP_arch has run.

    Parameters
    ----------
    json_path            : path returned by export_task_json()
    depth_mean           : float in [0,1]
    depth_min            : float in [0,1]
    depth_max            : float in [0,1]
    component_ratio      : string e.g. "7:1"
    num_attn_heads       : int
    num_mlp_blocks       : int
    motifs               : list of motif names, e.g. ["TokenCopy", "InductionHead"]
    tau_arch_text        : full learned clause, e.g. "arch_location(C) :- ..."
    tau_arch_confidence  : float in [0,1] (in-sample)
    ilp_signature        : S_ILP_arch dict from ArchitecturalILPLearner (Gap 6 / Eq. 20)
                           Keys: n_clauses, avg_depth, avg_length, complexity, predicates
    loo_f1               : leave-one-out F1 (out-of-sample estimate);
                           None when <3 positive circuits

    Returns
    -------
    Path to the updated JSON file (same as json_path).
    """
    with open(json_path) as fh:
        record = json.load(fh)

    # Build structured S_ILP_arch (Gap 6 — Eq. 20)
    _sig = ilp_signature or {}
    _arch_ilp_sig = {
        "n_clauses":   _safe(_sig.get("n_clauses",   1)),
        "avg_depth":   _safe(_sig.get("avg_depth",   1.0)),
        "avg_length":  _safe(_sig.get("avg_length",  1.0)),
        "complexity":  _safe(_sig.get("complexity",  0.1)),
        "predicates":  _safe(_sig.get("predicates",  [])),
    }

    record["layer3"] = {
        # Relative features — model-size-invariant, used for transfer comparison
        "depth_mean":          _safe(depth_mean),
        "depth_min":           _safe(depth_min),
        "depth_max":           _safe(depth_max),
        "layer_span":          f"[{depth_min:.2f}, {depth_max:.2f}]",
        "component_ratio":     component_ratio,
        "num_attn_heads":      _safe(num_attn_heads),
        "num_mlp_blocks":      _safe(num_mlp_blocks),
        "motifs":              motifs,
        "tau_arch_text":       tau_arch_text,
        "tau_arch_confidence": _safe(tau_arch_confidence),
        "loo_f1":              _safe(loo_f1) if loo_f1 is not None else None,
        "ilp_signature":       _arch_ilp_sig,           # Gap 6 ✓
        # Exact / absolute values — preserved for inspection, NOT used in transfer
        "exact":               exact or {},
        "learned_by":          "ILP_arch",
        "updated_at":          _ts(),
    }
    record["timestamp"] = _ts()

    with open(json_path, "w") as fh:
        json.dump(record, fh, indent=2)

    return json_path
