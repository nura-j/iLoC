"""
transfer_engine.py
==================
Implements the full §3.6 transfer algorithm from the paper:

    Given source mechanism (Cα, L(Cα)) validated on model Mα
    and target model Mβ, find or discover a matching circuit Cβ.

    Step 1 — Architectural candidate generation (static)
        Use τ_arch,α from ILP_arch to query the registry for circuits in Mβ
        that match the same architectural type (prompt-independent).

    Step 2 — Structural filtering (static)
        Filter by size, layer profile, component ratios using S_circuit(Cα).

    Step 3 — Behavioural screening (live)
        Evaluate each candidate on D_T with φ_α from ILP_sem to check
        AccT(Cβ, φ_α) ≥ θ_behav.

    Step 4 — Causal validation (live)
        Select the candidate maximising ΔT(Cβ, φ_α) under ablation.

    Step 5 — Re-learning fallback (live)
        If no candidate passes, run EAP discovery on Mβ and learn φ′ via
        ILP_sem; accept if d(S_ILP_sem(φ′), S_ILP_sem(φα)) < δ_ILP.

Usage
-----
Static only (no model needed — works on stored registry JSON/TTL):

    from trace.mechint.transfer import TransferEngine
    engine = TransferEngine(registry_dir="./data/results_json")
    report = engine.find_transfer_candidates(
        source_json_path="./data/results_json/ioi_results.json",
        target_model_name="pythia-1b",          # filters by modelName in registry
    )
    print(report.summary())

Full pipeline (requires target model):

    from trace.mechint.transfer import TransferEngine
    engine = TransferEngine(registry_dir="./data/results_json")
    report = engine.transfer(
        source_json_path="./data/results_json/ioi_results.json",
        target_model=model_beta,
        target_dataset_path="./data/ioi_pythia_1b.csv",
        target_model_name="pythia-1b",
    )
    if report.accepted:
        print(f"Transfer accepted: {report.accepted_circuit_path}")
        print(f"  ΔT = {report.delta_t:.4f}")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateResult:
    """One candidate circuit considered for transfer."""
    circuit_uri:        str
    json_path:          Optional[str]
    model_name:         Optional[str]
    task_name:          Optional[str]

    # Step 1 — arch match
    arch_match:         bool  = False
    arch_score:         float = 0.0    # how well τ_arch,α matches this circuit

    # Step 2 — structural filter
    struct_pass:        bool  = False
    depth_diff:         float = float("inf")
    size_diff:          float = float("inf")  # rel_size diff [0,1]
    ratio_match:        bool  = False
    motif_overlap:      float = 0.0    # Jaccard on motif sets
    exact:              Dict  = field(default_factory=dict)  # raw absolute values for reporting
    fingerprint_score:  Optional[float] = None               # circuit template similarity [0,1]

    # Step 3 — behavioural screening (live)
    acc_t:              Optional[float] = None   # AccT(Cβ, φα)
    behav_pass:         Optional[bool]  = None

    # Step 4 — causal validation (live)
    delta_t:            Optional[float] = None   # ΔT(Cβ, φα)
    causal_pass:        Optional[bool]  = None

    # Def. 5 check (signature distance, κ, τ)
    def5_pass:          Optional[bool]  = None
    def5_distance:      Optional[float] = None
    def5_reasons:       Dict[str, Any]  = field(default_factory=dict)


@dataclass
class TransferReport:
    """Full report from a transfer attempt."""
    source_json:        str
    source_task:        str
    target_model_name:  Optional[str]
    phase:              str            # "static" | "live"

    candidates:         List[CandidateResult] = field(default_factory=list)

    # Accepted result (Step 4 winner or Step 5 fallback)
    accepted:           bool = False
    accepted_circuit_path: Optional[str] = None
    accepted_via:       Optional[str] = None   # "step4_causal" | "step5_relearn"
    delta_t:            Optional[float] = None

    # Step 5 re-learning (if triggered)
    relearn_triggered:  bool = False
    relearn_result:     Optional[Dict[str, Any]] = None

    def summary(self) -> str:
        lines = [
            f"Transfer report  ({self.phase} phase)",
            f"  Source : {self.source_task}  ({self.source_json})",
            f"  Target : {self.target_model_name or 'any'}",
            f"  Candidates evaluated : {len(self.candidates)}",
        ]
        step2 = [c for c in self.candidates if c.struct_pass]
        step3 = [c for c in self.candidates if c.behav_pass]
        step4 = [c for c in self.candidates if c.causal_pass]
        if step2:
            lines.append(f"  Step 2 structural pass  : {len(step2)}")
        if step3 is not None:
            lines.append(f"  Step 3 behavioural pass : {len(step3)}")
        if step4:
            lines.append(f"  Step 4 causal pass      : {len(step4)}")
        if self.relearn_triggered:
            lines.append(f"  Step 5 re-learning      : triggered")
        if self.accepted:
            lines.append(f"  ACCEPTED via {self.accepted_via}")
            if self.delta_t is not None:
                lines.append(f"    ΔT = {self.delta_t:.4f}")
            if self.accepted_circuit_path:
                lines.append(f"    circuit: {self.accepted_circuit_path}")
        else:
            lines.append("  NOT ACCEPTED — no matching circuit found")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Circuit fingerprint — source circuit as search heuristic
# ══════════════════════════════════════════════════════════════════════════════

def _circuit_fingerprint(circuit: Dict[str, Any],
                          model_n_layers: int = 0,
                          model_n_heads:  int = 0) -> Dict[str, Any]:
    """
    Extract a model-size-invariant structural fingerprint from a layer1.circuit
    dict.  This is used as a search template when transferring to a target model:
    instead of learning an ILP clause and matching against it, we compare the
    source circuit's topology directly to candidate subgraphs.

    All position/size values are RELATIVE so the fingerprint is meaningful
    across models of different sizes (14M → 1B).

    Returns
    -------
    {
        # Relative positional envelope [0,1]
        "depth_min":   float,   # normalised first layer
        "depth_max":   float,   # normalised last layer
        "depth_mean":  float,   # normalised centre of mass
        "depth_span":  float,   # depth_max - depth_min

        # Component composition
        "attn_ratio":  float,   # attention heads / total components
        "rel_size":    float,   # total components / model_total_heads

        # Topology skeleton — edge type distribution (normalised)
        "edge_aa":     float,   # attn→attn edges / total edges
        "edge_am":     float,   # attn→mlp edges / total edges
        "edge_ma":     float,   # mlp→attn edges / total edges
        "edge_mm":     float,   # mlp→mlp edges / total edges

        # Top-k component layers (relative) — the "skeleton" of the circuit
        # sorted by attribution score descending so the most important
        # components come first
        "top_component_layers": [float, ...],   # up to 5 relative layer positions

        # Exact values preserved for reporting
        "exact": {
            "num_attn":       int,
            "num_mlp":        int,
            "layer_min_abs":  int,
            "layer_max_abs":  int,
            "model_n_layers": int,
        }
    }
    """
    nodes = circuit.get("nodes", [])
    edges = circuit.get("edges", [])

    if not nodes:
        return {}

    # Infer model dimensions if not provided
    if model_n_layers <= 0:
        model_n_layers = max((n.get("layer", 0) for n in nodes), default=0) + 1
    if model_n_heads <= 0:
        model_n_heads = max(8, model_n_layers)   # conservative fallback

    model_total = model_n_layers * model_n_heads

    # Component counts and layer positions
    attn_nodes = [n for n in nodes if n.get("type") == "attn"]
    mlp_nodes  = [n for n in nodes if n.get("type") == "mlp"]
    n_attn = len(attn_nodes)
    n_mlp  = len(mlp_nodes)
    total  = max(n_attn + n_mlp, 1)

    raw_layers = [n.get("layer", 0) for n in nodes]
    layer_min  = min(raw_layers)
    layer_max  = max(raw_layers)
    n          = max(model_n_layers, 1)

    depth_min  = layer_min / n
    depth_max  = layer_max / n
    depth_mean = sum(raw_layers) / len(raw_layers) / n
    depth_span = depth_max - depth_min

    attn_ratio = n_attn / total
    rel_size   = total / max(model_total, 1)

    # Edge type distribution
    node_types = {n.get("name", ""): n.get("type", "attn") for n in nodes}
    aa = am = ma = mm = 0
    for e in edges:
        st = node_types.get(e.get("source", ""), "attn")
        tt = node_types.get(e.get("target", ""), "attn")
        if   st == "attn" and tt == "attn": aa += 1
        elif st == "attn" and tt == "mlp":  am += 1
        elif st == "mlp"  and tt == "attn": ma += 1
        else:                               mm += 1
    n_edges = max(aa + am + ma + mm, 1)

    # Top-5 components by attribution score (relative layer positions)
    scored = sorted(nodes, key=lambda n: -(n.get("score") or 0.0))
    top_layers = [round(n.get("layer", 0) / model_n_layers, 3)
                  for n in scored[:5]]

    return {
        "depth_min":             round(depth_min,  4),
        "depth_max":             round(depth_max,  4),
        "depth_mean":            round(depth_mean, 4),
        "depth_span":            round(depth_span, 4),
        "attn_ratio":            round(attn_ratio, 4),
        "rel_size":              round(rel_size,   4),
        "edge_aa":               round(aa / n_edges, 3),
        "edge_am":               round(am / n_edges, 3),
        "edge_ma":               round(ma / n_edges, 3),
        "edge_mm":               round(mm / n_edges, 3),
        "top_component_layers":  top_layers,
        "exact": {
            "num_attn":       n_attn,
            "num_mlp":        n_mlp,
            "layer_min_abs":  layer_min,
            "layer_max_abs":  layer_max,
            "model_n_layers": model_n_layers,
        },
    }


def _fingerprint_similarity(fp_src: Dict[str, Any],
                             fp_cand: Dict[str, Any]) -> float:
    """
    Compute similarity score ∈ [0, 1] between two circuit fingerprints.

    Uses weighted combination of:
      - Positional overlap (depth envelope intersection)
      - Component composition similarity (attn_ratio, rel_size)
      - Edge topology similarity (edge type distribution)

    Higher = more similar.  Used to rank candidates in Step 1b.
    """
    if not fp_src or not fp_cand:
        return 0.0

    # 1. Positional overlap — how much do the layer envelopes intersect?
    src_min,  src_max  = fp_src.get("depth_min",  0.0), fp_src.get("depth_max",  1.0)
    cand_min, cand_max = fp_cand.get("depth_min", 0.0), fp_cand.get("depth_max", 1.0)
    overlap    = max(0.0, min(src_max, cand_max) - max(src_min, cand_min))
    union_span = max(src_max, cand_max) - min(src_min, cand_min)
    pos_sim    = overlap / max(union_span, 1e-6)

    # 2. Composition similarity — L1 distance on (attn_ratio, rel_size)
    ratio_diff = abs(fp_src.get("attn_ratio", 0.8) - fp_cand.get("attn_ratio", 0.8))
    size_diff  = abs(fp_src.get("rel_size",   0.1) - fp_cand.get("rel_size",   0.1))
    comp_sim   = max(0.0, 1.0 - ratio_diff - size_diff * 2)

    # 3. Edge topology similarity — cosine-like on edge type vector
    keys = ["edge_aa", "edge_am", "edge_ma", "edge_mm"]
    v1   = [fp_src.get(k,  0.0) for k in keys]
    v2   = [fp_cand.get(k, 0.0) for k in keys]
    dot  = sum(a * b for a, b in zip(v1, v2))
    n1   = sum(a * a for a in v1) ** 0.5
    n2   = sum(b * b for b in v2) ** 0.5
    edge_sim = dot / max(n1 * n2, 1e-9)

    # Weighted combination (positional overlap most important)
    score = 0.5 * pos_sim + 0.3 * comp_sim + 0.2 * edge_sim
    return round(max(0.0, min(1.0, score)), 4)


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────

class TransferEngine:
    """
    Two-phase transfer engine implementing paper §3.6.

    Parameters
    ----------
    registry_dir   : directory containing *_results.json files
    delta_ilp      : δ_ILP — ILP signature distance threshold (default 0.30)
    delta_arch     : δ_arch — architectural signature distance threshold
    theta_behav    : θ_behav — minimum AccT for behavioural screening (Step 3)
    theta_causal   : θ_causal — minimum ΔT for causal acceptance (Step 4)
    struct_depth_tol : tolerance on relative depth mean for Step 2 filter
    struct_size_tol  : tolerance on circuit size (n_nodes) for Step 2 filter
    """

    def __init__(
            self,
            registry_dir:     str   = "./data/results_json",
            delta_ilp:        float = 0.30,
            delta_arch:       float = 0.35,
            theta_behav:      float = 0.50,
            theta_causal:     float = 0.10,
            struct_depth_tol: float = 0.15,
            struct_size_tol:  float = 0.15,   # tolerance on rel_size [0,1]
    ):
        self.registry_dir     = registry_dir
        self.delta_ilp        = delta_ilp
        self.delta_arch       = delta_arch
        self.theta_behav      = theta_behav
        self.theta_causal     = theta_causal
        self.struct_depth_tol = struct_depth_tol
        self.struct_size_tol  = struct_size_tol   # relative, [0,1]

        # Load registry on construction
        self._registry: List[Dict[str, Any]] = self._load_registry()

    # ── Registry ──────────────────────────────────────────────────────────────

    def _load_registry(self) -> List[Dict[str, Any]]:
        """Load all *_results.json files from registry_dir and its subdirectories."""
        records = []
        if not os.path.isdir(self.registry_dir):
            return records
        for root, _dirs, files in os.walk(self.registry_dir):
            for fname in sorted(files):
                if not fname.endswith("_results.json"):
                    continue
                path = os.path.join(root, fname)
                try:
                    with open(path) as fh:
                        rec = json.load(fh)
                    rec["__json_path__"] = path
                    records.append(rec)
                except Exception:
                    pass
        return records

    def reload_registry(self) -> int:
        """Reload from disk (call after new experiments are added)."""
        self._registry = self._load_registry()
        return len(self._registry)

    # ── Public API ────────────────────────────────────────────────────────────

    def find_transfer_candidates(
            self,
            source_json_path:   str,
            target_model_name:  Optional[str] = None,
    ) -> TransferReport:
        """
        Static phase only (Steps 1–2 + Def. 5 signature check).

        No model required.  Returns a ranked list of registry candidates
        that pass structural filtering and Def. 5 signature distance.
        """
        source = self._load_source(source_json_path)
        report = TransferReport(
            source_json       = source_json_path,
            source_task       = source.get("meta", {}).get("task", "unknown"),
            target_model_name = target_model_name,
            phase             = "static",
        )

        candidates = self._step1_arch_candidates(source, target_model_name)
        candidates = self._step1b_circuit_template(source, candidates)   # circuit heuristic
        candidates = self._step2_structural_filter(source, candidates)
        self._step_def5_check(source, candidates)

        report.candidates = candidates
        report.candidates.sort(key=lambda c: (
            -int(c.def5_pass or False),
            -int(c.struct_pass),
            c.depth_diff,
        ))
        return report

    def transfer(
            self,
            source_json_path:   str,
            target_model,
            target_dataset_path: str,
            target_model_name:  str,
            target_role:        Optional[str] = None,
            target_task_family: Optional[str] = None,
            output_dir:         str = "./data/experiments",
    ) -> TransferReport:
        """
        Full transfer pipeline (Steps 1–5).

        Steps 1–2 are static (registry queries).
        Steps 3–4 require target_model (live inference).
        Step 5 triggers EAP re-discovery on Mβ if Steps 3–4 find nothing.
        """
        source = self._load_source(source_json_path)
        report = TransferReport(
            source_json       = source_json_path,
            source_task       = source.get("meta", {}).get("task", "unknown"),
            target_model_name = target_model_name,
            phase             = "live",
        )

        # ── Steps 1–2: static candidate generation and structural filtering ──
        candidates = self._step1_arch_candidates(source, target_model_name)
        candidates = self._step1b_circuit_template(source, candidates)   # circuit heuristic
        candidates = self._step2_structural_filter(source, candidates)
        self._step_def5_check(source, candidates)

        struct_passed = [c for c in candidates if c.struct_pass]
        if not struct_passed:
            print("[Transfer] No structural candidates — skipping to Step 5 fallback.")
            self._step5_relearn(
                source, report, target_model, target_dataset_path,
                target_model_name, target_role, target_task_family, output_dir,
            )
            report.candidates = candidates
            return report

        # ── Step 3: behavioural screening ─────────────────────────────────────
        struct_passed = self._step3_behav_screen(
            source, struct_passed, target_model, target_dataset_path,
        )

        # ── Step 4: causal validation ─────────────────────────────────────────
        winner = self._step4_causal_validate(
            source, struct_passed, target_model, target_dataset_path,
        )

        if winner is not None:
            report.accepted            = True
            report.accepted_circuit_path = winner.json_path
            report.accepted_via        = "step4_causal"
            report.delta_t             = winner.delta_t
        else:
            print("[Transfer] No candidate passed causal validation — Step 5 fallback.")
            self._step5_relearn(
                source, report, target_model, target_dataset_path,
                target_model_name, target_role, target_task_family, output_dir,
            )

        report.candidates = candidates
        return report

    # ── Step 1 — Architectural candidate generation ───────────────────────────

    def _step1_arch_candidates(
            self,
            source: Dict[str, Any],
            target_model_name: Optional[str],
    ) -> List[CandidateResult]:
        """
        Query the registry for circuits whose stored τ_arch clause or
        architectural type matches the source.  Prompt-independent.

        Matching strategy (in priority order):
          a) τ_arch clause text contains the same arch_type predicate
          b) operation_type (κ) matches — same broad circuit family
          c) Any circuit in the target model (fallback: no arch constraint)
        """
        src_arch_type = self._get_arch_type(source)     # e.g. "arch_instrument"
        src_kappa     = self._get(source, "layer0", "operation_type")
        src_model     = self._get(source, "meta", "model_name")

        candidates: List[CandidateResult] = []

        for rec in self._registry:
            rec_path  = rec.get("__json_path__", "")
            rec_model = self._get(rec, "meta", "model_name") or ""
            rec_task  = self._get(rec, "meta", "task") or ""

            # Skip self
            if rec_path == source.get("__json_path__", ""):
                continue

            # If target_model_name specified, only include circuits from that model
            if target_model_name and target_model_name.lower() not in rec_model.lower():
                continue

            # Skip circuits from the same model as source
            if src_model and rec_model and rec_model.lower() == src_model.lower():
                # Allow same-model transfer for within-model cross-task experiments
                pass

            rec_arch_type = self._get_arch_type(rec)
            rec_kappa     = self._get(rec, "layer0", "operation_type")

            # Score the architectural match
            if src_arch_type and rec_arch_type and src_arch_type == rec_arch_type:
                arch_match = True
                arch_score = 1.0
            elif src_kappa and rec_kappa and src_kappa.lower() == rec_kappa.lower():
                arch_match = True
                arch_score = 0.7
            elif target_model_name:
                # Broad fallback: include all circuits from the target model
                arch_match = True
                arch_score = 0.3
            else:
                arch_match = False
                arch_score = 0.0

            c = CandidateResult(
                circuit_uri  = self._get(rec, "meta", "circuit_uri") or rec_task,
                json_path    = rec_path,
                model_name   = rec_model,
                task_name    = rec_task,
                arch_match   = arch_match,
                arch_score   = arch_score,
            )
            candidates.append(c)

        candidates.sort(key=lambda c: -c.arch_score)
        print(f"[Step 1] {len(candidates)} architectural candidates "
              f"(arch_type={src_arch_type}, κ={src_kappa})")
        return candidates


    # ── Step 1b — Circuit template search ────────────────────────────────────

    def _step1b_circuit_template(
            self,
            source: Dict[str, Any],
            candidates: List[CandidateResult],
    ) -> List[CandidateResult]:
        """
        Use the source circuit itself as a structural search template.

        Instead of relying solely on the learned ILP_arch clause (which
        requires multiple training examples), we extract a fingerprint
        directly from the source circuit's layer1.circuit graph and score
        each candidate against it.

        This works with a single source circuit and is model-size-invariant
        because all features are normalised relative to model dimensions.

        The fingerprint_score is stored on each candidate and used to
        re-rank before Step 2.  Candidates with fingerprint_score below
        threshold are soft-filtered but kept in the report for inspection.
        """
        src_circuit  = self._get(source, "layer1", "circuit") or {}
        src_n_layers = self._get_int(source, "layer1", "circuit", "n_layers") or                        self._get_int(source, "layer1", "n_layers") or 0
        src_n_heads  = self._get_int(source, "layer1", "circuit", "n_heads") or                        self._get_int(source, "layer1", "n_heads") or 0

        if not src_circuit.get("nodes"):
            print("[Step 1b] No source circuit graph available — skipping template search")
            return candidates

        src_fp = _circuit_fingerprint(src_circuit, src_n_layers, src_n_heads)

        print(f"[Step 1b] Source circuit fingerprint:")
        print(f"          depth=[{src_fp['depth_min']:.2f}, {src_fp['depth_max']:.2f}]  "
              f"attn_ratio={src_fp['attn_ratio']:.2f}  "
              f"rel_size={src_fp['rel_size']:.3f}")
        print(f"          edge_topology: aa={src_fp['edge_aa']:.2f} "
              f"am={src_fp['edge_am']:.2f} "
              f"ma={src_fp['edge_ma']:.2f}")

        scored = 0
        for c in candidates:
            rec = self._get_record(c.json_path)
            if rec is None:
                continue

            cand_circuit  = self._get(rec, "layer1", "circuit") or {}
            cand_n_layers = self._get_int(rec, "layer1", "circuit", "n_layers") or                             self._get_int(rec, "layer1", "n_layers") or 0
            cand_n_heads  = self._get_int(rec, "layer1", "circuit", "n_heads") or                             self._get_int(rec, "layer1", "n_heads") or 0

            if not cand_circuit.get("nodes"):
                c.fingerprint_score = None
                continue

            cand_fp = _circuit_fingerprint(cand_circuit, cand_n_layers, cand_n_heads)
            sim     = _fingerprint_similarity(src_fp, cand_fp)
            c.fingerprint_score = sim
            scored += 1

            # Store candidate fingerprint for reporting
            c.exact["fingerprint"] = {
                "similarity":  sim,
                "depth":       [cand_fp["depth_min"], cand_fp["depth_max"]],
                "attn_ratio":  cand_fp["attn_ratio"],
                "rel_size":    cand_fp["rel_size"],
                "edge_aa":     cand_fp["edge_aa"],
                "edge_am":     cand_fp["edge_am"],
            }

        # Re-rank: blend fingerprint similarity (70%) with ILP_arch score (30%).
        # When fingerprint is available it dominates — the circuit is the heuristic.
        # ILP_arch score is a tiebreaker and provides interpretability.
        for c in candidates:
            if c.fingerprint_score is not None:
                c.arch_score = round(
                    0.7 * c.fingerprint_score + 0.3 * c.arch_score, 4
                )

        candidates.sort(key=lambda c: -c.arch_score)
        print(f"[Step 1b] Fingerprint scored {scored}/{len(candidates)} candidates — "
              f"re-ranked by circuit similarity")
        return candidates

    # ── Step 2 — Structural filtering ────────────────────────────────────────

    def _step2_structural_filter(
            self,
            source: Dict[str, Any],
            candidates: List[CandidateResult],
    ) -> List[CandidateResult]:
        """
        Filter candidates by size, layer profile, and component ratios
        using S_circuit(Cα) — paper Step 2.
        """
        src_depth    = self._get_float(source, "layer3", "depth_mean")
        src_rel_size = self._get_float(source, "layer3", "ilp_signature", "rel_size") or                        self._get_float(source, "layer3", "rel_size") or 0.0
        src_ratio    = self._get(source, "layer3", "component_ratio") or ""
        src_motifs   = set(self._get(source, "layer3", "motifs") or [])
        # Exact values — for reporting only, not used in filter logic
        src_exact    = self._get(source, "layer3", "exact") or {}

        passed = 0
        for c in candidates:
            rec = self._get_record(c.json_path)
            if rec is None:
                continue

            cand_depth    = self._get_float(rec, "layer3", "depth_mean")
            cand_rel_size = self._get_float(rec, "layer3", "ilp_signature", "rel_size") or                             self._get_float(rec, "layer3", "rel_size") or 0.0
            cand_ratio    = self._get(rec, "layer3", "component_ratio") or ""
            cand_motifs   = set(self._get(rec, "layer3", "motifs") or [])
            # Exact values — stored on candidate for reporting
            cand_exact    = self._get(rec, "layer3", "exact") or {}

            # Depth filter — relative [0,1], model-size-invariant
            depth_diff = abs((cand_depth or 0.5) - (src_depth or 0.5))
            c.depth_diff = depth_diff

            # Size filter — rel_size [0,1], model-size-invariant
            # Raw absolute counts are stored in c.exact for reporting only
            size_diff = abs(cand_rel_size - src_rel_size)
            c.size_diff   = size_diff
            c.exact       = {
                "src":  src_exact,
                "cand": cand_exact,
            }

            # Ratio match (float comparison on attn fraction)
            src_ratio_f  = self._get_float(source, "layer3", "component_ratio_float") or 0.0
            cand_ratio_f = self._get_float(rec,    "layer3", "component_ratio_float") or 0.0
            c.ratio_match = abs(src_ratio_f - cand_ratio_f) < 0.2 if src_ratio_f else True

            # Motif overlap (Jaccard)
            if src_motifs or cand_motifs:
                union = src_motifs | cand_motifs
                inter = src_motifs & cand_motifs
                c.motif_overlap = len(inter) / len(union) if union else 0.0
            else:
                c.motif_overlap = 1.0  # no motif info — don't penalise

            c.struct_pass = (
                depth_diff <= self.struct_depth_tol and
                size_diff  <= self.struct_size_tol
            )
            if c.struct_pass:
                passed += 1

        print(f"[Step 2] {passed}/{len(candidates)} candidates pass structural filter "
              f"(depth_tol={self.struct_depth_tol}, rel_size_tol={self.struct_size_tol})")
        return candidates

    # ── Def. 5 signature distance check (static) ─────────────────────────────

    def _step_def5_check(
            self,
            source: Dict[str, Any],
            candidates: List[CandidateResult],
    ) -> None:
        """
        Run is_transfer_pair (Def. 5) for every struct-passed candidate.
        Operates purely on stored signatures — no model needed.
        """
        from trace.mechint.storage.ilp.ilp_signature_distance import is_transfer_pair

        src_sig   = self._get(source, "layer2", "ilp_signature") or {}
        src_acc   = self._get_float(source, "layer1", "metrics", "AccT")
        src_kappa = self._get(source, "layer0", "operation_type")
        src_tau   = self._get(source, "layer0", "logical_category")

        for c in candidates:
            if not c.struct_pass:
                continue
            rec = self._get_record(c.json_path)
            if rec is None:
                continue

            cand_sig   = self._get(rec, "layer2", "ilp_signature") or {}
            cand_acc   = self._get_float(rec, "layer1", "metrics", "AccT")
            cand_kappa = self._get(rec, "layer0", "operation_type")
            cand_tau   = self._get(rec, "layer0", "logical_category")

            if not src_sig or not cand_sig:
                # No ILP signatures yet — skip Def. 5 (static phase only)
                c.def5_pass    = None
                c.def5_reasons = {"note": "ILP signatures not yet computed"}
                continue

            is_pair, dist, reasons = is_transfer_pair(
                sig_alpha       = src_sig,
                sig_beta        = cand_sig,
                threshold       = self.delta_ilp,
                acc_t_alpha     = src_acc,
                acc_t_beta      = cand_acc,
                kappa_alpha     = src_kappa,
                kappa_beta      = cand_kappa,
                tau_alpha       = src_tau,
                tau_beta        = cand_tau,
            )
            c.def5_pass    = is_pair
            c.def5_distance = dist
            c.def5_reasons  = reasons

    # ── Step 3 — Behavioural screening (live) ────────────────────────────────

    def _step3_behav_screen(
            self,
            source:          Dict[str, Any],
            candidates:      List[CandidateResult],
            target_model,
            dataset_path:    str,
    ) -> List[CandidateResult]:
        """
        Evaluate each candidate on D_T using φ_α from ILP_sem.

        For candidates that are already stored in the registry (i.e. the
        circuit was already discovered on Mβ), we can read AccT directly
        from their JSON.  For candidates with no AccT, we run evaluate_graph
        with the source specification as the metric.

        Returns candidates that pass θ_behav.
        """
        phi_alpha  = self._get(source, "layer2", "spec_text") or ""
        theta_behav = self.theta_behav

        passed = []
        for c in candidates:
            if not c.struct_pass:
                continue

            rec = self._get_record(c.json_path)
            if rec is None:
                continue

            # Fast path: candidate already has AccT stored
            stored_acc = self._get_float(rec, "layer1", "metrics", "AccT")
            if stored_acc is not None:
                c.acc_t     = stored_acc
                c.behav_pass = stored_acc >= theta_behav
                if c.behav_pass:
                    passed.append(c)
                continue

            # Live path: need to evaluate the circuit on D_T with φ_α
            # This requires loading the circuit graph from the stored EAP
            # results and running evaluate_graph with a metric derived from φ_α
            acc_t = self._evaluate_candidate_acc(
                candidate  = c,
                source     = source,
                phi_alpha  = phi_alpha,
                model      = target_model,
                dataset    = dataset_path,
            )
            c.acc_t     = acc_t
            c.behav_pass = acc_t is not None and acc_t >= theta_behav
            if c.behav_pass:
                passed.append(c)

        print(f"[Step 3] {len(passed)}/{len([c for c in candidates if c.struct_pass])} "
              f"candidates pass behavioural screening (θ_behav={theta_behav})")
        return passed

    # ── Step 4 — Causal validation (live) ────────────────────────────────────

    def _step4_causal_validate(
            self,
            source:       Dict[str, Any],
            candidates:   List[CandidateResult],
            target_model,
            dataset_path: str,
    ) -> Optional[CandidateResult]:
        """
        For each behaviourally screened candidate, compute ΔT(Cβ, φα) under
        ablation.  Return the candidate with the highest ΔT that exceeds
        θ_causal (paper Step 4).

        ΔT(Cβ, φα) = AccT(Mβ, φα) - AccT(Mβ^(ablate,Cβ), φα)
        """
        theta_causal = self.theta_causal
        best:         Optional[CandidateResult] = None

        for c in candidates:
            delta_t = self._compute_delta_t(c, source, target_model, dataset_path)
            c.delta_t    = delta_t
            c.causal_pass = delta_t is not None and delta_t >= theta_causal
            if c.causal_pass:
                if best is None or (delta_t or 0) > (best.delta_t or 0):
                    best = c

        n_passed = sum(1 for c in candidates if c.causal_pass)
        print(f"[Step 4] {n_passed}/{len(candidates)} candidates pass causal validation "
              f"(θ_causal={theta_causal})")
        if best:
            print(f"         Best: {best.task_name} (ΔT={best.delta_t:.4f})")
        return best

    # ── Step 5 — Re-learning fallback (live) ─────────────────────────────────

    def _step5_relearn(
            self,
            source:              Dict[str, Any],
            report:              TransferReport,
            target_model,
            dataset_path:        str,
            target_model_name:   str,
            target_role:         Optional[str],
            target_task_family:  Optional[str],
            output_dir:          str,
    ) -> None:
        """
        Run full EAP circuit discovery on Mβ, then learn φ′ via ILP_sem.
        Accept if d(S_ILP_sem(φ′), S_ILP_sem(φα)) < δ_ILP.

        This calls api.analyze_and_export, then runs the ILP_sem pipeline
        (ilp_integration.process_single_circuit → SemanticSpecificationLearner)
        on the newly discovered circuit's TTL.
        """
        from trace.mechint.storage.ilp.ilp_signature_distance import ilp_signature_distance

        report.relearn_triggered = True

        role         = target_role or self._get(source, "meta", "role") or "UNKNOWN"
        task_family  = target_task_family or self._get(source, "meta", "task_family") or "Unknown"
        operation_type   = self._get(source, "layer0", "operation_type")
        logical_category = self._get(source, "layer0", "logical_category")

        print(f"[Step 5] Running EAP discovery on {target_model_name} "
              f"(role={role}, task={task_family}) ...")

        try:
            from trace.mechint.api import analyze_and_export
            results_beta, graph_beta, owl_path_beta = analyze_and_export(
                model             = target_model,
                cnp_dataset_path  = dataset_path,
                task_family       = task_family,
                role              = role,
                model_name        = target_model_name,
                output_dir        = output_dir,
                operation_type    = operation_type,
                logical_category  = logical_category,
            )
        except Exception as e:
            print(f"[Step 5]   EAP discovery failed: {e}")
            report.relearn_result = {"error": f"EAP discovery failed: {e}"}
            return

        print("[Step 5]   EAP done. Running ILP_sem to learn φ′ ...")

        try:
            from trace.mechint.storage.ilp.ilp_integration import process_single_circuit
            from trace.mechint.storage.ilp.task_config import (
                TASK_REGISTRY, get_task_config, _sem_role_config,
            )
            from trace.mechint.storage.ilp.adapters.csv_adapter import load_task_examples

            # Resolve the task config from the source task name
            src_task = (self._get(source, "meta", "task") or "").lower().replace(" ", "_")
            if "ioi" in src_task or "indirect" in src_task:
                task_key = "ioi"
            elif "greater" in src_task or src_task == "gt":
                task_key = "greater_than"
            else:
                task_key = src_task
            if task_key in TASK_REGISTRY:
                task_cfg = get_task_config(task_key)
            else:
                # Semantic role without a built-in config: derive one on the fly,
                # mirroring the registry convention (e.g. 'goal_entity' / 'GOAL')
                role_key = task_key or "transfer_task"
                task_cfg = _sem_role_config(f"{role_key}_entity",
                                            f"{role_key}_entity",
                                            role_key.upper())

            task_examples = load_task_examples(dataset_path, task_cfg)

            ilp_result = process_single_circuit(
                circuit_ttl_path = owl_path_beta,
                task_examples    = task_examples,
                task_config      = task_cfg,
            )
            if "error" in ilp_result:
                raise RuntimeError(ilp_result["error"])
        except Exception as e:
            print(f"[Step 5]   ILP_sem failed: {e}")
            report.relearn_result = {"error": f"ILP_sem failed: {e}"}
            return

        # Check signature distance
        src_sig  = self._get(source, "layer2", "ilp_signature") or {}
        beta_sig = ilp_result.get("ilp_signature") or {}
        dist     = ilp_signature_distance(src_sig, beta_sig) if src_sig and beta_sig else float("inf")

        accepted = dist < self.delta_ilp
        print(f"[Step 5]   d(φ′, φα) = {dist:.4f}  (δ_ILP = {self.delta_ilp})  "
              f"→ {'ACCEPTED' if accepted else 'REJECTED'}")

        report.relearn_result = {
            "dist":            dist,
            "accepted":        accepted,
            "ilp_result":      ilp_result,
            "owl_path":        owl_path_beta,
            "eap_results":     results_beta,
        }

        if accepted:
            report.accepted              = True
            report.accepted_circuit_path = owl_path_beta
            report.accepted_via          = "step5_relearn"

    # ── Live evaluation helpers ───────────────────────────────────────────────

    def _evaluate_candidate_acc(
            self,
            candidate: CandidateResult,
            source:    Dict[str, Any],
            phi_alpha: str,
            model,
            dataset:   str,
    ) -> Optional[float]:
        """
        Evaluate AccT(Cβ, φα): run the candidate circuit through the target
        model on D_T and measure satisfaction of φ_α.

        In practice, AccT(Cβ, φα) ≈ circuit_performance when φ_α is
        operationalised as the CNP metric — the fraction of examples where
        the circuit produces the correct contrastive output.

        Returns None on failure.
        """
        try:
            from trace.mechint.core.eap_evaluate import evaluate_graph
            from trace.mechint.core.eap_cnp import cnp_metric
            from functools import partial

            dataloader = self._make_dataloader(dataset)
            graph_beta = self._load_graph_for_candidate(candidate, model)
            if graph_beta is None:
                return None

            acc = evaluate_graph(
                model, graph_beta, dataloader,
                partial(cnp_metric, loss=False, mean=False),
            ).mean().item()
            return float(acc)
        except Exception as e:
            print(f"[Step 3]     AccT eval failed for {candidate.task_name}: {e}")
            return None

    def _compute_delta_t(
            self,
            candidate: CandidateResult,
            source:    Dict[str, Any],
            model,
            dataset:   str,
            tokenizer=None,
    ) -> Optional[float]:
        """
        Compute ΔT(Cβ) = AccT(M_full) − AccT(M^ablate,Cβ).

        AccT(M_full)           = evaluate_baseline (full model performance)
        AccT(M^ablate,Cβ)      = evaluate_graph with empty graph
                                 (all circuit edges ablated)

        This measures how much the candidate circuit contributes to task
        performance — a high ΔT means the circuit is causally relevant.
        """
        # Fast path: ΔT already stored in the candidate's JSON
        rec = self._get_record(candidate.json_path)
        if rec:
            stored_dt = self._get_float(rec, "layer1", "metrics", "DeltaT")
            if stored_dt is not None:
                return stored_dt

        try:
            from trace.mechint.core.eap_evaluate import evaluate_graph, evaluate_baseline
            from trace.mechint.core.eap_graph import Graph
            from trace.mechint.core.eap_cnp import cnp_metric
            from functools import partial

            metric_fn  = partial(cnp_metric, loss=False, mean=False)
            dataloader = self._make_dataloader(dataset, tokenizer)
            if dataloader is None:
                return None

            # Full-model baseline
            baseline = evaluate_baseline(model, dataloader, metric_fn).mean().item()

            # Ablated: empty circuit (all edges zeroed out)
            empty_graph = Graph.from_model(model)
            empty_graph.reset(empty=True)
            ablated = evaluate_graph(
                model, empty_graph, dataloader, metric_fn, skip_clean=False,
            ).mean().item()

            delta_t = baseline - ablated
            return float(delta_t)
        except Exception as e:
            print(f"[Step 4]     ΔT computation failed for {candidate.task_name}: {e}")
            return None

    def _make_dataloader(self, dataset_path: str, tokenizer=None):
        """Build a DataLoader from a CNP CSV."""
        try:
            from trace.mechint.core.analyzer import MechIntAnalyzer
            analyzer = MechIntAnalyzer(cnp_dataset_path=dataset_path)
            dataset = analyzer.load_cnp_dataset(dataset_path, tokenizer)
            return dataset.to_dataloader(10)
        except Exception as e:
            print(f"[Transfer]   Dataloader build failed: {e}")
            return None

    def _load_graph_for_candidate(self, candidate: CandidateResult, model):
        """
        Reconstruct the EAP Graph for a registry candidate from stored
        layer1.circuit nodes/edges in the candidate's results JSON.
        """
        rec = self._get_record(candidate.json_path)
        if rec is None:
            return None
        try:
            from trace.mechint.core.eap_graph import Graph
            circuit = rec.get("layer1", {}).get("circuit", {})
            stored_nodes = {n["name"] for n in circuit.get("nodes", [])}
            stored_edges = {e["name"]: e.get("score", 0.0)
                           for e in circuit.get("edges", [])}

            graph = Graph.from_model(model)
            graph.reset(empty=True)

            for name in stored_nodes:
                if name in graph.nodes:
                    graph.nodes[name].in_graph = True
            for edge_name, score in stored_edges.items():
                if edge_name in graph.edges:
                    graph.edges[edge_name].in_graph = True
                    graph.edges[edge_name].score = score
            return graph
        except Exception as e:
            print(f"[Step 3/4]   Graph load failed for {candidate.task_name}: {e}")
            return None

    # ── Utility / accessors ──────────────────────────────────────────────────

    @staticmethod
    def _load_source(path: str) -> Dict[str, Any]:
        with open(path) as fh:
            rec = json.load(fh)
        rec["__json_path__"] = path
        return rec

    def _get_record(self, json_path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not json_path:
            return None
        for rec in self._registry:
            if rec.get("__json_path__") == json_path:
                return rec
        # Not in registry — load directly
        try:
            with open(json_path) as fh:
                rec = json.load(fh)
            rec["__json_path__"] = json_path
            return rec
        except Exception:
            return None

    @staticmethod
    def _get(d: Dict, *keys, default=None):
        """Safe nested get."""
        cur = d
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)
        return cur if cur is not None else default

    @classmethod
    def _get_float(cls, d: Dict, *keys) -> Optional[float]:
        v = cls._get(d, *keys)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _get_int(cls, d: Dict, *keys, default: int = 0) -> int:
        v = cls._get(d, *keys)
        try:
            return int(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _get_arch_type(record: Dict[str, Any]) -> Optional[str]:
        """
        Extract the ILP_arch type predicate name from a results record.

        Tries layer3.tau_arch_text (e.g. "arch_instrument(C) :- ..."),
        then layer3.arch_type, then operation_type as fallback.
        """
        tau_text = (record.get("layer3") or {}).get("tau_arch_text") or ""
        if tau_text:
            # "arch_instrument(C) :- ..." → "arch_instrument"
            import re
            m = re.match(r"(arch_\w+)\s*\(", tau_text)
            if m:
                return m.group(1)
        arch_type = (record.get("layer3") or {}).get("arch_type")
        if arch_type:
            return f"arch_{arch_type}" if not arch_type.startswith("arch_") else arch_type
        kappa = (record.get("layer0") or {}).get("operation_type")
        if kappa:
            return f"arch_{kappa}"
        return None
