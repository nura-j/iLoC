"""
ILP Architectural Signature Learner (ILP_arch)
==============================================
Implements §3.3 of the paper:

    ILP_arch learns *what circuits structurally look like* from the CKG,
    independently of prompts.

    Positive examples  : validated circuits of the same type
                         e.g. arch_location(C_loc_1), arch_location(C_loc_2)
    Negative examples  : circuits of other types
                         e.g. ¬arch_location(C_ioi_1), ¬arch_location(C_rand_1)

    Background knowledge B_arch is extracted from each CKG:
        node(C, v),   edge(C, u, v),   type(v, τ),   layer(v, ℓ),
        layer_span(C, ℓ_min, ℓ_max),
        has_motif(C, m),
        component_ratio(C, τ, r),
        size(C, n)

    Learned hypothesis H_arch (Eq. 4, e.g. for instrument circuits):
        arch_instrument(C) :-
            has_motif(C, scaffold_event_entity),
            layer_span(C, Lmin, Lmax), Lmin >= 4, Lmax =< 10,
            component_ratio(C, attention, R), R > 0.6,
            size(C, N), N < 30.

    ILP_arch signature:
        S_ILP_arch(τ_arch) = ⟨n_clauses, P_struct, complexity⟩   (Eq. 20)

    confidence_arch : fraction of positive circuits correctly classified.

Original code was heuristic feature extraction only — no Prolog clause
generation. This version:
  1. Extracts structural background knowledge from each OWL graph.
  2. Learns threshold constraints from feature distributions.
  3. Generates a well-formed Prolog clause matching the paper's Eq. 4.
  4. Evaluates the clause on held-out circuits for confidence.
  5. Computes S_ILP_arch signature with the correct fields.

"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from rdflib import Graph, Namespace, URIRef

_MECHINT_NS = Namespace("http://purl.org/mechint#")

# ── Topology-inferred motif names ────────────────────────────────────────────
# These are the canonical motif identifiers shared by:
#   - OWLExporter._detect_motifs()          (edge-walk detection on live graph)
#   - extract_background_knowledge()        (SPARQL read-back from stored TTL)
#   - _add_motifs_from_edges_sparql()       (edge-level SPARQL on stored TTL)
#   - schema.MechIntSchema.motif_types      (OWL class declarations)
#
# Changing a name here requires changing it in all four places.
_TOPOLOGY_MOTIFS = {
    'scaffold_event_entity',   # A → A → M  spanning ≥2 layers
    'induction_head',          # A[L] → A[L+1] via Q-stream
    'token_copy',              # early-A → late-A via V-stream
    'ff_chain',                # M[L] → M[L+1]  consecutive MLPs
    'broadcast_head',          # A with out-degree ≥ 3
    'attn_mlp_attn',           # A → M → A
}

# Fallback heuristics used ONLY when no edge triples exist in the OWL graph
# (e.g. older TTL files that predate topology storage).
# These deliberately mirror the topology conditions so they produce the same
# names as the edge-walk detector when aggregate statistics are all we have.
_HEURISTIC_FALLBACK = {
    "scaffold_event_entity": lambda f: (
        f["n_attn"] >= 2 and f["n_mlp"] >= 1 and f["layer_span"] >= 2
    ),
    "induction_head": lambda f: (
        f["n_attn"] >= 2 and f["attn_ratio"] > 0.7 and f["layer_span"] >= 1
    ),
    "token_copy": lambda f: (
        f["depth_mean"] < 0.4 and f["n_attn"] >= 1
    ),
    "ff_chain": lambda f: f["attn_ratio"] < 0.4 and f["n_mlp"] >= 2,
    "broadcast_head": lambda f: (
        f["n_attn"] >= 1 and f.get("hub_count", 0) >= 1
    ),
}


# ============================================================
# Topology-based motif reader (SPARQL on stored OWL)
# ============================================================

def _add_motifs_from_edges_sparql(
        circuit_graph,
        circuit_uri: URIRef,
        ns: str,
        feats: Dict[str, Any],
) -> List[str]:
    """
    Read topology-inferred motifs directly from the stored OWL graph.

    OWLExporter._detect_motifs() writes one mechint:hasMotif triple per
    motif type it finds via edge-walk.  The object is an OWL class URI such
    as mechint:ScaffoldEventEntity.  We map those URIs back to the canonical
    snake_case names expected by ilp_arch clause generation.

    Returns an empty list if no topology motifs are stored (caller should
    fall back to _HEURISTIC_FALLBACK).
    """
    # Map OWL local-name → canonical motif name
    # Must stay in sync with OWLExporter._detect_architectural_motifs()
    _OWL_TO_CANONICAL: Dict[str, str] = {
        'ScaffoldEventEntity': 'scaffold_event_entity',
        'InductionHead':       'induction_head',
        'TokenCopy':           'token_copy',
        'FeedForwardChain':    'ff_chain',
        'BroadcastHead':       'broadcast_head',
        'AttnMlpAttn':         'attn_mlp_attn',
        # Legacy → canonical (old TTL files)
        'BackwardAttention':   'induction_head',
    }

    motif_q = f"""
    PREFIX mechint: <{ns}>
    SELECT DISTINCT ?motif WHERE {{
        <{circuit_uri}> mechint:hasMotif ?motif
    }}
    """
    motifs: List[str] = []
    for row in circuit_graph.query(motif_q):
        local_name = str(row.motif).split('#')[-1]
        canonical = _OWL_TO_CANONICAL.get(local_name)
        if canonical and canonical not in motifs:
            motifs.append(canonical)
        elif local_name in _TOPOLOGY_MOTIFS and local_name not in motifs:
            # Already a canonical snake_case name stored directly
            motifs.append(local_name)

    # Additionally, check edge-level triples for motif types that may be
    # stored as motifType literals on Motif nodes (from _detect_motifs paths)
    motif_node_q = f"""
    PREFIX mechint: <{ns}>
    SELECT DISTINCT ?mtype WHERE {{
        <{circuit_uri}> mechint:hasMotif ?motif_node .
        ?motif_node mechint:motifType ?mtype .
    }}
    """
    for row in circuit_graph.query(motif_node_q):
        mtype = str(row.mtype)
        if mtype in _TOPOLOGY_MOTIFS and mtype not in motifs:
            motifs.append(mtype)

    return motifs


# ============================================================
# Background knowledge extraction from a single OWL graph
# ============================================================

def extract_background_knowledge(
        circuit_graph: Graph,
        circuit_uri:   URIRef,
        circuit_id:    str,
        n_layers:      int = 6,
) -> Dict[str, Any]:
    """
    Extract all B_arch structural predicates from one CKG.

    Returns a dict with:
        circuit_id, n_attn, n_mlp, layer_min, layer_max, layer_span,
        depth_mean, attn_ratio, motifs, size, component_ratio_str
    """
    ns = _MECHINT_NS

    # --- Count components by type and collect layer indices ---
    comp_q = f"""
    PREFIX mechint: <{ns}>
    PREFIX rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT ?comp ?ctype ?layer WHERE {{
        <{circuit_uri}> mechint:hasComponent ?comp .
        ?comp rdf:type ?ctype .
        OPTIONAL {{ ?comp mechint:inLayer ?layer }}
    }}
    """
    n_attn = 0
    n_mlp  = 0
    layers: List[int] = []

    for row in circuit_graph.query(comp_q):
        ctype = str(row.ctype).split('#')[-1]
        if 'AttentionHead' in ctype:
            n_attn += 1
        elif 'MLP' in ctype or 'Mlp' in ctype:
            n_mlp += 1
        if row.layer is not None:
            try:
                layers.append(int(row.layer))
            except (TypeError, ValueError):
                pass

    # --- Try reading pre-computed aggregates (set by OWL exporter) ---
    agg_q = f"""
    PREFIX mechint: <{ns}>
    SELECT ?depthMin ?depthMax ?depthMean ?ratio ?numAttn ?numMlp WHERE {{
        <{circuit_uri}> mechint:depthMin  ?depthMin .
        <{circuit_uri}> mechint:depthMax  ?depthMax .
        <{circuit_uri}> mechint:depthMean ?depthMean .
        OPTIONAL {{ <{circuit_uri}> mechint:componentRatio    ?ratio   }}
        OPTIONAL {{ <{circuit_uri}> mechint:numAttentionHeads ?numAttn }}
        OPTIONAL {{ <{circuit_uri}> mechint:numMLPBlocks      ?numMlp  }}
    }}
    """
    agg_rows = list(circuit_graph.query(agg_q))

    if agg_rows:
        row       = agg_rows[0]
        layer_min = float(row.depthMin)  if row.depthMin  else 0.0
        layer_max = float(row.depthMax)  if row.depthMax  else 1.0
        depth_mean= float(row.depthMean) if row.depthMean else 0.5
        if row.numAttn: n_attn = max(n_attn, int(row.numAttn))
        if row.numMlp:  n_mlp  = max(n_mlp,  int(row.numMlp))
    else:
        # Fall back to computed values from layer list
        if layers:
            layer_min  = min(layers) / max(n_layers - 1, 1)
            layer_max  = max(layers) / max(n_layers - 1, 1)
            depth_mean = sum(layers) / len(layers) / max(n_layers - 1, 1)
        else:
            layer_min = layer_max = depth_mean = 0.5

    size        = n_attn + n_mlp
    layer_span  = layer_max - layer_min
    attn_ratio  = n_attn / size if size > 0 else 0.5
    comp_ratio  = f"{n_attn}:{n_mlp}" if n_mlp > 0 else f"{n_attn}:0"

    feats = {
        "circuit_id":        circuit_id,
        "n_attn":            n_attn,
        "n_mlp":             n_mlp,
        "size":              size,
        "layer_min":         round(layer_min,  4),
        "layer_max":         round(layer_max,  4),
        "layer_span":        round(layer_span, 4),
        "depth_mean":        round(depth_mean, 4),
        "attn_ratio":        round(attn_ratio, 4),
        "component_ratio":   comp_ratio,
    }

    # --- Motif detection (topology-first, heuristic fallback) ----------------
    #
    # Primary path: read topology-inferred motifs from OWL edge triples.
    # OWLExporter._detect_motifs() now stores one hasMotif triple per motif
    # type found by the edge-walk detector, using the canonical names in
    # _TOPOLOGY_MOTIFS.  We read them here via SPARQL.
    #
    # Fallback: if no edge-based motifs are stored (old TTL), apply
    # _HEURISTIC_FALLBACK on the aggregate feature dict.  This produces
    # the same canonical names so downstream clause generation is unaffected.

    topology_motifs = _add_motifs_from_edges_sparql(
        circuit_graph, circuit_uri, ns, feats
    )

    if topology_motifs:
        motifs = topology_motifs
    else:
        # Fallback: heuristics on aggregate stats (old TTL / no edge triples)
        motifs = [
            name for name, test in _HEURISTIC_FALLBACK.items() if test(feats)
        ]

    feats["motifs"] = motifs
    return feats


# ============================================================
# Clause generation from feature distributions
# ============================================================

def _threshold(values: List[float], percentile: float) -> float:
    """Return a robust threshold at a given percentile."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, int(len(sorted_v) * percentile / 100) - 1)
    return round(sorted_v[idx], 3)


def _generate_arch_clause(
        arch_type: str,
        pos_feats: List[Dict],
        neg_feats: List[Dict],
) -> Tuple[str, List[str]]:
    """
    Learn a Prolog clause from feature distributions (ILP_arch, Eq. 4).

    Strategy (greedy ILP approximation):
    1. Extract feature distributions over positive circuits.
    2. For each structural predicate, find a threshold that separates
       positive from negative examples with ≥1 margin.
    3. Only include a predicate in the clause body if it contributes
       discriminative power (reduces false positives).
    4. Include common motifs if they appear in ≥ 50% of positive circuits.

    Returns (clause_text, body_predicate_names)
    """
    body_parts: List[str]  = []
    preds_used: List[str]  = []

    # --- layer_span ---
    pos_lmin  = [f["layer_min"] for f in pos_feats]
    pos_lmax  = [f["layer_max"] for f in pos_feats]
    neg_lmin  = [f["layer_min"] for f in neg_feats]
    neg_lmax  = [f["layer_max"] for f in neg_feats]

    span_lmin = _threshold(pos_lmin, 10)   # generous lower bound
    span_lmax = _threshold(pos_lmax, 90)   # generous upper bound

    # Include layer_span only if it discriminates
    if neg_feats:
        neg_out = sum(
            1 for f in neg_feats
            if not (span_lmin <= f["layer_min"] and f["layer_max"] <= span_lmax)
        )
        include_span = neg_out >= len(neg_feats) * 0.3
    else:
        include_span = True

    if include_span:
        body_parts.append(f"layer_span(C, Lmin, Lmax)")
        body_parts.append(f"Lmin >= {span_lmin}")
        body_parts.append(f"Lmax =< {span_lmax}")
        preds_used.append("layer_span")

    # --- component_ratio ---
    attn_ratios = [f["attn_ratio"] for f in pos_feats]
    min_ratio   = _threshold(attn_ratios, 20)
    if neg_feats:
        neg_out_r = sum(
            1 for f in neg_feats if f["attn_ratio"] < min_ratio
        )
        include_ratio = neg_out_r >= len(neg_feats) * 0.2 and min_ratio > 0.1
    else:
        include_ratio = min_ratio > 0.1

    if include_ratio:
        body_parts.append(f"component_ratio(C, attention, R)")
        body_parts.append(f"R > {min_ratio}")
        preds_used.append("component_ratio")

    # --- rel_size (relative to model capacity — cross-model invariant) ---
    # Use rel_size (circuit_components / model_total_heads) instead of raw
    # component count. This makes the clause meaningful across model sizes:
    # a location circuit in Pythia-14M and Pythia-1B may differ in absolute
    # size but should occupy a similar *fraction* of the model.
    rel_sizes   = [f.get("rel_size", f["size"] / max(f["n_layers"], 1))
                   for f in pos_feats]
    max_rel     = round(_threshold(rel_sizes, 90) + 0.05, 3)
    min_rel     = round(_threshold(rel_sizes, 10) - 0.05, 3)
    if neg_feats:
        neg_out_s = sum(
            1 for f in neg_feats
            if f.get("rel_size", 1.0) >= max_rel
        )
        include_size = neg_out_s >= len(neg_feats) * 0.2
    else:
        include_size = True

    if include_size and max_rel > 0.01:
        body_parts.append(f"rel_size(C, N)")
        body_parts.append(f"N < {max_rel}")
        preds_used.append("rel_size")

    # --- motifs ---
    all_motifs = [m for f in pos_feats for m in f.get("motifs", [])]
    motif_counts = Counter(all_motifs)
    common_motifs = [
        m for m, cnt in motif_counts.items()
        if cnt >= len(pos_feats) * 0.5 and m
    ]
    for m in common_motifs[:2]:   # at most 2 motif constraints
        body_parts.append(f"has_motif(C, {m})")
        preds_used.append("has_motif")

    # Build clause
    head = f"arch_{arch_type.lower()}(C)"
    if body_parts:
        body = ",\n    ".join(body_parts)
        clause = f"{head} :-\n    {body}."
    else:
        clause = f"{head} :- true."

    return clause, sorted(set(preds_used))


# ============================================================
# Public learner class
# ============================================================

class ArchitecturalSignatureLearner:
    """
    Learn architectural signatures (ILP_arch, §3.3) from multiple circuits.

    Unlike the original, this class:
    1. Extracts B_arch background knowledge from each OWL graph.
    2. Generates a real Prolog clause (Eq. 4).
    3. Evaluates the clause for confidence.
    4. Computes the ILP_arch signature S_ILP_arch (Eq. 20).

    Usage
    -----
    learner   = ArchitecturalSignatureLearner()
    signature = learner.learn_signature(
        pos_graphs=[ioi_graph_1, ioi_graph_2],
        neg_graphs=[loc_graph_1, rand_graph_1],
        arch_type="ioi",
        n_layers=6,
    )
    # signature keys: clause, predicates, confidence, ilp_signature,
    #                 depth_mean, layer_span, component_ratio, motifs,
    #                 num_attn, num_mlp
    """

    def __init__(self) -> None:
        self._ns = _MECHINT_NS

    # ----------------------------------------------------------------
    # Main entry point
    # ----------------------------------------------------------------

    def learn_signature(
            self,
            pos_graphs: List[Graph],
            neg_graphs:  List[Graph],
            arch_type:   str,
            n_layers:    int = 6,
    ) -> Dict:
        """
        Learn an architectural signature from positive and negative examples.

        Parameters
        ----------
        pos_graphs : OWL graphs of circuits with the *same* arch_type
        neg_graphs : OWL graphs of circuits of *other* types (negative examples)
        arch_type  : Name for the learned type, e.g. "ioi" or "location"
        n_layers   : Total layers in the model (for depth normalisation)

        Returns
        -------
        {
          'arch_type':        str,
          'clause':           str,    # full Prolog clause (Eq. 4)
          'predicates':       list,   # body predicate names
          'confidence':       float,  # fraction of pos circuits correctly classified
          'ilp_signature':    dict,   # S_ILP_arch (Eq. 20)
          # Aggregate feature summaries (for JSON registry):
          'depth_mean':       float,
          'depth_min':        float,
          'depth_max':        float,
          'layer_span':       float,
          'component_ratio':  str,
          'num_attn':         int,
          'num_mlp':          int,
          'motifs':           list,
        }
        """
        print("\n" + "=" * 60)
        print(f"ILP_arch  Architectural Signature Learning  [{arch_type}]")
        print("=" * 60)
        print(f"  Positive circuits: {len(pos_graphs)}")
        print(f"  Negative circuits: {len(neg_graphs)}")

        # Extract features
        pos_feats = [
            self._extract_features(g, arch_type, n_layers) for g in pos_graphs
        ]
        neg_feats = [
            self._extract_features(g, f"other_{i}", n_layers)
            for i, g in enumerate(neg_graphs)
        ]

        # Generate Prolog clause from feature distributions (Eq. 4)
        clause, preds = _generate_arch_clause(arch_type, pos_feats, neg_feats)
        print(f"\n  Learned clause:\n  {clause}")

        # Evaluate confidence on positive circuits
        confidence = self._evaluate_clause(clause, arch_type, pos_feats, neg_feats)
        print(f"\n  Confidence: {confidence:.4f}")

        # Aggregate structural statistics for JSON registry
        depths       = [f["depth_mean"]  for f in pos_feats]
        lmins        = [f["layer_min"]   for f in pos_feats]
        lmaxs        = [f["layer_max"]   for f in pos_feats]
        attn_counts  = [f["n_attn"]      for f in pos_feats]
        mlp_counts   = [f["n_mlp"]       for f in pos_feats]
        all_motifs   = [m for f in pos_feats for m in f.get("motifs", [])]
        common_motifs= [m for m, c in Counter(all_motifs).items()
                        if c >= max(len(pos_feats) * 0.5, 1)]

        avg_attn   = round(sum(attn_counts) / len(attn_counts), 1) if attn_counts else 0
        avg_mlp    = round(sum(mlp_counts)  / len(mlp_counts),  1) if mlp_counts  else 0
        depth_mean = round(sum(depths) / len(depths), 4)           if depths      else 0.5
        depth_min  = round(min(lmins), 4)                          if lmins       else 0.0
        depth_max  = round(max(lmaxs), 4)                          if lmaxs       else 1.0
        span       = round(depth_max - depth_min, 4)

        comp_ratio_str = (
            f"{round(avg_attn)}:{round(avg_mlp)}"
            if avg_mlp > 0 else f"{round(avg_attn)}:0"
        )

        sig = self._compute_ilp_arch_signature(clause, preds)
        print(f"  S_ILP_arch: {sig}")

        return {
            "arch_type":        arch_type,
            "clause":           clause,
            "predicates":       preds,
            "confidence":       confidence,
            "ilp_signature":    sig,
            "depth_mean":       depth_mean,
            "depth_min":        depth_min,
            "depth_max":        depth_max,
            "layer_span":       span,
            "component_ratio":  comp_ratio_str,
            "num_attn":         int(round(avg_attn)),
            "num_mlp":          int(round(avg_mlp)),
            "motifs":           common_motifs,
        }

    # ----------------------------------------------------------------
    # Single-graph variant (backward compatible with original API)
    # ----------------------------------------------------------------

    def learn_signature_single(
            self,
            circuit_graph: Graph,
            arch_type:     str = "unknown",
            n_layers:      int = 6,
    ) -> Dict:
        """Learn from a single circuit graph (creates a generic signature)."""
        return self.learn_signature(
            pos_graphs=[circuit_graph],
            neg_graphs=[],
            arch_type=arch_type,
            n_layers=n_layers,
        )

    # ----------------------------------------------------------------
    # Backward-compatible original API
    # ----------------------------------------------------------------

    def get_architectural_features(
            self,
            circuit_graph: Graph,
            circuit_uri:   URIRef,
            n_layers:      int = 6,
    ) -> Dict:
        """Extract features from a single circuit (original API)."""
        cid = self._simplify_uri(circuit_uri)
        return extract_background_knowledge(
            circuit_graph, circuit_uri, cid, n_layers
        )

    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------

    def _extract_features(
            self, graph: Graph, arch_type: str, n_layers: int
    ) -> Dict:
        uri = self._get_circuit_uri(graph)
        cid = self._simplify_uri(uri)
        return extract_background_knowledge(graph, uri, cid, n_layers)

    def _evaluate_clause(
            self,
            clause: str,
            arch_type: str,
            pos_feats: List[Dict],
            neg_feats: List[Dict],
    ) -> float:
        """
        Evaluate the clause by checking which circuits it correctly classifies.

        confidence_arch = (true_positives + true_negatives) / total
        """
        tp = tn = fp = fn = 0

        for f in pos_feats:
            if self._clause_matches(clause, arch_type, f):
                tp += 1
            else:
                fn += 1

        for f in neg_feats:
            if self._clause_matches(clause, arch_type, f):
                fp += 1
            else:
                tn += 1

        total = tp + tn + fp + fn
        return round((tp + tn) / total, 4) if total > 0 else 0.5

    def _clause_matches(
            self, clause: str, arch_type: str, feats: Dict
    ) -> bool:
        """
        Evaluate the learned clause against a feature dict symbolically.
        Supports constraints: Lmin >= X, Lmax =< X, R > X, N < X, has_motif, ...
        """
        if ':-' not in clause:
            return True   # unconditional clause

        _, body = clause.split(':-', 1)
        body = body.rstrip('.').strip()
        conditions = [c.strip() for c in body.split(',') if c.strip()]

        for cond in conditions:
            cond = cond.strip()

            # Numeric comparison: "Lmin >= X", "R > X", "N < X", "Lmax =< X"
            m = re.match(r'^(\w+)\s*(>=|=<|>|<|=:=|=\\=)\s*([\d.]+)$', cond)
            if m:
                var_name, op, rhs_str = m.groups()
                rhs = float(rhs_str)
                lhs = self._resolve_var(var_name, arch_type, feats)
                if lhs is None:
                    continue
                ok = (
                    (op == '>='   and lhs >= rhs) or
                    (op == '=<'   and lhs <= rhs) or
                    (op == '>'    and lhs >  rhs) or
                    (op == '<'    and lhs <  rhs) or
                    (op == '=:='  and lhs == rhs) or
                    (op == '=\\=' and lhs != rhs)
                )
                if not ok:
                    return False
                continue

            # has_motif(C, motif_name)
            m2 = re.match(r'^has_motif\(\w+,\s*(\w+)\)$', cond)
            if m2:
                motif_name = m2.group(1)
                if motif_name not in feats.get("motifs", []):
                    return False
                continue

            # layer_span(C, ...)  / component_ratio(C, ...) / size(C, ...) — binding
            # these just bind variables, the actual constraint is in a comparison
            if re.match(r'^\w+\(', cond):
                continue

        return True

    def _resolve_var(self, var: str, arch_type: str, feats: Dict) -> Optional[float]:
        """Map Prolog variable names to feature values."""
        mapping = {
            "Lmin": "layer_min",
            "Lmax": "layer_max",
            "R":    "attn_ratio",
            "N":    "size",
            "D":    "depth_mean",
        }
        key = mapping.get(var)
        if key and key in feats:
            return float(feats[key])
        return None

    def _compute_ilp_arch_signature(
            self, clause: str, predicates: List[str]
    ) -> Dict:
        """
        Compute S_ILP_arch signature (Eq. 20):
            ⟨n_clauses, P_struct, complexity⟩
        """
        n_clauses  = 1
        n_body     = len(predicates)
        complexity = n_clauses * n_body
        return {
            "n_clauses":   n_clauses,
            "predicates":  predicates,
            "complexity":  complexity,
        }

    def _get_circuit_uri(self, graph: Graph) -> URIRef:
        q = """
        SELECT ?c WHERE {
            ?c a <http://purl.org/mechint#Circuit> .
        } LIMIT 1
        """
        rows = list(graph.query(q))
        if rows:
            return rows[0][0]
        # Fallback: pick any subject
        for s, p, o in graph:
            return s
        raise ValueError("Empty graph — cannot find circuit URI")

    def _simplify_uri(self, uri) -> str:
        uri_str = str(uri)
        local = uri_str.split('#')[-1] if '#' in uri_str else uri_str.split('/')[-1]
        return re.sub(r'[^a-zA-Z0-9_]', '_', local)


# ══════════════════════════════════════════════════════════════════════════════
# ArchitecturalILPLearner and its helpers
# (merged from ilp_arch_full.py — consolidated into this single module)
#
# ArchitecturalILPLearner is the full ILP_arch learner (paper §3.3 + Appendix D).
# It uses topology-based motif detection via _infer_motifs() when operating on
# JSON circuit dicts (layer1.circuit), and _add_motifs_from_edges_sparql() when
# operating on OWL graphs — both produce the canonical _TOPOLOGY_MOTIFS names.
# ══════════════════════════════════════════════════════════════════════════════

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from rdflib import Graph, Namespace, URIRef


# ══════════════════════════════════════════════════════════════════
# 1.  RELATIVE FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════

def _features_from_circuit_dict(
        circuit: Dict[str, Any],
        model_n_layers:    int  = 0,   # 0 = infer from circuit itself
        model_n_heads:     int  = 0,   # 0 = infer from circuit itself
        model_total_heads: int  = 0,   # 0 = n_layers * n_heads
) -> Dict[str, Any]:
    """
    Extract all structural features from a layer1.circuit dict.

    All depth/ratio/size values are RELATIVE to model dimensions.

    Parameters
    ----------
    circuit          : layer1.circuit dict from export_task_json
    model_n_layers   : total transformer layers (for normalisation)
    model_n_heads    : attention heads per layer
    model_total_heads: total possible attention heads

    Returns
    -------
    Feature dict with keys:
        num_nodes, num_edges, num_attn, num_mlp, n_layers (model)
        depth_min, depth_max, depth_mean  (all in [0,1] relative)
        depth_span  (= depth_max - depth_min)
        component_ratio_float  (attn / total, [0,1])
        component_ratio_str    (e.g. "10:2")
        rel_size               (circuit nodes / model_total_heads)
        density                (edges / nodes, or 0)
        attn_fraction_of_model (num_attn / model_total_heads)
        motifs                 (list[str], inferred from topology)
        node_type_seq          (sorted list of (layer, type) pairs)
        layer_type_pairs       (Counter of (type_src, type_tgt) edge pairs)
        path3_types            (Counter of (A,B,C) 3-node type paths)
        skip_connections       (count of layer-skipping edges)
        hub_components         (list of (name, in+out degree) for hubs)
        layer_attn_profile     (dict layer→fraction_of_attn_at_that_layer)
    """
    nodes = circuit.get("nodes", [])
    edges = circuit.get("edges", [])

    attn_nodes = [n for n in nodes if n.get("type") == "attn"]
    mlp_nodes  = [n for n in nodes if n.get("type") == "mlp"]
    num_attn   = len(attn_nodes)
    num_mlp    = len(mlp_nodes)
    total_comps = num_attn + num_mlp or 1

    # ── Layer / depth (relative) ──────────────────────────────────────────
    raw_layers = [n["layer"] for n in nodes if "layer" in n]
    if model_n_layers <= 0:
        # Infer model size: max observed layer + 1
        model_n_layers = (max(raw_layers) + 1) if raw_layers else 6

    # Clip to valid range
    raw_layers_valid = [min(l, model_n_layers - 1) for l in raw_layers]
    depth_min  = min(raw_layers_valid) / model_n_layers if raw_layers_valid else 0.0
    depth_max  = max(raw_layers_valid) / model_n_layers if raw_layers_valid else 0.0
    depth_mean = (sum(raw_layers_valid) / len(raw_layers_valid) / model_n_layers
                  if raw_layers_valid else 0.0)
    depth_span = depth_max - depth_min

    # ── Relative size ──────────────────────────────────────────────────────
    if model_total_heads <= 0:
        n_heads = model_n_heads if model_n_heads > 0 else max(4, num_attn)
        model_total_heads = model_n_layers * n_heads
    rel_size = len(nodes) / max(model_total_heads, 1)

    # ── Density ────────────────────────────────────────────────────────────
    density = len(edges) / len(nodes) if nodes else 0.0

    # ── Attn fraction of full model ────────────────────────────────────────
    attn_fraction_of_model = num_attn / max(model_total_heads, 1)

    # ── Node-type sequence (sorted by layer) ──────────────────────────────
    node_type_seq = sorted(
        [(n.get("layer", 0), n.get("type", "?")) for n in nodes],
        key=lambda x: x[0]
    )

    # ── Per-layer attention profile ────────────────────────────────────────
    layer_attn_counts: Dict[int, int] = defaultdict(int)
    for n in attn_nodes:
        layer_attn_counts[n.get("layer", 0)] += 1
    layer_attn_profile = {
        l: c / max(num_attn, 1) for l, c in layer_attn_counts.items()
    }

    # ── Edge topology: build adjacency for graph-level motif extraction ────
    name_to_type: Dict[str, str] = {
        n.get("name", f"n{i}"): n.get("type", "?")
        for i, n in enumerate(nodes)
    }
    name_to_layer: Dict[str, int] = {
        n.get("name", f"n{i}"): n.get("layer", 0)
        for i, n in enumerate(nodes)
    }

    # degree counting for hub detection
    in_degree:  Counter = Counter()
    out_degree: Counter = Counter()
    for e in edges:
        s = e.get("source", "")
        t = e.get("target", "")
        out_degree[s] += 1
        in_degree[t]  += 1

    # 2-node type-pair motifs (edge type patterns)
    # Also accumulate attribution scores per edge-type pair
    pair_counts:  Counter = Counter()
    pair_scores:  Dict[tuple, List[float]] = {}   # (src_type, tgt_type) → [scores]
    skip_connections = 0
    for e in edges:
        s, t   = e.get("source", ""), e.get("target", "")
        ts, tt = name_to_type.get(s, "?"), name_to_type.get(t, "?")
        pair_counts[(ts, tt)] += 1
        pair_scores.setdefault((ts, tt), []).append(
            abs(float(e.get("score", 0.0))))   # abs — EAP scores can be negative
        ls = name_to_layer.get(s, 0)
        lt = name_to_layer.get(t, 0)
        if abs(lt - ls) > 1:
            skip_connections += 1

    # Mean attribution score per edge-type pair (0.0 if no scores stored)
    pair_mean_score: Dict[tuple, float] = {
        k: (sum(v) / len(v)) for k, v in pair_scores.items()
    }

    # 3-node path type motifs + path scores
    adj:         Dict[str, List[str]]   = defaultdict(list)
    edge_score_map: Dict[tuple, float]  = {}  # (src,tgt) → score
    for e in edges:
        s = e.get("source", ""); t = e.get("target", "")
        adj[s].append(t)
        edge_score_map[(s, t)] = abs(float(e.get("score", 0.0)))

    path3_counts: Counter = Counter()
    path3_scores: Dict[tuple, List[float]] = {}
    for src, mid_list in adj.items():
        ts = name_to_type.get(src, "?")
        for mid in mid_list:
            tm = name_to_type.get(mid, "?")
            for end in adj.get(mid, []):
                te = name_to_type.get(end, "?")
                path3_counts[(ts, tm, te)] += 1
                # path score = geometric mean of two edge scores
                s1 = edge_score_map.get((src, mid), 0.0)
                s2 = edge_score_map.get((mid, end), 0.0)
                path3_scores.setdefault((ts, tm, te), []).append(
                    math.sqrt(s1 * s2) if s1 > 0 and s2 > 0 else 0.0)

    path3_mean_score: Dict[tuple, float] = {
        k: (sum(v) / len(v)) for k, v in path3_scores.items()
    }

    # Hub components (degree > mean + 1*std)
    all_degrees = [in_degree[n.get("name","")] + out_degree[n.get("name","")]
                   for n in nodes]
    if len(all_degrees) > 1:
        mean_deg = sum(all_degrees) / len(all_degrees)
        std_deg  = math.sqrt(sum((d - mean_deg)**2 for d in all_degrees)
                             / len(all_degrees))
        hub_threshold = mean_deg + std_deg
    else:
        hub_threshold = 1.0
    hub_components = [
        (n.get("name", "?"),
         in_degree[n.get("name","")] + out_degree[n.get("name","")])
        for n in nodes
        if (in_degree[n.get("name","")] + out_degree[n.get("name","")]) >= hub_threshold
    ]
    hub_components.sort(key=lambda x: -x[1])

    # ── Infer motifs from topology ─────────────────────────────────────────
    motifs = _infer_motifs(
        num_attn=num_attn, num_mlp=num_mlp,
        depth_min=depth_min, depth_mean=depth_mean, depth_max=depth_max,
        depth_span=depth_span,
        component_ratio_float=num_attn / total_comps,
        pair_counts=pair_counts,
        path3_counts=path3_counts,
        skip_connections=skip_connections,
        hub_count=len(hub_components),
        layer_attn_profile=layer_attn_profile,
        model_n_layers=model_n_layers,
    )

    return {
        # ── Relative features (model-size-invariant, used for transfer) ──────
        "num_nodes":               len(nodes),
        "num_edges":               len(edges),
        "num_attn":                num_attn,
        "num_mlp":                 num_mlp,
        "n_layers":                model_n_layers,
        "depth_min":               round(depth_min,  4),   # normalised [0,1]
        "depth_max":               round(depth_max,  4),   # normalised [0,1]
        "depth_mean":              round(depth_mean, 4),   # normalised [0,1]
        "depth_span":              round(depth_span, 4),   # normalised [0,1]
        "component_ratio_float":   round(num_attn / total_comps, 4),
        "component_ratio_str":     f"{num_attn}:{num_mlp}",
        "rel_size":                round(rel_size, 4),     # size / model_total_heads
        "density":                 round(density, 4),
        "attn_fraction_of_model":  round(attn_fraction_of_model, 4),
        "skip_connections":        skip_connections,
        "hub_count":               len(hub_components),
        "hub_components":          hub_components[:5],
        "node_type_seq":           node_type_seq,
        "layer_attn_profile":      layer_attn_profile,
        "layer_type_pairs":        dict(pair_counts),
        "path3_types":             dict(path3_counts),
        "pair_mean_score":         {str(k): v for k, v in pair_mean_score.items()},
        "path3_mean_score":        {str(k): v for k, v in path3_mean_score.items()},
        "motifs":                  motifs,
        # ── Exact / absolute records (preserved for inspection, NOT used in
        #    transfer distance — comparing raw counts across models is meaningless)
        "exact": {
            "num_attn":        num_attn,
            "num_mlp":         num_mlp,
            "total_components": total_comps,
            "layer_min_abs":   min(raw_layers_valid) if raw_layers_valid else 0,
            "layer_max_abs":   max(raw_layers_valid) if raw_layers_valid else 0,
            "layer_indices":   sorted(set(raw_layers_valid)),
            "model_n_layers":  model_n_layers,
            "model_n_heads":   model_n_heads,
            "model_total_heads": model_total_heads,
        },
    }


def _features_from_rdf_graph(g: Graph, circuit_uri: URIRef,
                               model_n_layers: int = 0) -> Dict[str, Any]:
    """Extract features from a CKG rdflib Graph (SPARQL-based)."""
    ns = Namespace("http://purl.org/mechint#")

    # Components
    q_comp = f"""
    PREFIX mechint: <{ns}>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT ?comp ?type ?layer ?head WHERE {{
        <{circuit_uri}> mechint:hasComponent ?comp .
        ?comp rdf:type ?type .
        OPTIONAL {{ ?comp mechint:inLayer ?layer }}
        OPTIONAL {{ ?comp mechint:headIndex ?head }}
    }}"""
    comp_rows = list(g.query(q_comp))

    # Edges
    q_edge = f"""
    PREFIX mechint: <{ns}>
    SELECT ?src ?tgt WHERE {{
        <{circuit_uri}> mechint:hasMemberEdge ?e .
        ?e mechint:hasSource ?src ; mechint:hasTarget ?tgt .
    }}"""
    edge_rows = list(g.query(q_edge))

    def _short(uri):
        s = str(uri)
        return s.split("#")[-1] if "#" in s else s.split("/")[-1]

    nodes = []
    for r in comp_rows:
        tp = "attn" if "AttentionHead" in str(r.type) else "mlp"
        nodes.append({
            "name":  _short(r.comp),
            "type":  tp,
            "layer": int(r.layer) if r.layer is not None else 0,
        })

    edges = [
        {"source": _short(r.src), "target": _short(r.tgt)}
        for r in edge_rows
    ]

    circuit_dict = {
        "nodes": nodes,
        "edges": edges,
        "n_layers": model_n_layers or (max(n["layer"] for n in nodes) + 1 if nodes else 6),
    }
    return _features_from_circuit_dict(circuit_dict,
                                        model_n_layers=model_n_layers)


# ══════════════════════════════════════════════════════════════════
# 2.  TOPOLOGY-INFERRED MOTIFS  (replaces heuristic rule table)
# ══════════════════════════════════════════════════════════════════

def _infer_motifs(
        num_attn: int,
        num_mlp:  int,
        depth_min: float,
        depth_mean: float,
        depth_max: float,
        depth_span: float,
        component_ratio_float: float,
        pair_counts:  Counter,
        path3_counts: Counter,
        skip_connections: int,
        hub_count: int,
        layer_attn_profile: Dict[int, float],
        model_n_layers: int,
) -> List[str]:
    """
    Infer motif labels from actual graph topology.

    Each motif corresponds to a structural property that can be
    tested with a simple predicate in the ILP hypothesis.

    Motifs are RELATIVE:
      - depth thresholds are fractions of model depth
      - ratio thresholds are circuit-size-independent

    Returns list of motif name strings that apply.
    """
    motifs: List[str] = []

    # ── Edge-topology motifs ──────────────────────────────────────────────
    # attn→attn  (attention-to-attention routing, e.g. induction heads)
    if pair_counts.get(("attn", "attn"), 0) >= 2:
        motifs.append("attn_to_attn")

    # attn→mlp   (standard residual: attention feeds MLP)
    if pair_counts.get(("attn", "mlp"), 0) >= 1:
        motifs.append("attn_feeds_mlp")

    # mlp→attn   (MLP gates attention — less common, strong signal)
    if pair_counts.get(("mlp", "attn"), 0) >= 1:
        motifs.append("mlp_gates_attn")

    # ── 3-node path motifs ─────────────────────────────────────────────────
    # attn→attn→attn  (pure attention chain = induction-like)
    if path3_counts.get(("attn", "attn", "attn"), 0) >= 1:
        motifs.append("attn_chain_3")

    # attn→mlp→attn   (sandwich: MLP in the middle)
    if path3_counts.get(("attn", "mlp", "attn"), 0) >= 1:
        motifs.append("attn_mlp_attn_sandwich")

    # attn→attn→mlp   (scaffold→event→entity, paper Appendix D)
    if path3_counts.get(("attn", "attn", "mlp"), 0) >= 1:
        motifs.append("scaffold_event_entity")

    # ── Depth profile motifs ───────────────────────────────────────────────
    # early_circuit: starts in first third of the model
    if depth_min < 0.33:
        motifs.append("early_circuit")

    # late_circuit: ends in last third
    if depth_max > 0.67:
        motifs.append("late_circuit")

    # deep_span: circuit spans more than half the model depth
    if depth_span > 0.5:
        motifs.append("deep_span")

    # shallow_span: circuit confined to one third
    if depth_span < 0.33:
        motifs.append("shallow_span")

    # ── Component-ratio motifs ─────────────────────────────────────────────
    # attn_dominant: >75% attention heads
    if component_ratio_float > 0.75:
        motifs.append("attn_dominant")

    # balanced_mix: neither type dominates (33–67%)
    if 0.33 <= component_ratio_float <= 0.67:
        motifs.append("balanced_mix")

    # mlp_heavy: >50% MLP
    if component_ratio_float < 0.5 and num_mlp >= 2:
        motifs.append("mlp_heavy")

    # ── Connectivity motifs ────────────────────────────────────────────────
    # skip_connection_present
    if skip_connections > 0:
        motifs.append("has_skip_connections")

    # hub_node_present: at least one high-degree node
    if hub_count >= 1:
        motifs.append("has_hub_node")

    # ── Layer-profile motifs (relative to model) ───────────────────────────
    # mid_layer_peak: most attention concentrated in middle layers
    if layer_attn_profile:
        peak_layer = max(layer_attn_profile, key=layer_attn_profile.get)
        peak_rel   = (peak_layer + 1) / max(model_n_layers, 1)
        if 0.3 < peak_rel < 0.7:
            motifs.append("mid_layer_attn_peak")
        elif peak_rel <= 0.3:
            motifs.append("early_attn_peak")
        else:
            motifs.append("late_attn_peak")

    return motifs


# ══════════════════════════════════════════════════════════════════
# 3.  DISCRIMINATIVE MOTIF SELECTION
# ══════════════════════════════════════════════════════════════════

def _binary_entropy(p: float) -> float:
    """H(p) = -p log2 p - (1-p) log2(1-p)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def _motif_scores(motif: str,
                  pos_features: List[Dict],
                  neg_features: List[Dict],
) -> Dict[str, Any]:
    """
    Compute all discriminative scores for a single motif.

    Returns a dict with keys:
        ig          — standard information gain  H(Y) - H(Y|present)
        gain_ratio  — IG / H(present)  [corrects IG bias toward rare features]
        gini_drop   — Gini impurity reduction
        attr_weight — mean EAP attribution of edges constituting this motif
                      in positive circuits (0.0 if no score data available)
        attr_ig     — attribution-weighted IG  =  ig * attr_weight
        tp, fp, tn, fn — contingency table
    """
    n_pos = len(pos_features)
    n_neg = len(neg_features)
    n_all = n_pos + n_neg
    if n_all == 0:
        return {"ig": 0.0, "gain_ratio": 0.0, "gini_drop": 0.0,
                "attr_weight": 0.0, "attr_ig": 0.0,
                "tp": 0, "fp": 0, "tn": 0, "fn": 0}

    tp = sum(1 for f in pos_features if motif in f.get("motifs", []))
    fp = sum(1 for f in neg_features if motif in f.get("motifs", []))
    tn = n_neg - fp
    fn = n_pos - tp
    n_with    = tp + fp
    n_without = tn + fn

    # ── Standard IG ──────────────────────────────────────────────────────
    p_pos    = n_pos / n_all
    h_prior  = _binary_entropy(p_pos)
    h_with   = _binary_entropy(tp / n_with)    if n_with    > 0 else 0.0
    h_without= _binary_entropy(fn / n_without) if n_without > 0 else 0.0
    h_cond   = (n_with / n_all) * h_with + (n_without / n_all) * h_without
    ig       = max(0.0, h_prior - h_cond)

    # ── Gain Ratio ────────────────────────────────────────────────────────
    # Divides IG by split information H(present/absent) to correct for
    # features that are very rare (high split info → low gain ratio).
    p_with   = n_with / n_all
    h_split  = _binary_entropy(p_with)
    gain_ratio = ig / h_split if h_split > 1e-9 else ig

    # ── Gini impurity reduction ───────────────────────────────────────────
    def _gini(pos: int, total: int) -> float:
        if total == 0: return 0.0
        p = pos / total
        return 1.0 - p * p - (1.0 - p) * (1.0 - p)
    gini_before = _gini(n_pos, n_all)
    gini_with   = _gini(tp, n_with)    if n_with    > 0 else 0.0
    gini_without= _gini(fn, n_without) if n_without > 0 else 0.0
    gini_after  = (n_with / n_all) * gini_with + (n_without / n_all) * gini_without
    gini_drop   = max(0.0, gini_before - gini_after)

    # ── Attribution weight ────────────────────────────────────────────────
    # Infer the motif's structural signature (edge-type pair or triple),
    # then look up the mean EAP attribution score for those edge types
    # in the positive circuits.  This converts "does the motif exist"
    # into "how strongly does the circuit express this motif".
    #
    # Mapping from motif name to the edge-type pair/triple it corresponds to:
    _MOTIF_EDGE_MAP: Dict[str, Any] = {
        "attn_to_attn":           ("attn", "attn"),
        "attn_feeds_mlp":         ("attn", "mlp"),
        "mlp_gates_attn":         ("mlp",  "attn"),
        "attn_chain_3":           ("attn", "attn", "attn"),
        "attn_mlp_attn_sandwich": ("attn", "mlp",  "attn"),
        "scaffold_event_entity":  ("attn", "attn", "mlp"),
    }
    edge_key = _MOTIF_EDGE_MAP.get(motif)
    attr_scores: List[float] = []
    for f in pos_features:
        if edge_key is None:
            break
        if len(edge_key) == 2:
            key_str = str(edge_key)
            v = f.get("pair_mean_score", {}).get(key_str, 0.0)
        else:
            key_str = str(edge_key)
            v = f.get("path3_mean_score", {}).get(key_str, 0.0)
        if v > 0.0:
            attr_scores.append(v)
    attr_weight = sum(attr_scores) / len(attr_scores) if attr_scores else 0.0

    # Normalise attr_weight to [0,1] relative to max observed across positives
    # (so it acts as a multiplier rather than an absolute scale factor)
    attr_ig = ig * (1.0 + attr_weight)   # attr_weight boosts IG, never zeroes it

    return {
        "ig":          round(ig,          6),
        "gain_ratio":  round(gain_ratio,  6),
        "gini_drop":   round(gini_drop,   6),
        "attr_weight": round(attr_weight, 6),
        "attr_ig":     round(attr_ig,     6),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _select_discriminative_motifs(
        pos_features: List[Dict],
        neg_features: List[Dict],
        max_motifs:   int = 8,
        metric:       str = "attr_ig",   # primary sort key
) -> List[Tuple[str, float]]:
    """
    Rank motifs observed in positives by a discriminative metric.

    metric choices:
        "ig"         — standard information gain  (classic, biased for rare motifs)
        "gain_ratio" — IG / split-entropy         (corrects rarity bias)
        "gini_drop"  — Gini impurity reduction    (similar to IG, less skew-sensitive)
        "attr_ig"    — attribution-weighted IG    (default: boosts structurally
                        important motifs by their mean EAP edge score)

    Returns sorted list of (motif_name, primary_score), highest first.
    Full score breakdown stored in motif_score_details (accessible via caller).
    """
    all_pos_motifs: Set[str] = set()
    for f in pos_features:
        all_pos_motifs.update(f.get("motifs", []))

    ranked = []
    for motif in all_pos_motifs:
        scores = _motif_scores(motif, pos_features, neg_features)
        primary = scores.get(metric, scores["ig"])
        if primary > 0.0:
            ranked.append((motif, primary, scores))

    ranked.sort(key=lambda x: -x[1])

    # Return (name, score) pairs for backward compat; full breakdown attached
    return [(m, s) for m, s, _ in ranked[:max_motifs]]


def _motif_score_table(
        pos_features: List[Dict],
        neg_features: List[Dict],
) -> List[Dict[str, Any]]:
    """
    Full scoring table for all motifs: all metrics side-by-side.
    Used for printing and paper tables.
    """
    all_pos_motifs: Set[str] = set()
    for f in pos_features:
        all_pos_motifs.update(f.get("motifs", []))

    rows = []
    for motif in sorted(all_pos_motifs):
        s = _motif_scores(motif, pos_features, neg_features)
        rows.append({"motif": motif, **s})

    rows.sort(key=lambda r: -r["attr_ig"])
    return rows


# ══════════════════════════════════════════════════════════════════
# 4.  BACKGROUND KNOWLEDGE BUILDER  (per-node facts, paper App D)
# ══════════════════════════════════════════════════════════════════

def _build_arch_background(circuit_id: str,
                            features: Dict[str, Any],
                            nodes: Optional[List[Dict]] = None,
                            edges: Optional[List[Dict]] = None) -> str:
    """
    Build B_arch: full structural Prolog facts for one circuit.

    Includes per-node facts matching paper Appendix D:
        node(C, v), type(v, tau), layer(v, l)
        edge(C, u, v)
        layer_span(C, lmin, lmax)
        has_motif(C, m)
        component_ratio(C, tau, r)
        size(C, n)
        rel_size(C, r)           [relative to model]
        density(C, d)            [edges/nodes]
        skip_connections(C, k)
        hub_count(C, k)
    """
    facts = []
    cid   = circuit_id
    n     = features["n_layers"]

    # ── Aggregate predicates ──────────────────────────────────────────────
    facts.append(f"size({cid}, {features['num_nodes']}).")
    facts.append(f"num_components({cid}, attention, {features['num_attn']}).")
    facts.append(f"num_components({cid}, mlp, {features['num_mlp']}).")
    lmin = round(features["depth_min"]  * n)
    lmax = round(features["depth_max"]  * n)
    facts.append(f"layer_span({cid}, {lmin}, {lmax}).")
    r = round(features["component_ratio_float"], 2)
    facts.append(f"component_ratio({cid}, attention, {r}).")
    facts.append(f"component_ratio({cid}, mlp, {round(1 - r, 2)}).")
    facts.append(f"rel_size({cid}, {features['rel_size']}).")
    facts.append(f"density({cid}, {features['density']}).")
    facts.append(f"skip_connections({cid}, {features['skip_connections']}).")
    facts.append(f"hub_count({cid}, {features['hub_count']}).")

    for m in features.get("motifs", []):
        facts.append(f"has_motif({cid}, {m}).")

    # ── Per-node facts (paper Appendix D: node/type/layer) ────────────────
    if nodes:
        facts.append("% per-node facts")
        for i, node in enumerate(nodes):
            name  = re.sub(r"[^a-zA-Z0-9_]", "_", node.get("name", f"v{i}"))
            ntype = node.get("type", "unknown")
            layer = node.get("layer", 0)
            facts.append(f"node({cid}, {name}).")
            facts.append(f"type({name}, {ntype}).")
            facts.append(f"layer({name}, {layer}).")

    # ── Per-edge facts ─────────────────────────────────────────────────────
    if edges:
        facts.append("% per-edge facts")
        for i, edge in enumerate(edges):
            s = re.sub(r"[^a-zA-Z0-9_]", "_", edge.get("source", f"s{i}"))
            t = re.sub(r"[^a-zA-Z0-9_]", "_", edge.get("target", f"t{i}"))
            facts.append(f"edge({cid}, {s}, {t}).")

    return "\n".join(facts)



# ══════════════════════════════════════════════════════════════════
# 4b. BLIND STRUCTURAL CLAUSE GENERATION
#     ILP searches over raw graph primitives (edge, type, layer)
#     without seeing named motif labels.  Best clause may or may not
#     correspond to a named motif — that is determined post-hoc.
# ══════════════════════════════════════════════════════════════════

def _generate_blind_structural_clauses(
        target_type:  str,
        pos_raw:      List[Tuple[List[Dict], List[Dict]]],
        neg_raw:      List[Tuple[List[Dict], List[Dict]]],
        pos_features: List[Dict],
) -> List[str]:
    """
    Generate candidate clauses that reference raw graph facts
    (edge, type, layer, component_ratio, layer_span) directly —
    no named motif labels involved.

    ILP is free to discover any edge-topology pattern that
    discriminates positives from negatives.  The vocabulary is:

        edge(C, U, V)           — directed edge in circuit
        type(U, attn|mlp)       — component type
        layer(U, L)             — absolute layer index
        component_ratio(C, attention, R), R > thresh
        layer_span(C, Lmin, Lmax), Lmin >= k, Lmax =< k
        skip_connections(C, K), K > 0
        hub_count(C, K), K >= 1

    Clauses are generated by:
      (a) enumerating edge-type pairs/triples observed in positives
      (b) combining with ratio/span constraints drawn from pos_features

    This is "blind" because neither the generator nor the evaluator
    looks at has_motif facts — they work purely on graph structure.
    Named motifs are compared post-hoc in _match_blind_to_named_motif.
    """
    t          = target_type
    hypotheses = []

    if not pos_raw:
        return hypotheses

    # ── Aggregate bounds from positive features ───────────────────────────
    if pos_features:
        n          = pos_features[0].get("n_layers", 6) or 6
        lmin_all   = min(round(f["depth_min"] * f["n_layers"]) for f in pos_features)
        lmax_all   = max(round(f["depth_max"] * f["n_layers"]) for f in pos_features)
        mean_ratio = (sum(f["component_ratio_float"] for f in pos_features)
                      / len(pos_features))
        thresh     = round(max(0.0, mean_ratio - 0.10), 2)
        max_rel    = round(max(f["rel_size"] for f in pos_features) + 0.05, 3)
    else:
        lmin_all, lmax_all, thresh, max_rel = 0, 6, 0.5, 0.5

    span_constraint = (f"layer_span(C, Lmin, Lmax), "
                       f"Lmin >= {max(0, lmin_all - 1)}, "
                       f"Lmax =< {lmax_all + 1}")

    # ── Collect edge-type pair and triple patterns from positives ─────────
    # For each positive circuit, build a set of (src_type, tgt_type) pairs
    # and (src, mid, tgt) triples observed in the graph.
    pos_pair_sets:   List[Set[Tuple]] = []
    pos_triple_sets: List[Set[Tuple]] = []

    for nodes, edges in pos_raw:
        node_types = {n.get("name", ""): n.get("type", "attn") for n in nodes}
        pairs:   Set[Tuple] = set()
        triples: Set[Tuple] = set()
        # Build adjacency for triple detection
        adj: Dict[str, List[str]] = {}
        for e in edges:
            s = e.get("source", ""); tgt = e.get("target", "")
            adj.setdefault(s, []).append(tgt)
            st = node_types.get(s, "attn"); tt = node_types.get(tgt, "attn")
            pairs.add((st, tt))
        for u, vs in adj.items():
            ut = node_types.get(u, "attn")
            for v in vs:
                vt = node_types.get(v, "attn")
                for w in adj.get(v, []):
                    wt = node_types.get(w, "attn")
                    triples.add((ut, vt, wt))
        pos_pair_sets.append(pairs)
        pos_triple_sets.append(triples)

    # Only use patterns present in ALL positives (strongest signal)
    if pos_pair_sets:
        common_pairs   = pos_pair_sets[0].copy()
        common_triples = pos_triple_sets[0].copy()
        for ps in pos_pair_sets[1:]:
            common_pairs   &= ps
        for ts in pos_triple_sets[1:]:
            common_triples &= ts
    else:
        common_pairs, common_triples = set(), set()

    # ── Blind clause templates ────────────────────────────────────────────

    # B1: pure ratio (no edge topology)
    hypotheses.append(
        f"arch_{t}(C) :- component_ratio(C, attention, R), R > {thresh}."
    )

    # B2: ratio + span
    hypotheses.append(
        f"arch_{t}(C) :- {span_constraint}, "
        f"component_ratio(C, attention, R), R > {thresh}."
    )

    # B3: rel_size + ratio (model-relative)
    hypotheses.append(
        f"arch_{t}(C) :- rel_size(C, S), S =< {max_rel}, "
        f"component_ratio(C, attention, R), R > {thresh}."
    )

    # B4: one clause per common edge-type pair
    for (st, tt) in sorted(common_pairs):
        hypotheses.append(
            f"arch_{t}(C) :- "
            f"edge(C, U, V), type(U, {st}), type(V, {tt}), "
            f"{span_constraint}."
        )
        # Also with ratio constraint
        hypotheses.append(
            f"arch_{t}(C) :- "
            f"edge(C, U, V), type(U, {st}), type(V, {tt}), "
            f"component_ratio(C, attention, R), R > {thresh}."
        )

    # B5: one clause per common edge-type triple (3-node path)
    for (ut, vt, wt) in sorted(common_triples):
        hypotheses.append(
            f"arch_{t}(C) :- "
            f"edge(C, U, V), edge(C, V, W), "
            f"type(U, {ut}), type(V, {vt}), type(W, {wt}), "
            f"{span_constraint}."
        )

    # B6: connectivity
    hypotheses.append(
        f"arch_{t}(C) :- skip_connections(C, K), K > 0, "
        f"component_ratio(C, attention, R), R > {thresh}."
    )
    hypotheses.append(
        f"arch_{t}(C) :- hub_count(C, K), K >= 1, "
        f"{span_constraint}."
    )

    return hypotheses


# Named-motif pattern definitions — used ONLY for post-hoc comparison.
# These are never shown to the ILP learner during blind clause generation.
_BLIND_MOTIF_PATTERNS: Dict[str, str] = {
    # key = motif name, value = structural signature for matching
    "attn_to_attn":           "edge(C,U,V),type(U,attn),type(V,attn)",
    "attn_feeds_mlp":         "edge(C,U,V),type(U,attn),type(V,mlp)",
    "mlp_gates_attn":         "edge(C,U,V),type(U,mlp),type(V,attn)",
    "attn_chain_3":           "edge(C,U,V),edge(C,V,W),type(U,attn),type(V,attn),type(W,attn)",
    "attn_mlp_attn_sandwich": "edge(C,U,V),edge(C,V,W),type(U,attn),type(V,mlp),type(W,attn)",
    "scaffold_event_entity":  "edge(C,U,V),edge(C,V,W),type(U,attn),type(V,attn),type(W,mlp)",
}


def _match_blind_to_named_motif(clause: str) -> Optional[str]:
    """
    Post-hoc: check whether a blind structural clause matches a named motif.

    Returns the motif name if matched, None if the clause is a genuinely
    novel pattern not in the hand-coded vocabulary.

    Matching is purely syntactic — normalise whitespace, strip variable
    names (U/V/W are arbitrary), and compare edge-type skeletons.
    """
    # Normalise: lowercase, collapse whitespace, remove variable names
    def _skeleton(text: str) -> str:
        text = re.sub(r"\s+", "", text.lower())
        # Replace variable names (single uppercase letter or Lmin/Lmax etc)
        # with a placeholder so U,V,W don't matter
        text = re.sub(r"\b[A-Z][a-zA-Z0-9]*\b", "?", text)
        return text

    if ":-" not in clause:
        return None
    _, body = clause.split(":-", 1)
    body_skel = _skeleton(body.rstrip("."))

    for motif_name, pattern in _BLIND_MOTIF_PATTERNS.items():
        pat_skel = _skeleton(pattern)
        # Check if all atoms of the pattern appear in the body skeleton
        pat_atoms = set(pat_skel.split(","))
        body_atoms = set(body_skel.split(","))
        if pat_atoms.issubset(body_atoms):
            return motif_name

    return None  # novel pattern — not in named vocabulary

# ══════════════════════════════════════════════════════════════════
# 5.  HYPOTHESIS GENERATION (paper Eq.4 template family)
# ══════════════════════════════════════════════════════════════════

def _generate_arch_hypotheses(target_type: str,
                               pos_features: List[Dict],
                               neg_features: List[Dict]) -> List[str]:
    """
    Generate candidate arch_<type>(C) hypotheses.

    Uses discriminative motifs (ranked by information gain) and
    relative feature bounds drawn from positive circuits.

    Template family matches paper Eq.4 / Appendix D:
        arch_<t>(C) :- has_motif(C, m),
                        layer_span(C, Lmin, Lmax), Lmin >= k1, Lmax =< k2,
                        component_ratio(C, attention, R), R > thresh,
                        size(C, N), N < max.
    """
    t = target_type
    hypotheses: List[str] = []

    if not pos_features:
        return [f"arch_{t}(C) :- size(C, N), N > 0."]

    # ── Bounds from positive circuits ──────────────────────────────────────
    n = pos_features[0]["n_layers"]  # assume same model
    lmin_all   = min(round(f["depth_min"]  * f["n_layers"]) for f in pos_features)
    lmax_all   = max(round(f["depth_max"]  * f["n_layers"]) for f in pos_features)
    max_size   = max(f["num_nodes"] for f in pos_features)
    mean_ratio = sum(f["component_ratio_float"] for f in pos_features) / len(pos_features)
    neg_max_size = max((f["num_nodes"] for f in neg_features), default=max_size + 1)
    thresh     = round(max(0.0, mean_ratio - 0.10), 2)

    # Relative bounds (for rel_size predicate)
    max_rel    = round(max(f["rel_size"] for f in pos_features) + 0.05, 3)

    # ── Discriminative motifs ──────────────────────────────────────────────
    disc_motifs = _select_discriminative_motifs(pos_features, neg_features)
    # disc_motifs is [(motif_name, attr_ig_score), ...]
    top_motifs  = [m for m, _ in disc_motifs[:4]]

    # Common motifs (appear in ALL positives)
    if pos_features:
        common_motifs = set(pos_features[0].get("motifs", []))
        for f in pos_features[1:]:
            common_motifs &= set(f.get("motifs", []))
    else:
        common_motifs = set()

    # ── Template 0: most general (size only) ──────────────────────────────
    hypotheses.append(f"arch_{t}(C) :- size(C, N), N < {max_size + 5}.")

    # ── Template 1: layer_span + size ─────────────────────────────────────
    hypotheses.append(
        f"arch_{t}(C) :- layer_span(C, Lmin, Lmax), "
        f"Lmin >= {max(0, lmin_all - 1)}, Lmax =< {lmax_all + 1}, "
        f"size(C, N), N < {max_size + 5}."
    )

    # ── Template 2: component_ratio + layer_span ──────────────────────────
    hypotheses.append(
        f"arch_{t}(C) :- layer_span(C, Lmin, Lmax), "
        f"Lmin >= {max(0, lmin_all - 1)}, Lmax =< {lmax_all + 1}, "
        f"component_ratio(C, attention, R), R > {thresh}."
    )

    # ── Template 3: rel_size (relative to model, Eq.4 spirit) ─────────────
    hypotheses.append(
        f"arch_{t}(C) :- rel_size(C, S), S =< {max_rel}, "
        f"component_ratio(C, attention, R), R > {thresh}."
    )

    # ── Template 4+: one clause per discriminative motif (canonical Eq.4) ─
    for motif in top_motifs:
        hypotheses.append(
            f"arch_{t}(C) :- has_motif(C, {motif}), "
            f"layer_span(C, Lmin, Lmax), "
            f"Lmin >= {max(0, lmin_all - 1)}, Lmax =< {lmax_all + 1}, "
            f"component_ratio(C, attention, R), R > {thresh}, "
            f"size(C, N), N < {max_size + 5}."
        )

    # ── Template 5: common motifs only (if any) ────────────────────────────
    for motif in sorted(list(common_motifs)[:2]):
        hypotheses.append(
            f"arch_{t}(C) :- has_motif(C, {motif}), "
            f"component_ratio(C, attention, R), R > {thresh}."
        )

    # ── Template 6: skip-connections predicate ────────────────────────────
    if any(f.get("skip_connections", 0) > 0 for f in pos_features):
        hypotheses.append(
            f"arch_{t}(C) :- has_skip_connections(C), "
            f"size(C, N), N < {max_size + 5}."
        )

    # ── Template 7: hub-node predicate ────────────────────────────────────
    if any(f.get("hub_count", 0) > 0 for f in pos_features):
        hypotheses.append(
            f"arch_{t}(C) :- has_hub_node(C), "
            f"layer_span(C, Lmin, Lmax), "
            f"Lmin >= {max(0, lmin_all - 1)}, Lmax =< {lmax_all + 1}."
        )

    # ── Template 8: multi-motif conjunction (top 2 motifs) ────────────────
    if len(top_motifs) >= 2:
        m1, m2 = top_motifs[0], top_motifs[1]
        hypotheses.append(
            f"arch_{t}(C) :- has_motif(C, {m1}), has_motif(C, {m2}), "
            f"layer_span(C, Lmin, Lmax), "
            f"Lmin >= {max(0, lmin_all - 1)}, Lmax =< {lmax_all + 1}, "
            f"component_ratio(C, attention, R), R > {thresh}, "
            f"size(C, N), N < {max_size + 5}."
        )

    return hypotheses


def _generate_arch_hypotheses_with_blind(
        target_type:  str,
        pos_features: List[Dict],
        neg_features: List[Dict],
        pos_raw:      Optional[List[Tuple]] = None,
        neg_raw:      Optional[List[Tuple]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Returns (named_pool, blind_pool) — two separate hypothesis pools.

    Pool A (named):  clauses using has_motif labels  — interpretable, fixed vocab
    Pool B (blind):  clauses using edge/type/layer    — data-driven, no motif labels

    Both pools are evaluated on the same pos/neg split.  The winner is
    selected globally across both pools.  Post-hoc matching then checks
    whether the winning blind clause corresponds to a named motif.
    """
    named = _generate_arch_hypotheses(target_type, pos_features, neg_features)
    blind = _generate_blind_structural_clauses(
        target_type, pos_raw or [], neg_raw or [], pos_features
    )
    return named, blind


# ══════════════════════════════════════════════════════════════════
# 6.  CLAUSE EVALUATION (deterministic, consistent with Eq.2)
# ══════════════════════════════════════════════════════════════════

def _split_clause_body(body: str) -> List[str]:
    """
    Split a Prolog clause body on top-level commas only (not commas inside
    parentheses).  This is required because predicates like
    has_motif(C, mid_layer_attn_peak) and layer_span(C, Lmin, Lmax) contain
    commas inside their argument lists that must NOT be treated as separators.

    Examples
    --------
    "has_motif(C, foo), size(C, N), N < 5"
        → ["has_motif(C, foo)", "size(C, N)", "N < 5"]
    """
    tokens, depth, cur = [], 0, ""
    for ch in body:
        if ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            tok = cur.strip()
            if tok:
                tokens.append(tok)
            cur = ""
        else:
            cur += ch
    tok = cur.strip()
    if tok:
        tokens.append(tok)
    return tokens


def _parse_arch_conditions(clause: str) -> List[Dict[str, Any]]:
    """
    Parse a single arch_<t>(C) :- ... clause into condition dicts.

    Uses top-level comma splitting so predicates with multiple arguments
    (has_motif, layer_span, component_ratio) are correctly parsed.
    """
    if ":-" not in clause:
        return []
    _, body = clause.split(":-", 1)
    body    = body.strip().rstrip(".")
    tokens  = _split_clause_body(body)   # split on top-level commas only

    conditions: List[Dict] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # size(C, N)  followed by  N <op> V
        if re.match(r"size\s*\(", tok):
            i += 1
            if i < len(tokens):
                m = re.match(r"[NnSs]\s*([<>=]{1,2})\s*([\d.]+)", tokens[i])
                if m:
                    conditions.append({"kind": "size",
                                       "op": m.group(1), "val": float(m.group(2))})

        # layer_span(C, Lmin, Lmax)  followed by  Lmin >= k1, Lmax =< k2
        elif re.match(r"layer_span\s*\(", tok):
            i += 1
            while i < len(tokens):
                m = re.match(r"(Lmin|Lmax)\s*([<>=]{1,2})\s*([\d.]+)", tokens[i])
                if not m:
                    break
                var, op, val = m.group(1), m.group(2), float(m.group(3))
                if var == "Lmin":
                    conditions.append({"kind": "lmin", "op": ">=", "val": val})
                else:
                    conditions.append({"kind": "lmax", "op": "<=", "val": val})
                i += 1
            continue

        # component_ratio(C, attention, R)  followed by  R > V
        elif re.match(r"component_ratio\s*\(", tok) and "attention" in tok:
            i += 1
            if i < len(tokens):
                m = re.match(r"[Rr]\s*([<>=]{1,2})\s*([\d.]+)", tokens[i])
                if m:
                    conditions.append({"kind": "ratio_attn",
                                       "op": m.group(1), "val": float(m.group(2))})

        # rel_size(C, S)  followed by  S =< V
        elif re.match(r"rel_size\s*\(", tok):
            i += 1
            if i < len(tokens):
                m = re.match(r"[Ss]\s*([<>=]{1,2})\s*([\d.]+)", tokens[i])
                if m:
                    conditions.append({"kind": "rel_size",
                                       "op": m.group(1), "val": float(m.group(2))})

        # has_motif(C, motif_name)  — motif name is inside the parens, not split off
        elif re.match(r"has_motif\s*\(", tok):
            m = re.match(r"has_motif\s*\(\s*\w+\s*,\s*(\w+)\s*\)", tok)
            if m:
                conditions.append({"kind": "motif", "motif": m.group(1)})

        # has_skip_connections(C)
        elif "has_skip_connections" in tok:
            conditions.append({"kind": "skip_present"})

        # has_hub_node(C)
        elif "has_hub_node" in tok:
            conditions.append({"kind": "hub_present"})

        # edge(C, U, V)  — blind structural clause atom
        elif re.match(r"edge\s*\(", tok):
            m = re.match(r"edge\s*\(\s*\w+\s*,\s*(\w+)\s*,\s*(\w+)\s*\)", tok)
            if m:
                conditions.append({"kind": "edge",
                                   "src_var": m.group(1), "tgt_var": m.group(2)})

        # type(V, attn|mlp)  — bind variable to node type
        elif re.match(r"type\s*\(", tok):
            m = re.match(r"type\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)", tok)
            if m:
                conditions.append({"kind": "node_type",
                                   "var": m.group(1), "ntype": m.group(2)})

        # layer(V, L)  — bind variable to layer index
        elif re.match(r"layer\s*\(", tok):
            m = re.match(r"layer\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)", tok)
            if m:
                conditions.append({"kind": "node_layer", "var": m.group(1),
                                   "layer_var": m.group(2)})

        # skip_connections(C, K), K > 0
        elif re.match(r"skip_connections\s*\(", tok):
            pass   # consumed by the K > 0 comparison token that follows

        # hub_count(C, K), K >= 1
        elif re.match(r"hub_count\s*\(", tok):
            pass   # consumed by K >= 1 comparison token that follows

        # K > 0   (after skip_connections)
        elif re.match(r"K\s*>\s*0", tok):
            conditions.append({"kind": "skip_gt0"})

        # K >= 1  (after hub_count)
        elif re.match(r"K\s*>=\s*1", tok):
            conditions.append({"kind": "hub_ge1"})

        i += 1

    return conditions


_OPS = {
    "<":  lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "=<": lambda a, b: a <= b,
    "=>": lambda a, b: a >= b,
}


def _evaluate_arch_clause(
        clause:   str,
        features: Dict[str, Any],
        raw:      Optional[Tuple[List[Dict], List[Dict]]] = None,
) -> bool:
    """
    Return True if the circuit satisfies all conditions in clause.

    Parameters
    ----------
    clause   : Prolog clause string
    features : aggregate feature dict (from _features_from_circuit_dict)
    raw      : (nodes, edges) lists — required for blind edge/type conditions.
               If None, blind conditions that need graph topology return False
               (conservative: blind clause fails without graph data).
    """
    conditions = _parse_arch_conditions(clause)
    if not conditions:
        return True

    n      = features.get("n_layers", 6) or 6
    lmin_i = round(features.get("depth_min", 0.0) * n)
    lmax_i = round(features.get("depth_max", 1.0) * n)

    # Build graph structures lazily (only when a blind condition is encountered)
    _graph_built = False
    _node_type:   Dict[str, str] = {}
    _adj:         Dict[str, List[str]] = {}  # src -> [tgt, ...]

    def _ensure_graph():
        nonlocal _graph_built
        if _graph_built:
            return
        if raw:
            nodes, edges = raw
            for node in nodes:
                _node_type[node.get("name", "")] = node.get("type", "attn")
            for edge in edges:
                s = edge.get("source", ""); t = edge.get("target", "")
                _adj.setdefault(s, []).append(t)
        _graph_built = True

    # Variable bindings for blind clause evaluation (U, V, W etc.)
    # We use a simple forward-chaining interpreter: iterate over edges
    # and check type/layer constraints per binding.
    _bindings: Dict[str, Any] = {}

    for cond in conditions:
        k = cond["kind"]
        op_fn = _OPS.get(cond.get("op", "<"), lambda *_: True)

        # ── Named-motif conditions ────────────────────────────────────────
        if k == "size":
            if not op_fn(features.get("num_nodes", 0), cond["val"]):
                return False
        elif k == "lmin":
            if not (lmin_i >= cond["val"]):
                return False
        elif k == "lmax":
            if not (lmax_i <= cond["val"]):
                return False
        elif k == "ratio_attn":
            if not op_fn(features.get("component_ratio_float", 0.0), cond["val"]):
                return False
        elif k == "rel_size":
            if not op_fn(features.get("rel_size", 0.0), cond["val"]):
                return False
        elif k == "motif":
            if cond["motif"] not in features.get("motifs", []):
                return False
        elif k == "skip_present":
            if features.get("skip_connections", 0) == 0:
                return False
        elif k == "hub_present":
            if features.get("hub_count", 0) == 0:
                return False
        elif k == "skip_gt0":
            if features.get("skip_connections", 0) == 0:
                return False
        elif k == "hub_ge1":
            if features.get("hub_count", 0) < 1:
                return False

        # ── Blind structural conditions ───────────────────────────────────
        # Evaluated by checking existence of a satisfying variable binding
        # in the actual graph.  All edge/type/layer conditions in the clause
        # are collected and checked together below.
        elif k in ("edge", "node_type", "node_layer"):
            pass   # deferred — evaluated in one pass below

    # ── Blind graph evaluation (deferred pass) ────────────────────────────
    blind_conds = [c for c in conditions
                   if c["kind"] in ("edge", "node_type", "node_layer")]
    if not blind_conds:
        return True   # no blind conditions — already fully evaluated above

    _ensure_graph()
    if not raw:
        return False  # blind conditions require graph data

    nodes, edges = raw

    # Collect required edge patterns and type constraints
    # edge(C,U,V) establishes a binding U→src, V→tgt
    # type(U, attn) constrains the type of the node bound to U
    # We check: does there exist an assignment of variables to nodes
    # satisfying all edge and type constraints simultaneously?

    # Extract edge variable pairs and type constraints
    edge_conds  = [c for c in blind_conds if c["kind"] == "edge"]
    type_conds  = {c["var"]: c["ntype"] for c in blind_conds
                   if c["kind"] == "node_type"}

    if not edge_conds:
        return True   # only layer conditions, skip

    # For 1-edge clauses: try all edges
    if len(edge_conds) == 1:
        ec = edge_conds[0]
        u_var, v_var = ec["src_var"], ec["tgt_var"]
        u_type = type_conds.get(u_var)
        v_type = type_conds.get(v_var)
        for e in edges:
            s = e.get("source", ""); t = e.get("target", "")
            if u_type and _node_type.get(s) != u_type:
                continue
            if v_type and _node_type.get(t) != v_type:
                continue
            return True
        return False

    # For 2-edge clauses (triple path): share a variable (V in U→V→W)
    if len(edge_conds) == 2:
        ec1, ec2 = edge_conds[0], edge_conds[1]
        # Detect shared variable (the middle node)
        if ec1["tgt_var"] == ec2["src_var"]:
            u_var, v_var, w_var = ec1["src_var"], ec1["tgt_var"], ec2["tgt_var"]
        elif ec1["src_var"] == ec2["tgt_var"]:
            u_var, v_var, w_var = ec2["src_var"], ec1["src_var"], ec1["tgt_var"]
        else:
            # No shared variable — treat as two independent edges
            return True   # conservative: pass
        u_type = type_conds.get(u_var)
        v_type = type_conds.get(v_var)
        w_type = type_conds.get(w_var)
        for e1 in edges:
            u = e1.get("source", ""); v = e1.get("target", "")
            if u_type and _node_type.get(u) != u_type: continue
            if v_type and _node_type.get(v) != v_type: continue
            for w in _adj.get(v, []):
                if w_type and _node_type.get(w) != w_type: continue
                return True
        return False

    return True   # >2 edge conditions: conservative pass


def _score_clause(clause: str,
                  pos_features: List[Dict],
                  neg_features: List[Dict],
                  pos_raw: Optional[List[Tuple]] = None,
                  neg_raw: Optional[List[Tuple]] = None,
                  ) -> Tuple[float, float, float]:
    """coverage, precision, F1 for a hypothesis clause (paper B.3).

    pos_raw / neg_raw are (nodes, edges) tuples required for blind
    structural clauses that reference edge/type/layer directly.
    """
    tp = sum(
        1 for i, f in enumerate(pos_features)
        if _evaluate_arch_clause(clause, f,
                                  raw=pos_raw[i] if pos_raw else None)
    )
    fp = sum(
        1 for i, f in enumerate(neg_features)
        if _evaluate_arch_clause(clause, f,
                                  raw=neg_raw[i] if neg_raw else None)
    )
    coverage  = tp / len(pos_features) if pos_features else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * coverage * precision / (coverage + precision)
          if (coverage + precision) else 0.0)
    return coverage, precision, f1


# ══════════════════════════════════════════════════════════════════
# 7.  ILP SIGNATURE (Eq.20) + CONFIDENCE (Eq.2)
# ══════════════════════════════════════════════════════════════════

def _arch_signature(clause: str,
                    motifs: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    S_ILP_arch = <n_clauses, P_struct, motifs, complexity> (paper Eq.20).

    Motifs are included in the signature because they are the primary
    discriminator between circuit types — the predicate set alone is
    insufficient when all clauses use the same structural predicate (e.g.
    rel_size) but represent genuinely different circuit topologies.
    """
    predicates: List[str] = []
    if ":-" in clause:
        _, body = clause.split(":-", 1)
        predicates = sorted(set(re.findall(r"([a-z_]\w*)\(", body)))

    body_part  = clause.split(":-")[1].rstrip(".") if ":-" in clause else ""
    # Count literals by splitting on commas at the top level (depth 0),
    # not commas inside predicate arguments.
    n_literals = 0
    paren_depth = 0
    for ch in body_part:
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        elif ch == "," and paren_depth == 0:
            n_literals += 1
    n_literals += 1 if body_part.strip() else 0  # N commas → N+1 literals
    depth = 0; cur = 0
    for ch in body_part:
        if ch == "(":
            cur += 1; depth = max(depth, cur)
        elif ch == ")":
            cur -= 1
    depth = max(1, depth)
    complexity = n_literals * depth / 10.0

    return {
        "n_clauses":  1,
        "predicates": predicates,
        "motifs":     sorted(set(motifs or [])),   # topology discriminator
        "avg_length": n_literals,
        "avg_depth":  depth,
        "complexity": complexity,
    }


def _arch_confidence(clause: str,
                     pos_features: List[Dict],
                     neg_features: List[Dict]) -> float:
    """confidence = tp/|Ep| (Eq.2) with soft consistency penalty (Eq.15)."""
    if not pos_features:
        return 0.0
    tp = sum(1 for f in pos_features if _evaluate_arch_clause(clause, f))
    fp = sum(1 for f in neg_features if _evaluate_arch_clause(clause, f))

    conf = tp / len(pos_features)
    if neg_features and fp > 0:
        fn_rate = fp / len(neg_features)
        conf   *= (1.0 - fn_rate)
    return max(0.0, min(1.0, conf))


# ══════════════════════════════════════════════════════════════════
# 8.  MAIN LEARNER CLASS
# ══════════════════════════════════════════════════════════════════

class ArchitecturalILPLearner:
    """
    Full ILP_arch learner matching paper §3.3 + Appendix D.

    Motifs are inferred from actual graph topology (not hard-coded).
    All features are relative to model size.
    Discriminative motif selection uses information gain before
    hypothesis generation.

    Usage:
        learner = ArchitecturalILPLearner()
        result  = learner.learn_from_json_records(
                      target_type="ioi",
                      positive_records=[ioi_json],
                      negative_records=[location_json])
    """

    def learn_from_json_records(
            self,
            target_type:      str,
            positive_records: List[Dict[str, Any]],
            negative_records: List[Dict[str, Any]],
            model_n_layers:   int = 0,
            model_n_heads:    int = 0,
    ) -> Dict[str, Any]:
        """Learn from task JSON records (output of results_export.py)."""
        print(f"\n{'='*60}")
        print(f"ILP_arch: arch_{target_type}")
        print(f"{'='*60}")
        print(f"  Positive : {len(positive_records)}   Negative : {len(negative_records)}")

        def _extract(rec):
            c = rec.get("layer1", {}).get("circuit", {})
            return _features_from_circuit_dict(
                c, model_n_layers=model_n_layers, model_n_heads=model_n_heads)

        # Keep raw nodes/edges for B_arch per-node facts
        def _raw(rec):
            c = rec.get("layer1", {}).get("circuit", {})
            return c.get("nodes", []), c.get("edges", [])

        pos_features = [_extract(r) for r in positive_records]
        neg_features = [_extract(r) for r in negative_records]
        pos_raw      = [_raw(r)     for r in positive_records]
        neg_raw      = [_raw(r)     for r in negative_records]

        return self._learn(target_type, pos_features, neg_features,
                           pos_raw, neg_raw)

    def learn_from_graphs(
            self,
            target_type:     str,
            positive_graphs: List[Graph],
            negative_graphs: List[Graph],
            model_n_layers:  int = 0,
    ) -> Dict[str, Any]:
        """Learn from rdflib CKG Graph objects."""
        print(f"\n{'='*60}")
        print(f"ILP_arch: arch_{target_type}")
        print(f"{'='*60}")
        print(f"  Positive : {len(positive_graphs)}   Negative : {len(negative_graphs)}")

        def _get_uri(g: Graph) -> URIRef:
            res = list(g.query(
                "SELECT ?c WHERE { ?c a <http://purl.org/mechint#Circuit> . } LIMIT 1"))
            if res:
                return res[0][0]
            raise ValueError("No circuit found in graph")

        pos_features = [_features_from_rdf_graph(g, _get_uri(g), model_n_layers)
                        for g in positive_graphs]
        neg_features = [_features_from_rdf_graph(g, _get_uri(g), model_n_layers)
                        for g in negative_graphs]

        return self._learn(target_type, pos_features, neg_features)

    def extract_features_from_json(self, record: Dict[str, Any],
                                    model_n_layers: int = 0) -> Dict[str, Any]:
        c = record.get("layer1", {}).get("circuit", {})
        return _features_from_circuit_dict(c, model_n_layers=model_n_layers)

    def extract_features_from_graph(self, graph: Graph,
                                     model_n_layers: int = 0) -> Dict[str, Any]:
        res = list(graph.query(
            "SELECT ?c WHERE { ?c a <http://purl.org/mechint#Circuit> . } LIMIT 1"))
        if not res:
            raise ValueError("No circuit in graph")
        return _features_from_rdf_graph(graph, res[0][0], model_n_layers)

    # ── Core learning ─────────────────────────────────────────────────────

    def _select_best_clause(
            self,
            target_type:  str,
            pos_features: List[Dict],
            neg_features: List[Dict],
            pos_raw: Optional[List[Tuple]] = None,
            neg_raw: Optional[List[Tuple]] = None,
    ) -> Tuple[str, str, List[Tuple[float, int, str, str]]]:
        """
        Hypothesis generation + F1 scoring + winner selection (quiet).

        Returns (best_clause, best_pool, all_scored) where all_scored is a
        list of (f1, n_literals, clause, pool) sorted best-first. Shared by
        _learn() and the leave-one-out evaluation so both use the exact
        same selection procedure.
        """
        named_pool, blind_pool = _generate_arch_hypotheses_with_blind(
            target_type, pos_features, neg_features, pos_raw, neg_raw)

        # Each entry: (f1, n_literals, clause, pool)
        # n_literals used as tie-breaker: simpler clause preferred (Occam)
        def _n_lits(clause: str) -> int:
            if ":-" not in clause:
                return 1
            _, body = clause.split(":-", 1)
            return len(_split_clause_body(body.rstrip(".")))

        all_scored: List[Tuple[float, int, str, str]] = []

        for h in named_pool:
            cov, prec, f1 = _score_clause(h, pos_features, neg_features)
            all_scored.append((f1, _n_lits(h), h, "named"))

        for h in blind_pool:
            cov, prec, f1 = _score_clause(
                h, pos_features, neg_features, pos_raw, neg_raw)
            all_scored.append((f1, _n_lits(h), h, "blind"))

        # Sort: highest F1, then fewest literals (Occam's razor tie-break)
        all_scored.sort(key=lambda x: (-x[0], x[1]))

        if not all_scored:
            return (f"arch_{target_type}(C) :- size(C, N), N > 0.",
                    "fallback", all_scored)
        _, _, best_clause, best_pool = all_scored[0]
        return best_clause, best_pool, all_scored

    def _evaluate_clause_loo(
            self,
            target_type:  str,
            pos_features: List[Dict],
            neg_features: List[Dict],
            pos_raw: Optional[List[Tuple]] = None,
            neg_raw: Optional[List[Tuple]] = None,
    ) -> Optional[float]:
        """
        Leave-one-out F1 over the positive circuits.

        For each held-out positive, the clause is re-learned from the
        remaining positives and ALL negatives via the same selection
        procedure as _learn(), then tested on the held-out positive and all
        negatives. TP/FN/FP/TN are pooled across folds and a single F1 is
        returned (out-of-sample estimate).

        Returns None when there are fewer than 3 positives — with a single
        positive the in-sample F1 is a consistency check, not a
        generalisation estimate.
        """
        n = len(pos_features)
        if n < 3:
            print(f"  [LOO] only {n} positive circuit(s) — in-sample F1 is a "
                  f"consistency check only; loo_f1 = None")
            return None

        tp = fn = fp = 0
        for i in range(n):
            train_pos = pos_features[:i] + pos_features[i+1:]
            train_raw = ([r for j, r in enumerate(pos_raw) if j != i]
                         if pos_raw else None)
            clause_i, _, _ = self._select_best_clause(
                target_type, train_pos, neg_features, train_raw, neg_raw)

            held_raw = pos_raw[i] if pos_raw else None
            if _evaluate_arch_clause(clause_i, pos_features[i], raw=held_raw):
                tp += 1
            else:
                fn += 1
            for j, nf in enumerate(neg_features):
                if _evaluate_arch_clause(clause_i, nf,
                                         raw=neg_raw[j] if neg_raw else None):
                    fp += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        return (2 * precision * recall / (precision + recall)
                if (precision + recall) else 0.0)

    def _learn(self,
               target_type:  str,
               pos_features: List[Dict],
               neg_features: List[Dict],
               pos_raw: Optional[List[Tuple]] = None,
               neg_raw: Optional[List[Tuple]] = None,
               ) -> Dict[str, Any]:

        # 1. Discriminative motifs — full metric table
        score_table = _motif_score_table(pos_features, neg_features)
        disc_motifs = _select_discriminative_motifs(pos_features, neg_features,
                                                     metric="attr_ig")
        print(f"\n  Discriminative motifs (attr_ig ↓ = attribution-weighted IG):")
        print(f"    {'motif':<35} {'IG':>6}  {'GainR':>6}  {'Gini':>6}  "
              f"{'AttrW':>6}  {'AttrIG':>7}")
        print(f"    {'─'*35} {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}")
        for row in score_table:
            print(f"    {row['motif']:<35} "
                  f"{row['ig']:>6.4f}  "
                  f"{row['gain_ratio']:>6.4f}  "
                  f"{row['gini_drop']:>6.4f}  "
                  f"{row['attr_weight']:>6.4f}  "
                  f"{row['attr_ig']:>7.4f}")

        # 2+3. Generate hypotheses (Pool A named + Pool B blind), score both
        # pools with the shared selection procedure
        _, _, all_scored = self._select_best_clause(
            target_type, pos_features, neg_features, pos_raw, neg_raw)
        n_named = sum(1 for *_r, p in all_scored if p == "named")
        n_blind = sum(1 for *_r, p in all_scored if p == "blind")
        print(f"\n  Generated {n_named} named-motif hypotheses  "
              f"(Pool A)")
        print(f"  Generated {n_blind} blind structural hypotheses "
              f"(Pool B)")

        # ── Report full scoring table ────────────────────────────────────
        print(f"\n  Hypothesis scores (F1 ↓, literals ↑ as tie-break):")
        print(f"    {'F1':>5}  {'lits':>4}  {'pool':6}  clause")
        print(f"    {'─'*5}  {'─'*4}  {'─'*6}  {'─'*50}")
        for f1_i, nl_i, h_i, pool_i in all_scored:
            h_short = h_i[:55] + "…" if len(h_i) > 55 else h_i
            bmatch  = (_match_blind_to_named_motif(h_i)
                       if pool_i == "blind" else "")
            tag     = f"  [{bmatch}]" if bmatch else (
                      "  [NOVEL]" if pool_i == "blind" else "")
            print(f"    {f1_i:5.3f}  {nl_i:4d}  {pool_i:6}  {h_short}{tag}")

        # ── Select winner ────────────────────────────────────────────────
        if not all_scored:
            best_clause = f"arch_{target_type}(C) :- size(C, N), N > 0."
            best_pool   = "fallback"
            best_f1 = best_cov = best_prec = 0.0
        else:
            best_f1_val, _, best_clause, best_pool = all_scored[0]
            best_f1  = best_f1_val
            # Recompute cov/prec for the winner
            if best_pool == "named":
                best_cov, best_prec, _ = _score_clause(
                    best_clause, pos_features, neg_features)
            else:
                best_cov, best_prec, _ = _score_clause(
                    best_clause, pos_features, neg_features, pos_raw, neg_raw)

        # ── Blind pool summary ───────────────────────────────────────────
        blind_scored = [(f1_i, nl_i, h_i) for f1_i, nl_i, h_i, p in all_scored
                        if p == "blind"]
        best_blind_f1 = blind_scored[0][0] if blind_scored else 0.0
        best_named_f1 = max((f1_i for f1_i, _, _, p in all_scored if p == "named"),
                            default=0.0)

        print(f"\n  Pool summary:")
        print(f"    Pool A best F1 = {best_named_f1:.3f}  (named motifs)")
        print(f"    Pool B best F1 = {best_blind_f1:.3f}  (blind structural)")
        if best_blind_f1 > best_named_f1:
            print(f"    → ILP learned structure BEYOND named vocabulary")
        elif abs(best_blind_f1 - best_named_f1) < 0.001 and blind_scored:
            print(f"    → Blind clauses matched named performance — "
                  f"ILP independently confirmed structure")
        else:
            print(f"    → Named motifs sufficient — blind search found nothing new")

        # ── Post-hoc match for winner ────────────────────────────────────
        blind_motif_match: Optional[str] = None
        if best_pool == "blind":
            blind_motif_match = _match_blind_to_named_motif(best_clause)

        print(f"\n  Winner  F1={best_f1:.3f}  "
              f"cov={best_cov:.3f}  prec={best_prec:.3f}  "
              f"[pool={best_pool}]")
        print(f"  {best_clause}")
        if best_pool == "blind":
            if blind_motif_match:
                print(f"  ↳ matches named motif: {blind_motif_match}  "
                      f"(ILP rediscovered a known pattern)")
            else:
                print(f"  ↳ NOVEL pattern — not in named vocabulary")

        # 4. Confidence (Eq.2) + ILP signature (Eq.20)
        # Collect positive motifs first so they go into the signature
        _pos_motifs: List[str] = []
        _seen_m: Set[str] = set()
        for _f in pos_features:
            for _m in _f.get("motifs", []):
                if _m not in _seen_m:
                    _pos_motifs.append(_m); _seen_m.add(_m)
        confidence = _arch_confidence(best_clause, pos_features, neg_features)
        sig        = _arch_signature(best_clause, motifs=_pos_motifs)
        print(f"\n  Confidence : {confidence:.3f}")
        print(f"  Signature  : {sig}")

        # 4b. Leave-one-out F1 (out-of-sample estimate; None when <3 positives)
        loo_f1 = self._evaluate_clause_loo(
            target_type, pos_features, neg_features, pos_raw, neg_raw)
        if loo_f1 is not None:
            print(f"  LOO-F1     : {loo_f1:.3f}  (in-sample F1 = {best_f1:.3f})")

        # 5. Build B_arch with per-node facts
        bg_parts = []
        for i, f in enumerate(pos_features):
            nodes_i = pos_raw[i][0] if pos_raw else None
            edges_i = pos_raw[i][1] if pos_raw else None
            bg_parts.append(f"% positive circuit {i+1}")
            bg_parts.append(_build_arch_background(
                f"C_pos{i+1}", f, nodes_i, edges_i))
        for i, f in enumerate(neg_features):
            nodes_i = neg_raw[i][0] if neg_raw else None
            edges_i = neg_raw[i][1] if neg_raw else None
            bg_parts.append(f"% negative circuit {i+1}")
            bg_parts.append(_build_arch_background(
                f"C_neg{i+1}", f, nodes_i, edges_i))

        # 6. Collect all motifs observed in positives
        all_motifs: List[str] = []
        seen: Set[str] = set()
        for f in pos_features:
            for m in f.get("motifs", []):
                if m not in seen:
                    all_motifs.append(m); seen.add(m)

        _not = chr(92) + "+"
        pos_ex = [f"arch_{target_type}(C_pos{i+1})." for i in range(len(pos_features))]
        neg_ex = [f"{_not}arch_{target_type}(C_neg{i+1})." for i in range(len(neg_features))]

        return {
            "clause_text":          best_clause,
            "predicates":           sig["predicates"],
            "confidence":           confidence,
            "loo_f1":               loo_f1,   # out-of-sample; None when <3 positives
            "ilp_signature":        sig,
            "coverage":             best_cov,
            "precision":            best_prec,
            "f1":                   best_f1,
            "hypothesis_pool":      best_pool,         # "named" | "blind" | "fallback"
            "blind_motif_match":    blind_motif_match, # named motif if blind won, else None
            "discriminative_motifs": disc_motifs,      # [(name, IG), ...]
            "motifs":               all_motifs,
            "positive_examples":    pos_ex,
            "negative_examples":    neg_ex,
            "background_facts":     "\n".join(bg_parts),
            "features":             pos_features,
        }
