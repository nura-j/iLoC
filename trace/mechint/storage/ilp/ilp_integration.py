"""
ILP Integration Module (Updated)
=================================
Wires together all ILP components and integrates with the OWL infrastructure.
Task-specific behaviour is injected via TaskConfig — no hardcoded predicates,
role maps, or field names.

Changes vs original
-------------------
*   ILP_arch now uses ArchitecturalILPLearner (ilp_arch.py) instead of
    the stub ArchitecturalSignatureLearner (ilp_arch.py).
    The new learner:
      - Infers motifs from actual graph topology (edge type-pairs, 3-node paths,
        hub nodes, skip connections, layer-attn profiles) — not hard-coded rules.
      - All features are RELATIVE to model size (rel_size, attn_fraction_of_model).
      - Uses information gain to select discriminative motifs before generating
        ILP hypotheses.
      - Builds per-node B_arch facts (paper Appendix D).
      - Returns the full result dict including discriminative_motifs, ilp_signature,
        clause_text, confidence, F1, background_facts, etc.

*   New methods:
      learn_architectural_signature_from_graphs(pos_graphs, neg_graphs, arch_type)
          → full result dict from new learner (preferred when you have OWL graphs)
      learn_architectural_signature_from_json(pos_records, neg_records, arch_type)
          → full result dict from new learner (preferred when you have JSON records)

*   learn_architectural_signature() (original API) still works — it now delegates
    to learn_architectural_signature_from_graphs() internally.

*   process_circuit() (ILP_sem pipeline) is unchanged.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from rdflib import Graph, URIRef, Namespace

from .task_config import TaskConfig
from .predicate_extraction import PredicateExtractor
from .ilp_sem import SemanticSpecificationLearner

# ── Architectural learner: use the full implementation ────────────────────────
# Try to import from the local outputs path first (when running scripts from
# the COMPASS directory), then fall back to the package-relative stub so that
# the package can still be imported even if ilp_arch_full is not on the path.
try:
    from ilp_arch import ArchitecturalILPLearner as _FullArchLearner
    _FULL_ARCH_AVAILABLE = True
except ImportError:
    try:
        from .ilp_arch import ArchitecturalILPLearner as _FullArchLearner
        _FULL_ARCH_AVAILABLE = True
    except ImportError:
        _FULL_ARCH_AVAILABLE = False

# ilp_arch is now the full implementation; no separate stub needed
_StubArchLearner = None


class ILPIntegration:
    """
    Complete ILP pipeline integration.

    Connects:
    - Module 1: Predicate extraction  (config-driven, unchanged)
    - Module 2: ILP_sem learning      (config-driven, unchanged)
    - Module 3: ILP_arch learning     (now uses full ArchitecturalILPLearner)
    - OWL Exporter: Storage           (optional, unchanged)
    """

    def __init__(
            self,
            task_config: TaskConfig,
            owl_exporter=None,
            schema=None,
            model_n_layers: int = 0,   # 0 = infer from circuit data
            model_n_heads:  int = 0,   # 0 = infer from circuit data
    ):
        """
        Parameters
        ----------
        task_config     : TaskConfig specifying the learning task
        owl_exporter    : OWLExporter instance (optional)
        schema          : MECHINT_SCHEMA (optional)
        model_n_layers  : Total transformer layers for relative feature normalisation.
                          Set to the actual model size (e.g. 6 for Pythia-14M).
                          0 means infer from the circuit being analysed.
        model_n_heads   : Attention heads per layer for relative size normalisation.
        """
        self.task_config    = task_config
        self.owl_exporter   = owl_exporter
        self.schema         = schema
        self.model_n_layers = model_n_layers
        self.model_n_heads  = model_n_heads

        self.predicate_extractor = PredicateExtractor(task_config, schema)
        self.sem_learner         = SemanticSpecificationLearner(task_config)

        # Full arch learner
        if not _FULL_ARCH_AVAILABLE:
            raise ImportError(
                "Could not import ArchitecturalILPLearner from ilp_arch. "
                "Ensure ilp_arch.py (the full implementation) is on the path."
            )
        self.arch_learner      = _FullArchLearner()
        self._arch_learner_new = True

        self.namespace = Namespace("http://purl.org/mechint#")

    # ══════════════════════════════════════════════════════════════════
    # ILP_sem pipeline (unchanged)
    # ══════════════════════════════════════════════════════════════════

    def process_circuit(
            self,
            circuit_ttl_path:   str,
            task_examples:      List[Dict],
            tokenizer_mapping:  Optional[Dict] = None,
            save_intermediate:  bool = True,
    ) -> Dict:
        """
        ILP_sem: complete semantic specification learning for one circuit.

        Parameters
        ----------
        circuit_ttl_path  : Path to circuit TTL file
        task_examples     : List of task examples (format depends on task config)
        tokenizer_mapping : Optional activation data
        save_intermediate : Save facts.pl and learned_spec.json alongside TTL

        Returns
        -------
        Dict with keys: specification_text, predicates, confidence, ilp_signature
        """
        task_name = self.task_config.task_name
        print(f"\n{'=' * 60}")
        print(f"ILP_sem: {Path(circuit_ttl_path).name}  [{task_name}]")
        print(f"{'=' * 60}")

        # 1. Load circuit
        print("\n1. Loading circuit OWL graph...")
        circuit_graph = Graph()
        circuit_graph.parse(circuit_ttl_path, format="turtle")
        circuit_uri = self._get_circuit_uri(circuit_graph)
        print(f"   {circuit_uri}")

        # 2. Extract predicate facts
        print("\n2. Extracting predicates (F(C,x) fact database)...")
        facts = self.predicate_extractor.build_fact_database(
            circuit_graph=circuit_graph,
            circuit_uri=circuit_uri,
            examples=task_examples,
            tokenizer_mapping=tokenizer_mapping,
            apply_cwa=True,
        )
        n_fact_lines = len([l for l in facts.split("\n") if l.strip()])
        print(f"   {n_fact_lines} fact lines extracted")

        if save_intermediate:
            facts_path = circuit_ttl_path.replace(".ttl", "_facts.pl")
            Path(facts_path).write_text(facts)
            print(f"   Facts -> {facts_path}")

        # 3. Separate positive / negative examples
        print("\n3. Parsing examples...")
        positive_examples = [l for l in facts.split("\n") if l.startswith("positive(")]
        negative_examples = [l for l in facts.split("\n") if l.startswith("negative(")]
        print(f"   Positive: {len(positive_examples)}   Negative: {len(negative_examples)}")

        if not positive_examples:
            print("   No positive examples — cannot learn")
            return {"error": "No positive examples"}

        # 4. Learn semantic specification (ILP_sem)
        print("\n4. ILP_sem: learning specification (paper §3.2)...")
        spec = self.sem_learner.learn_specification(
            facts=facts,
            positive_examples=positive_examples,
            negative_examples=negative_examples,
        )
        print(f"   Specification : {spec['specification_text']}")
        print(f"   Confidence    : {spec['confidence']:.3f}")

        # 5. Optional OWL storage
        if self.owl_exporter:
            print("\n5. Storing in OWL...")
            try:
                self.owl_exporter.add_learned_specification(
                    circuit_ttl_path=circuit_ttl_path,
                    spec_text=spec["specification_text"],
                    predicates=", ".join(spec["predicates"]),
                    confidence=spec["confidence"],
                    ilp_signature=spec["ilp_signature"],
                )
                print(f"   Stored in: {circuit_ttl_path}")
            except Exception as e:
                print(f"   OWL storage error: {e}")

        if save_intermediate:
            result_path = circuit_ttl_path.replace(".ttl", "_learned_spec.json")
            Path(result_path).write_text(json.dumps(spec, indent=2))
            print(f"   Result -> {result_path}")

        print(f"\n{'=' * 60}")
        print("ILP_sem COMPLETE")
        print(f"{'=' * 60}\n")
        return spec

    def process_batch(
            self,
            circuit_ttl_paths:             List[str],
            task_examples_per_circuit:     Dict[str, List[Dict]],
            tokenizer_mapping_per_circuit: Optional[Dict[str, Dict]] = None,
    ) -> List[Dict]:
        """
        ILP_sem pipeline over multiple circuits.
        """
        results = []
        print(f"\n{'=' * 60}")
        print(f"ILP_sem BATCH: {len(circuit_ttl_paths)} circuits  [{self.task_config.task_name}]")
        print(f"{'=' * 60}")

        for i, path in enumerate(circuit_ttl_paths, 1):
            print(f"\n[{i}/{len(circuit_ttl_paths)}] {Path(path).name}")
            examples = task_examples_per_circuit.get(path, [])
            tok_map  = (tokenizer_mapping_per_circuit or {}).get(path)
            try:
                spec = self.process_circuit(path, examples, tok_map)
                results.append({"circuit": path, "status": "success", "specification": spec})
            except Exception as e:
                print(f"   Error: {e}")
                results.append({"circuit": path, "status": "error", "error": str(e)})

        ok = sum(1 for r in results if r["status"] == "success")
        print(f"\n{'=' * 60}")
        print(f"ILP_sem BATCH DONE: {ok}/{len(results)} successful")
        print(f"{'=' * 60}")
        return results

    # ══════════════════════════════════════════════════════════════════
    # ILP_arch pipeline — new full learner
    # ══════════════════════════════════════════════════════════════════

    def learn_architectural_signature_from_graphs(
            self,
            pos_graphs:  List[Graph],
            neg_graphs:  List[Graph],
            arch_type:   str,
            save_background_to: Optional[str] = None,
    ) -> Dict:
        """
        ILP_arch from OWL graphs (paper §3.3).

        Uses the full ArchitecturalILPLearner:
          - Infers motifs from actual edge topology (not heuristics)
          - Relative features (rel_size, attn_fraction_of_model, depth as fraction)
          - Discriminative motif selection via information gain
          - Per-node B_arch facts (Appendix D)
          - F1-optimal hypothesis selection from Eq.4 template family
          - Confidence = tp/|Ep| with consistency penalty (Eq.2/15)
          - ILP signature S_ILP_arch (Eq.20)

        Parameters
        ----------
        pos_graphs            : OWL graphs of circuits with arch_type
        neg_graphs            : OWL graphs of other-type circuits (negatives)
        arch_type             : Name for the learned type, e.g. "ioi"
        save_background_to    : If given, writes B_arch Prolog facts to this path

        Returns
        -------
        Full result dict with keys:
          clause_text, predicates, confidence, ilp_signature,
          coverage, precision, f1,
          discriminative_motifs  [(name, ig_score), ...],
          motifs, background_facts, features,
          positive_examples, negative_examples
        """
        print(f"\n{'=' * 60}")
        print(f"ILP_arch: arch_{arch_type}  (from OWL graphs)")
        print(f"{'=' * 60}")

        if self._arch_learner_new:
            result = self.arch_learner.learn_from_graphs(
                target_type=arch_type,
                positive_graphs=pos_graphs,
                negative_graphs=neg_graphs,
                model_n_layers=self.model_n_layers,
            )
        else:
            # Fallback to stub: construct a minimal compatible result
            stub_sig = self.arch_learner.learn_signature(
                pos_graphs=pos_graphs,
                neg_graphs=neg_graphs,
                arch_type=arch_type,
                n_layers=self.model_n_layers or 6,
            )
            result = {
                "clause_text":          stub_sig.get("clause", f"arch_{arch_type}(C) :- true."),
                "predicates":           stub_sig.get("predicates", []),
                "confidence":           stub_sig.get("confidence", 0.5),
                "ilp_signature":        stub_sig.get("ilp_signature", {}),
                "coverage":             stub_sig.get("confidence", 0.5),
                "precision":            stub_sig.get("confidence", 0.5),
                "f1":                   stub_sig.get("confidence", 0.5),
                "discriminative_motifs": [],
                "motifs":               stub_sig.get("motifs", []),
                "background_facts":     "",
                "features":             [],
                "positive_examples":    [],
                "negative_examples":    [],
            }

        if save_background_to:
            _write_background(save_background_to, arch_type, result)

        return result

    def learn_architectural_signature_from_json(
            self,
            positive_records: List[Dict],
            negative_records: List[Dict],
            arch_type:        str,
            save_background_to: Optional[str] = None,
    ) -> Dict:
        """
        ILP_arch from task JSON records (paper §3.3).

        Same as learn_architectural_signature_from_graphs but accepts the
        JSON dicts output by results_export.export_task_json() instead of
        rdflib Graph objects.  This is the preferred path.

        Parameters
        ----------
        positive_records   : List of task result dicts with layer1.circuit
        negative_records   : List of task result dicts for negative examples
        arch_type          : Name for the learned type
        save_background_to : Optional path for B_arch .pl output

        Returns
        -------
        Same full result dict as learn_architectural_signature_from_graphs()
        """
        print(f"\n{'=' * 60}")
        print(f"ILP_arch: arch_{arch_type}  (from JSON records)")
        print(f"{'=' * 60}")

        if not self._arch_learner_new:
            print("[ILPIntegration] Warning: ilp_arch (full) not available; "
                  "returning stub result for JSON path.")
            return {
                "clause_text":         f"arch_{arch_type}(C) :- true.",
                "predicates":          [],
                "confidence":          0.5,
                "ilp_signature":       {"n_clauses": 1, "predicates": [],
                                        "avg_length": 1, "avg_depth": 1, "complexity": 0.1},
                "coverage":            0.5,
                "precision":           0.5,
                "f1":                  0.5,
                "discriminative_motifs": [],
                "motifs":              [],
                "background_facts":    "",
                "features":            [],
                "positive_examples":   [],
                "negative_examples":   [],
            }

        result = self.arch_learner.learn_from_json_records(
            target_type=arch_type,
            positive_records=positive_records,
            negative_records=negative_records,
            model_n_layers=self.model_n_layers,
            model_n_heads=self.model_n_heads,
        )

        if save_background_to:
            _write_background(save_background_to, arch_type, result)

        return result

    # ── Original API (backward compatible) ───────────────────────────────────

    def learn_architectural_signature(
            self,
            circuit_ttl_paths: List[str],
            task_family:       Optional[str] = None,
            neg_ttl_paths:     Optional[List[str]] = None,
    ) -> Dict:
        """
        Original API: learn arch signature from TTL file paths.

        Now delegates to learn_architectural_signature_from_graphs() with the
        full learner.  Negative graphs are loaded from neg_ttl_paths if given.

        Parameters
        ----------
        circuit_ttl_paths : Positive circuit TTL files
        task_family       : Name for the arch type (defaults to task_config.task_name)
        neg_ttl_paths     : Optional negative circuit TTL files
        """
        arch_type = task_family or self.task_config.task_name

        print(f"\n{'=' * 60}")
        print(f"ILP_arch: loading {len(circuit_ttl_paths)} positive TTL(s)")
        print(f"{'=' * 60}")

        pos_graphs = []
        for p in circuit_ttl_paths:
            g = Graph(); g.parse(p, format="turtle")
            pos_graphs.append(g)

        neg_graphs = []
        for p in (neg_ttl_paths or []):
            g = Graph(); g.parse(p, format="turtle")
            neg_graphs.append(g)

        return self.learn_architectural_signature_from_graphs(
            pos_graphs=pos_graphs,
            neg_graphs=neg_graphs,
            arch_type=arch_type,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_circuit_uri(self, graph: Graph) -> URIRef:
        results = list(graph.query(
            "SELECT ?c WHERE { ?c a <http://purl.org/mechint#Circuit> . } LIMIT 1"))
        if results:
            return results[0][0]
        raise ValueError("No Circuit found in OWL graph")


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def _write_background(path: str, arch_type: str, result: Dict) -> None:
    """Write B_arch Prolog facts and examples to a .pl file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        f.write(f"% ILP_arch background knowledge — arch_{arch_type}\n")
        f.write(f"% Learned clause:\n%   {result.get('clause_text', '')}\n")
        f.write(f"% Confidence : {result.get('confidence', 0):.3f}\n")
        f.write(f"% F1         : {result.get('f1', 0):.3f}\n")
        disc = result.get("discriminative_motifs", [])
        if disc:
            f.write("% Discriminative motifs (by information gain):\n")
            for m, ig in disc[:6]:
                f.write(f"%   {m:<35} IG={ig:.4f}\n")
        f.write("\n")
        f.write(result.get("background_facts", ""))
        pos_ex = result.get("positive_examples", [])
        neg_ex = result.get("negative_examples", [])
        if pos_ex:
            f.write("\n\n% Positive examples:\n")
            f.write("\n".join(pos_ex))
        if neg_ex:
            f.write("\n\n% Negative examples:\n")
            f.write("\n".join(neg_ex))
    print(f"   B_arch -> {path}")


# ══════════════════════════════════════════════════════════════════
# Convenience function (unchanged public API)
# ══════════════════════════════════════════════════════════════════

def process_single_circuit(
        circuit_ttl_path: str,
        task_examples:    List[Dict],
        task_config:      TaskConfig,
        owl_exporter=None,
        schema=None,
) -> Dict:
    """
    Convenience function: ILP_sem for a single circuit TTL.
    """
    integration = ILPIntegration(task_config, owl_exporter, schema)
    return integration.process_circuit(circuit_ttl_path, task_examples)
