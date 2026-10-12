"""
OWL Circuit Exporter - Four-Layer Architecture
Exports EAP-discovered circuits to OWL knowledge graphs

Layers:
  0. Provenance: Task, experiment, model metadata
  1. Structural: Components, edges, scores from EAP
  2. Semantic: Learned specifications (ILP_sem / CFS)
  3. Architectural: Motifs, depths, signatures (for transfer )
  4. Meta-knowledge: Comparisons, subsumption (via SPARQL inference)
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from pathlib import Path

from rdflib import Graph as RDFGraph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD, OWL

from ..ontology.schema import MECHINT_SCHEMA, MECHINT, MECHINT_BASE


class OWLExporter:
    """
    Export circuits to OWL/Turtle format with full 4-layer representation
    """

    def __init__(self, base_uri: str = MECHINT_BASE, output_dir: str = "./data/experiments"):
        self.base_uri = base_uri
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.schema = MECHINT_SCHEMA
        self.namespace = MECHINT

    def _add_pattern_triples(self, g, circuit_uri: URIRef, patterns: Dict[str, Any]):
        """Add pattern information as RDF triples"""

        # Add hub components
        for hub in patterns['hub_components']:
            component_name = hub['component']
            component_uri = self._get_or_create_component_uri(component_name)

            # Mark as hub
            hub_uri = URIRef(f"{circuit_uri}/hub_{component_name}")
            g.add((hub_uri, RDF.type, self.schema.namespace.HubComponent))
            g.add((hub_uri, self.schema.namespace.aboutComponent, component_uri))
            g.add((hub_uri, self.schema.namespace.degree, Literal(hub['degree'], datatype=XSD.integer)))
            g.add((circuit_uri, self.schema.namespace.hasHubComponent, hub_uri))

        # Add skip connections
        for idx, skip in enumerate(patterns['skip_connections']):
            skip_uri = URIRef(f"{circuit_uri}/skip_{idx}")
            g.add((skip_uri, RDF.type, self.schema.namespace.SkipConnection))
            g.add((skip_uri, self.schema.namespace.sourceComponent,
                   self._get_or_create_component_uri(skip['source'])))
            g.add((skip_uri, self.schema.namespace.targetComponent,
                   self._get_or_create_component_uri(skip['target'])))
            g.add((skip_uri, self.schema.namespace.layerGap,
                   Literal(skip['layer_gap'], datatype=XSD.integer)))
            g.add((circuit_uri, self.schema.namespace.hasSkipConnection, skip_uri))

        # Add motifs
        for idx, motif in enumerate(patterns['motifs']):
            motif_uri = URIRef(f"{circuit_uri}/motif_{idx}")
            g.add((motif_uri, RDF.type, self.schema.namespace.Motif))
            g.add((motif_uri, self.schema.namespace.motifType,
                   Literal(motif['type'])))

            # Add path as RDF list
            path_str = " -> ".join(motif['path'])
            g.add((motif_uri, self.schema.namespace.motifPath, Literal(path_str)))
            g.add((circuit_uri, self.schema.namespace.hasMotif, motif_uri))


    def _get_or_create_component_uri(self, component_name: str) -> URIRef:
        """Get or create component URI"""
        return URIRef(f"{self.schema.base_uri}/component/{component_name}")

    def export_circuit(
            self,
            graph,  # EAP Graph object
            experiment_id: str,
            step: int,
            config: Dict[str, Any],
            metadata: Dict[str, Any],
            eap_results: Dict[str, Any],
            task_family: str = None,
            role_or_phenomenon: str = None,
            operation_type: Optional[str] = None,   # κ: e.g. "binding", "selection"
            logical_category: Optional[str] = None, # τ: "PREDICATE" | "ROLE" | "OPERATOR"
    ) -> str:
        """
        Export circuit to OWL format.

        Args:
            graph: EAP Graph object (from trace.eap.graph)
            experiment_id: Unique identifier for this experiment
            step: Training/checkpoint step number
            config: Configuration dict (method, top_k, model settings)
            metadata: Metadata dict (model_name, researcher, n_layers, etc.)
            eap_results: EAP analysis results (faithfulness, performance)
            task_family: Task family (e.g., "SemanticRoles", "IOI", "GreaterThan")
            role_or_phenomenon: Specific role/phenomenon (e.g., "INSTRUMENT", "NameMover")
            operation_type: κ from L(C) = <sigma, kappa, phi, tau> (paper Eq. 1)
                            e.g. "binding" for role circuits, "selection" for IOI
            logical_category: τ from L(C); one of "PREDICATE", "ROLE", "OPERATOR"

        Returns:
            Path to exported .ttl file
        """

        print(f"\n[OWL Export] Starting export for {experiment_id} (step {step})...")
        print(f"\n[OWL Export DEBUG]")
        print(f"  experiment_id: {experiment_id}")
        print(f"  task_family: '{task_family}' (type: {type(task_family)})")
        print(f"  role_or_phenomenon: '{role_or_phenomenon}' (type: {type(role_or_phenomenon)})")

        # Create RDF graph
        rdf_graph = RDFGraph()
        rdf_graph.bind("mechint", self.namespace)
        rdf_graph.bind("owl", OWL)
        rdf_graph.bind("rdf", RDF)
        rdf_graph.bind("rdfs", RDFS)
        rdf_graph.bind("xsd", XSD)

        # Generate URIs
        circuit_uri = self._make_uri(f"circuit_{experiment_id}_step{step}")
        experiment_uri = self._make_uri(f"experiment_{experiment_id}")

        # ========================================
        # LAYER 0: PROVENANCE
        # ========================================
        print("[OWL Export] Layer 0: Adding provenance...")
        self._add_provenance(
            rdf_graph, circuit_uri, experiment_uri,
            step, metadata, config, task_family, experiment_id, role_or_phenomenon,
            operation_type=operation_type,
            logical_category=logical_category,
        )
        # ========================================
        # LAYER 1: STRUCTURAL
        # ========================================
        print("[OWL Export] Layer 1: Adding structural information...")
        node_count, edge_count = self._add_structural_layer(
            rdf_graph, circuit_uri, graph, eap_results, config
        )

        # ========================================
        # LAYER 2: SEMANTIC (Prepared for ILP)
        # ========================================
        print("[OWL Export] Layer 2: Preparing semantic layer...")
        self._prepare_semantic_layer(
            rdf_graph, circuit_uri, task_family, role_or_phenomenon
        )

        print(f"[OWL] Detecting circuit patterns...")
        patterns = self._detect_circuit_patterns(graph)

        self._add_pattern_triples(rdf_graph, circuit_uri, patterns)
        # ========================================
        # LAYER 3: ARCHITECTURAL
        # ========================================
        print("[OWL Export] Layer 3: Computing architectural features...")
        self._add_architectural_layer(
            rdf_graph, circuit_uri, graph, metadata
        )

        # Save to file
        output_path = self._save_graph(rdf_graph, experiment_id, step)

        print(f"[OWL Export]  Export complete!")
        print(f"[OWL Export]    - File: {output_path}")
        print(f"[OWL Export]    - Nodes: {node_count}, Edges: {edge_count}")
        print(f"[OWL Export]    - Task: {task_family}/{role_or_phenomenon}")

        return str(output_path)

    # ========================================
    # LAYER 0: PROVENANCE
    # ========================================

    def _add_provenance(
            self,
            rdf_graph: RDFGraph,
            circuit_uri: URIRef,
            experiment_uri: URIRef,
            step: int,
            metadata: Dict[str, Any],
            config: Dict[str, Any],
            task_family: Optional[str],
            experiment_id: Optional[str],
            role_or_phenomenon: Optional[str],
            operation_type: Optional[str] = None,
            logical_category: Optional[str] = None,
    ):
        """Add Layer 0: Provenance metadata (κ and τ from paper Eq. 1)"""

        # Determine circuit type from task and role
        print(f"\n[DEBUG] task_family: '{task_family}'")
        print(f"[DEBUG] role_or_phenomenon: '{role_or_phenomenon}'")

        # Determine circuit type from task and role
        if task_family and role_or_phenomenon:
            circuit_type = self.schema.get_circuit_type(task_family, role_or_phenomenon)
            print(f"[DEBUG] circuit_type returned: {circuit_type}")
        else:
            circuit_type = self.namespace.Circuit
            print(f"[DEBUG] Using default Circuit (no task/role provided)")

        # LAYER 0: PROVENANCE - Circuit type assertions (experimental labels)
        # These indicate what we DESIGNED to test, not what we LEARNED
        rdf_graph.add((circuit_uri, RDF.type, circuit_type))  # e.g., InstrumentCircuit
        rdf_graph.add((circuit_uri, RDF.type, self.namespace.Circuit))  # Base type

        # Add hierarchical type if applicable (RoleCircuit, IOICircuit, etc.)
        if circuit_type in [
            self.namespace.InstrumentCircuit,
            self.namespace.LocationCircuit,
            self.namespace.TimeCircuit,
            self.namespace.MannerCircuit,
            self.namespace.PurposeCircuit,
            self.namespace.BeneficiaryCircuit,
            self.namespace.AgentCircuit,
            self.namespace.PatientCircuit,
            self.namespace.ThemeCircuit,
        ]:
            rdf_graph.add((circuit_uri, RDF.type, self.namespace.RoleCircuit))
            rdf_graph.add((circuit_uri, RDF.type, self.namespace.TaskSpecificCircuit))
            print(f"[DEBUG] Also added RoleCircuit + TaskSpecificCircuit type")

        elif circuit_type in [
            self.namespace.IOICircuit,
            self.namespace.NameMoverCircuit,
            self.namespace.DuplicateTokenCircuit,
            self.namespace.SInhibitionCircuit,
        ]:
            rdf_graph.add((circuit_uri, RDF.type, self.namespace.IOICircuit))
            rdf_graph.add((circuit_uri, RDF.type, self.namespace.TaskSpecificCircuit))
            print(f"[DEBUG] Also added IOICircuit + TaskSpecificCircuit type")

        elif circuit_type in [
            self.namespace.NumericalCircuit,
            self.namespace.MagnitudeComparisonCircuit,
            self.namespace.DigitExtractionCircuit,
        ]:
            rdf_graph.add((circuit_uri, RDF.type, self.namespace.NumericalCircuit))
            rdf_graph.add((circuit_uri, RDF.type, self.namespace.TaskSpecificCircuit))
            print(f"[DEBUG] Also added NumericalCircuit + TaskSpecificCircuit type")

        # NOTE: Computational patterns (BindingPattern, etc.) are NOT assigned here!
        # They are LEARNED by ILP_sem and added via add_learned_specification()
        # This separation allows us to discover when experimental labels don't match learned patterns

        # Experiment
        rdf_graph.add((experiment_uri, RDF.type, self.namespace.Experiment))
        rdf_graph.add((circuit_uri, self.namespace.discoveredIn, experiment_uri))

        # Task information
        if task_family:
            task_family_uri = getattr(self.namespace, task_family, None)
            if task_family_uri:
                rdf_graph.add((circuit_uri, self.namespace.belongsToTaskFamily, task_family_uri))
            rdf_graph.add((circuit_uri, self.namespace.taskName, Literal(task_family)))


        if role_or_phenomenon:
            rdf_graph.add((circuit_uri, self.namespace.roleName, Literal(role_or_phenomenon)))

        # Model
        model_name = metadata.get("model_name", "unknown")
        model_uri = self._make_uri(f"model_{model_name}")
        rdf_graph.add((model_uri, RDF.type, self.namespace.Model))
        rdf_graph.add((circuit_uri, self.namespace.fromModel, model_uri))
        rdf_graph.add((model_uri, self.namespace.modelName, Literal(model_name)))

        # Model type
        model_type = config.get("model_type") or metadata.get("model_type")
        if model_type:
            rdf_graph.add((model_uri, RDFS.comment, Literal(f"model_type:{model_type}")))

        # Checkpoint
        checkpoint_uri = self._make_uri(f"checkpoint_{model_name}_step{step}")
        rdf_graph.add((checkpoint_uri, RDF.type, self.namespace.Checkpoint))
        rdf_graph.add((circuit_uri, self.namespace.atCheckpoint, checkpoint_uri))
        rdf_graph.add((checkpoint_uri, self.namespace.step, Literal(step, datatype=XSD.integer)))

        # Timestamp
        timestamp = datetime.utcnow().isoformat() + "Z"
        rdf_graph.add((circuit_uri, self.namespace.timestamp, Literal(timestamp, datatype=XSD.dateTime)))

        # EAP method
        eap_method = config.get("method", metadata.get("eap_method", "unknown"))
        rdf_graph.add((circuit_uri, RDFS.comment, Literal(f"eapMethod:{eap_method}")))

        # Optional metadata
        if "description" in metadata:
            rdf_graph.add((experiment_uri, RDFS.comment, Literal(metadata["description"])))
        if "researcher" in metadata:
            rdf_graph.add((experiment_uri, RDFS.label, Literal(metadata["researcher"])))

        # Dataset
        if "dataset_path" in metadata:
            dataset_uri = self._make_uri(f"dataset_{experiment_id}")
            rdf_graph.add((dataset_uri, RDF.type, self.namespace.Dataset))
            rdf_graph.add((dataset_uri, self.namespace.datasetPath,
                               Literal(metadata["dataset_path"])))
            rdf_graph.add((circuit_uri, self.namespace.usedDataset, dataset_uri))

        # Intervention type
        if "intervention_type" in config:
            rdf_graph.add((circuit_uri, self.namespace.interventionType,
                            Literal(config["intervention_type"])))

        # κ (operation type) and τ (logical category) — paper Eq. 1 / L(C) fields
        # These also flow in from metadata if not passed explicitly (backward compat)
        _kappa = operation_type or metadata.get("operation_type")
        _tau   = logical_category or metadata.get("logical_category")
        if _kappa:
            rdf_graph.add((circuit_uri, self.namespace.operationType,
                           Literal(_kappa)))
        if _tau:
            rdf_graph.add((circuit_uri, self.namespace.logicalCategory,
                           Literal(_tau)))

    # ========================================
    # LAYER 1: STRUCTURAL
    # ========================================

    def _add_structural_layer(
            self,
            rdf_graph: RDFGraph,
            circuit_uri: URIRef,
            graph,  # EAP Graph
            eap_results: Dict[str, Any],
            config: Dict[str, Any],
    ) -> tuple:
        """Add Layer 1: Structural representation"""

        # Extract nodes and edges in circuit
        circuit_nodes = [node for node in graph.nodes.values() if node.in_graph]
        circuit_edges = [edge for edge in graph.edges.values() if edge.in_graph]

        # Add nodes
        for node in circuit_nodes:
            node_uri = self._make_uri(f"node_{node.name}")

            # Component type
            component_type = self.schema.get_component_type_for_node(node.name)
            rdf_graph.add((node_uri, RDF.type, component_type))
            rdf_graph.add((node_uri, RDF.type, self.namespace.Component))

            # Link to circuit
            rdf_graph.add((circuit_uri, self.namespace.hasMemberNode, node_uri))
            rdf_graph.add((circuit_uri, self.namespace.hasComponent, node_uri))

            # Properties
            rdf_graph.add((node_uri, self.namespace.address, Literal(node.name)))
            rdf_graph.add((node_uri, self.namespace.inGraph, Literal(True, datatype=XSD.boolean)))

            # Layer/head info
            structure = self.schema.parse_node_structure(node.name)
            if 'layer' in structure:
                rdf_graph.add((node_uri, self.namespace.inLayer,
                               Literal(structure['layer'], datatype=XSD.integer)))
            if 'head' in structure:
                rdf_graph.add((node_uri, self.namespace.head,
                               Literal(structure['head'], datatype=XSD.integer)))

        # Add edges
        for i, edge in enumerate(circuit_edges):
            # EAP Graph uses .parent/.child attributes (NOT .source/.target)
            edge_uri = self._make_uri(f"edge_{edge.parent.name}_to_{edge.child.name}")

            rdf_graph.add((edge_uri, RDF.type, self.namespace.Edge))

            # Link to circuit
            rdf_graph.add((circuit_uri, self.namespace.hasMemberEdge, edge_uri))

            # Edge structure
            src_uri = self._make_uri(f"node_{edge.parent.name}")
            dest_uri = self._make_uri(f"node_{edge.child.name}")
            rdf_graph.add((edge_uri, self.namespace.hasSource, src_uri))
            rdf_graph.add((edge_uri, self.namespace.hasTarget, dest_uri))
            rdf_graph.add((src_uri, self.namespace.connectsTo, dest_uri))

            # Properties
            rdf_graph.add((edge_uri, self.namespace.score,
                           Literal(float(edge.score), datatype=XSD.float)))
            rdf_graph.add((edge_uri, self.namespace.rank,
                           Literal(i + 1, datatype=XSD.integer)))
            rdf_graph.add((edge_uri, self.namespace.inGraph,
                           Literal(True, datatype=XSD.boolean)))

        # Circuit-level properties
        rdf_graph.add((circuit_uri, self.namespace.topK,
                       Literal(config.get("top_k", 0), datatype=XSD.integer)))

        # ========================================
        # PERFORMANCE METRICS (from EAP evaluation)
        # ========================================

        # Core performance metrics
        if "baseline_performance" in eap_results:
            baseline_perf = float(eap_results["baseline_performance"])
            rdf_graph.add((circuit_uri, self.namespace.baselinePerformance,
                           Literal(baseline_perf, datatype=XSD.float)))
            # Also store as taskAccuracy (Acc_T in draft §2.4)
            rdf_graph.add((circuit_uri, self.namespace.taskAccuracy,
                           Literal(baseline_perf, datatype=XSD.float)))
        else:
            # Default: unknown task accuracy
            rdf_graph.add((circuit_uri, self.namespace.taskAccuracy,
                           Literal(-1.0, datatype=XSD.float)))  # -1 = not available

        if "circuit_performance" in eap_results:
            rdf_graph.add((circuit_uri, self.namespace.circuitPerformance,
                           Literal(float(eap_results["circuit_performance"]), datatype=XSD.float)))
        else:
            # Default: unknown circuit performance
            rdf_graph.add((circuit_uri, self.namespace.circuitPerformance,
                           Literal(-1.0, datatype=XSD.float)))

        if "ablated_performance" in eap_results:
            rdf_graph.add((circuit_uri, self.namespace.ablatedPerformance,
                           Literal(float(eap_results["ablated_performance"]), datatype=XSD.float)))
        else:
            # Default: unknown ablated performance
            rdf_graph.add((circuit_uri, self.namespace.ablatedPerformance,
                           Literal(-1.0, datatype=XSD.float)))

        # Faithfulness metrics
        if "faithfulness" in eap_results:
            rdf_graph.add((circuit_uri, self.namespace.faithfulness,
                           Literal(float(eap_results["faithfulness"]), datatype=XSD.float)))
        else:
            # Default: unknown faithfulness
            rdf_graph.add((circuit_uri, self.namespace.faithfulness,
                           Literal(-1.0, datatype=XSD.float)))

        if "faithfulness_simple" in eap_results:
            rdf_graph.add((circuit_uri, self.namespace.faithfulnessSimple,
                           Literal(float(eap_results["faithfulness_simple"]), datatype=XSD.float)))
        else:
            # Default: unknown simple faithfulness
            rdf_graph.add((circuit_uri, self.namespace.faithfulnessSimple,
                           Literal(-1.0, datatype=XSD.float)))

        # Causal effect (Δ_T in draft §2.4)
        if "baseline_performance" in eap_results and "circuit_performance" in eap_results:
            causal_effect = eap_results["baseline_performance"] - eap_results["circuit_performance"]
            rdf_graph.add((circuit_uri, self.namespace.causalEffect,
                           Literal(float(causal_effect), datatype=XSD.float)))
        else:
            # Default: unknown causal effect
            rdf_graph.add((circuit_uri, self.namespace.causalEffect,
                           Literal(-1.0, datatype=XSD.float)))

        # Alternative: use performance_drop if provided directly
        if "performance_drop" in eap_results:
            rdf_graph.add((circuit_uri, self.namespace.performanceDrop,
                           Literal(float(eap_results["performance_drop"]), datatype=XSD.float)))
        else:
            # Default: unknown performance drop
            rdf_graph.add((circuit_uri, self.namespace.performanceDrop,
                           Literal(-1.0, datatype=XSD.float)))

        if "performance_drop_pct" in eap_results:
            rdf_graph.add((circuit_uri, self.namespace.performanceDropPct,
                           Literal(float(eap_results["performance_drop_pct"]), datatype=XSD.float)))
        else:
            # Default: unknown percentage drop
            rdf_graph.add((circuit_uri, self.namespace.performanceDropPct,
                           Literal(-1.0, datatype=XSD.float)))

        # ========================================
        # VALIDATION STATISTICS κ
        # ========================================
        # κ = ⟨Acc_T, Δ_T, confidence, ε-minimality⟩
        # - Acc_T: stored as taskAccuracy (above)
        # - Δ_T: stored as causalEffect (above)
        # - confidence: added later by ILP_sem in add_learned_specification()
        # - ε-minimality: NOT YET COMPUTED (requires greedy ablation validation)

        # Validation statistics (with defaults when not computed):
        if "epsilon_minimal" in eap_results:
            rdf_graph.add((circuit_uri, self.namespace.epsilonMinimal,
                           Literal(bool(eap_results["epsilon_minimal"]), datatype=XSD.boolean)))
        else:
            # Default: unknown ε-minimality (null/unknown - don't add property)
            pass  # Leave absent to indicate "not validated"

        if "permutation_pvalue" in eap_results:
            rdf_graph.add((circuit_uri, self.namespace.permutationPValue,
                           Literal(float(eap_results["permutation_pvalue"]), datatype=XSD.float)))
        else:
            # Default: unknown p-value (null/unknown - don't add property)
            pass  # Leave absent to indicate "not computed"

        return len(circuit_nodes), len(circuit_edges)

    # ========================================
    # LAYER 2: SEMANTIC (Prepared for ILP)
    # ========================================

    def _prepare_semantic_layer(
            self,
            rdf_graph: RDFGraph,
            circuit_uri: URIRef,
            task_family: Optional[str],
            role_or_phenomenon: Optional[str],
    ):
        """
        Prepare Layer 2: Semantic layer (populated later by ILP_sem / CFS).

        IMPORTANT: This method does NOT assign computational patterns!
        Patterns are learned by ILP_sem and added via add_learned_specification().

        This avoids conflating provenance (what we designed to test) with
        semantic inference (what ILP learned the circuit computes).

        """
        pass

    def _detect_circuit_patterns(self, graph: 'Graph') -> Dict[str, Any]:
        """Detect architectural patterns in circuit"""
        from collections import defaultdict

        patterns = {
            'motifs': [],
            'hub_components': [],
            'attention_heads': set(),
            'mlp_blocks': set(),
            'skip_connections': [],
            'qkv_patterns': defaultdict(list)
        }

        # Track component degrees (for hubs)
        component_degrees = defaultdict(int)

        # Analyze edges
        for edge_name, edge in graph.edges.items():
            if not edge.in_graph:
                continue

            # EAP Graph uses .parent/.child attributes
            src = edge.parent.name
            tgt = edge.child.name

            # Count connections (for hub detection)
            component_degrees[src] += 1
            component_degrees[tgt] += 1

            # Track component types
            if src.startswith('a'):
                patterns['attention_heads'].add(src)
            elif src.startswith('m'):
                patterns['mlp_blocks'].add(src)

            if tgt.startswith('a'):
                patterns['attention_heads'].add(tgt)
            elif tgt.startswith('m'):
                patterns['mlp_blocks'].add(tgt)

            # Detect skip connections (early to late layers)
            src_info = self.schema.parse_node_structure(src)
            tgt_info = self.schema.parse_node_structure(tgt)

            if src_info.get('layer') is not None and tgt_info.get('layer') is not None:
                layer_gap = tgt_info['layer'] - src_info['layer']
                if layer_gap > 2:  # Skip at least 2 layers
                    patterns['skip_connections'].append({
                        'source': src,
                        'target': tgt,
                        'layer_gap': layer_gap
                    })

            # Track Q/K/V patterns
            edge_type = edge.qkv  # 'q', 'k', 'v', or None
            if edge_type in ['q', 'k', 'v']:
                patterns['qkv_patterns'][edge_type].append({
                    'source': src,
                    'target': tgt,
                    'score': float(edge.score)
                })

        # Find hubs (components with many connections)
        hub_threshold = 5
        for component, degree in component_degrees.items():
            if degree >= hub_threshold:
                patterns['hub_components'].append({
                    'component': component,
                    'degree': degree
                })

        # Detect common motifs
        patterns['motifs'] = self._detect_motifs(graph)

        return patterns

    def _detect_motifs(self, graph: 'Graph') -> list:
        """
        Detect circuit motifs from edge topology.

        Each motif is a named subgraph pattern inferred by walking the
        actual EAP edges in the circuit (not aggregate statistics).
        Names are aligned with _MOTIF_HEURISTICS in ilp_arch.py so
        the SPARQL read-back in extract_background_knowledge() matches.

        Motifs detected
        ---------------
        attn_mlp_attn       : A → M → A  (existing — retained)
        scaffold_event_entity: A → A → M  chain spanning ≥2 layers (paper §3.5 example)
        induction_head      : A[L] → A[L+1] via Q-stream (next-layer Q dependency)
        token_copy          : early-A → late-A via V-stream (copy mechanism)
        ff_chain            : M[L] → M[L+1]  consecutive MLP blocks
        broadcast_head      : A with out-degree ≥ 3 (fan-out hub)
        """
        from collections import defaultdict

        circuit_edges = [e for e in graph.edges.values() if e.in_graph]
        if not circuit_edges:
            return []

        # Build adjacency structures keyed by node name
        # out_edges[src_name] = list of Edge
        out_edges: dict = defaultdict(list)
        in_degree:  dict = defaultdict(int)
        out_degree: dict = defaultdict(int)
        for e in circuit_edges:
            out_edges[e.parent.name].append(e)
            out_degree[e.parent.name] += 1
            in_degree[e.child.name]   += 1

        motifs = []
        seen: set = set()   # deduplicate by (type, path-tuple)

        def _is_attn(node) -> bool:
            return node.name.startswith('a')

        def _is_mlp(node) -> bool:
            return node.name.startswith('m')

        # Helper: get layer safely (Node.layer is always set)
        def _layer(node) -> int:
            return getattr(node, 'layer', -1)

        # ── 1. attn_mlp_attn : A → M → A ────────────────────────────────────
        for e1 in circuit_edges:
            if not (_is_attn(e1.parent) and _is_mlp(e1.child)):
                continue
            mlp_name = e1.child.name
            for e2 in out_edges.get(mlp_name, []):
                if not _is_attn(e2.child):
                    continue
                key = ('attn_mlp_attn', e1.parent.name, mlp_name, e2.child.name)
                if key not in seen:
                    seen.add(key)
                    motifs.append({
                        'type': 'attn_mlp_attn',
                        'path': [e1.parent.name, mlp_name, e2.child.name],
                    })

        # ── 2. scaffold_event_entity : A → A → M  spanning ≥ 2 layers ───────
        # Paper §3.5: head attends entity, passes to event head, then to MLP binder
        for e1 in circuit_edges:
            if not (_is_attn(e1.parent) and _is_attn(e1.child)):
                continue
            a1, a2 = e1.parent, e1.child
            if _layer(a2) <= _layer(a1):          # must go forward
                continue
            for e2 in out_edges.get(a2.name, []):
                if not _is_mlp(e2.child):
                    continue
                if _layer(e2.child) - _layer(a1) < 2:
                    continue
                key = ('scaffold_event_entity', a1.name, a2.name, e2.child.name)
                if key not in seen:
                    seen.add(key)
                    motifs.append({
                        'type': 'scaffold_event_entity',
                        'path': [a1.name, a2.name, e2.child.name],
                    })

        # ── 3. induction_head : A[L] → A[L+1] via Q-stream ──────────────────
        # Pattern: attention head at layer L influences the Q-input of head at L+1
        # This is detectable from EAP edges: parent=a{L}.h{*}, child=a{L+1}.h{*}, qkv='q'
        for e in circuit_edges:
            if not (_is_attn(e.parent) and _is_attn(e.child)):
                continue
            if e.qkv != 'q':
                continue
            if _layer(e.child) != _layer(e.parent) + 1:
                continue
            key = ('induction_head', e.parent.name, e.child.name)
            if key not in seen:
                seen.add(key)
                motifs.append({
                    'type': 'induction_head',
                    'path': [e.parent.name, e.child.name],
                    'qkv': 'q',
                })

        # ── 4. token_copy : early-A → late-A via V-stream ────────────────────
        # Pattern: attention head in early half of model feeds V-input of head
        # in late half — consistent with OV copy circuits
        n_layers = max((_layer(e.parent) for e in circuit_edges
                        if _is_attn(e.parent)), default=0) + 1
        mid = n_layers / 2.0
        for e in circuit_edges:
            if not (_is_attn(e.parent) and _is_attn(e.child)):
                continue
            if e.qkv != 'v':
                continue
            if _layer(e.parent) >= mid:
                continue
            if _layer(e.child) <= mid:
                continue
            key = ('token_copy', e.parent.name, e.child.name)
            if key not in seen:
                seen.add(key)
                motifs.append({
                    'type': 'token_copy',
                    'path': [e.parent.name, e.child.name],
                    'qkv': 'v',
                })

        # ── 5. ff_chain : M[L] → M[L+1]  consecutive MLPs ───────────────────
        for e in circuit_edges:
            if not (_is_mlp(e.parent) and _is_mlp(e.child)):
                continue
            if _layer(e.child) != _layer(e.parent) + 1:
                continue
            key = ('ff_chain', e.parent.name, e.child.name)
            if key not in seen:
                seen.add(key)
                motifs.append({
                    'type': 'ff_chain',
                    'path': [e.parent.name, e.child.name],
                })

        # ── 6. broadcast_head : A with out-degree ≥ 3 ────────────────────────
        for node in graph.nodes.values():
            if not node.in_graph:
                continue
            if not _is_attn(node):
                continue
            if out_degree.get(node.name, 0) >= 3:
                key = ('broadcast_head', node.name)
                if key not in seen:
                    seen.add(key)
                    targets = [e.child.name for e in out_edges.get(node.name, [])]
                    motifs.append({
                        'type': 'broadcast_head',
                        'path': [node.name] + targets[:4],  # cap for readability
                        'out_degree': out_degree[node.name],
                    })

        return motifs

    def _detect_dominant_components(self, graph) -> Dict:
        """Identify which component types dominate the circuit"""
        attn_count = sum(1 for n in graph.nodes.values()
                         if n.in_graph and 'a' in n.name)
        mlp_count = sum(1 for n in graph.nodes.values()
                        if n.in_graph and 'm' in n.name)

        return {
            'dominant_type': 'attention' if attn_count > mlp_count else 'mlp',
            'attn_ratio': attn_count / (attn_count + mlp_count),
            'attn_count': attn_count,
            'mlp_count': mlp_count,
        }

    def _detect_backward_attention(self, graph) -> List[Dict]:
        """Find attention heads that attend backward (to previous tokens)"""
        # Look for Q→K edges where source layer > target layer
        backward_heads = []
        for edge in graph.edges.values():
            if not edge.in_graph:
                continue
            if edge.qkv in ['q', 'k']:
                # EAP Graph uses .parent/.child attributes
                src_layer = self.schema.parse_node_structure(edge.parent.name).get('layer')
                tgt_layer = self.schema.parse_node_structure(edge.child.name).get('layer')
                if src_layer is not None and tgt_layer is not None:
                    if src_layer > tgt_layer:  # Attending backward
                        backward_heads.append({
                            'head': edge.parent.name,
                            'layer_diff': src_layer - tgt_layer,
                            'edge_type': edge.qkv,
                        })
        return backward_heads

    def _detect_induction_heads(self, graph) -> List[Dict]:
        """Detect induction head patterns (prev token → copy).

        NOTE: EAP-IG edges are typed as q/k/v only. OV circuits are weight-space
        constructs (TransformerLens hook_OV) not present as runtime activation edges.
        Induction head detection requires separate weight-space analysis, not
        implementable from the EAP graph alone.
        """
        return []

    # ========================================
    # LAYER 3: ARCHITECTURAL
    # ========================================

    def _add_architectural_layer(
            self,
            rdf_graph: RDFGraph,
            circuit_uri: URIRef,
            graph,  # EAP Graph
            metadata: Dict[str, Any],
    ):
        """
        Add Layer 3: Architectural signatures.

        IMPORTANT: This method computes HEURISTIC architectural features:
        - Relative depths (ρ_min, ρ_max, ρ̄) - computed from graph structure
        - Component ratios (attn:mlp) - computed from node counts
        - Motifs (backward attention, skip connections) - detected heuristically

        These are NOT the learned architectural signature τ_arch from ILP_arch.
        τ_arch would be added later via add_learned_architectural_signature().

        """

        circuit_nodes = [node for node in graph.nodes.values() if node.in_graph]

        if not circuit_nodes:
            return

        # Get model depth
        model_depth = metadata.get("n_layers")
        if model_depth is None:
            layers = []
            for node in circuit_nodes:
                structure = self.schema.parse_node_structure(node.name)
                if 'layer' in structure:
                    layers.append(structure['layer'])
            if layers:
                model_depth = max(layers) + 1

        # Compute relative depths
        relative_depths = []
        for node in circuit_nodes:
            structure = self.schema.parse_node_structure(node.name)
            if 'layer' in structure and model_depth:
                rel_depth = structure['layer'] / model_depth
                relative_depths.append(rel_depth)

                # Per-node depth
                node_uri = self._make_uri(f"node_{node.name}")
                rdf_graph.add((node_uri, self.namespace.hasRelativeDepth,
                               Literal(float(rel_depth), datatype=XSD.float)))

        # Aggregate depths
        if relative_depths:
            rdf_graph.add((circuit_uri, self.namespace.depthMin,
                           Literal(float(min(relative_depths)), datatype=XSD.float)))
            rdf_graph.add((circuit_uri, self.namespace.depthMax,
                           Literal(float(max(relative_depths)), datatype=XSD.float)))
            rdf_graph.add((circuit_uri, self.namespace.depthMean,
                           Literal(float(sum(relative_depths) / len(relative_depths)), datatype=XSD.float)))

        # Count components
        attn_count = sum(1 for n in circuit_nodes if n.name.startswith('a'))
        mlp_count = sum(1 for n in circuit_nodes if n.name.startswith('m'))

        rdf_graph.add((circuit_uri, self.namespace.numAttentionHeads,
                       Literal(attn_count, datatype=XSD.integer)))
        rdf_graph.add((circuit_uri, self.namespace.numMLPBlocks,
                       Literal(mlp_count, datatype=XSD.integer)))

        # Component ratio
        component_ratio = f"{attn_count}:{mlp_count}"
        rdf_graph.add((circuit_uri, self.namespace.componentRatio, Literal(component_ratio)))

        # Detect motifs
        motifs = self._detect_architectural_motifs(graph, circuit_nodes)
        for motif in motifs:
            rdf_graph.add((circuit_uri, self.namespace.hasMotif, motif))

        # Create signature object (placeholder for ILP_arch)
        # NOTE: This is a COMPUTED signature based on heuristic features.
        # True architectural signature (τ_arch) would be learned by ILP_arch
        # and added via add_learned_architectural_signature() (not yet implemented).
        sig_uri = self._make_uri(f"arch_sig_{circuit_uri.split('#')[-1]}")
        rdf_graph.add((sig_uri, RDF.type, self.namespace.ArchitecturalSignature))
        rdf_graph.add((circuit_uri, self.namespace.hasArchitecturalSignature, sig_uri))

        # Copy key properties to signature (computed features, not learned rules)
        if relative_depths:
            rdf_graph.add((sig_uri, self.namespace.depthMin,
                           Literal(float(min(relative_depths)), datatype=XSD.float)))
            rdf_graph.add((sig_uri, self.namespace.depthMax,
                           Literal(float(max(relative_depths)), datatype=XSD.float)))
        rdf_graph.add((sig_uri, self.namespace.componentRatio, Literal(component_ratio)))
        for motif in motifs:
            rdf_graph.add((sig_uri, self.namespace.hasMotif, motif))

        # signatureText would be added by ILP_arch (e.g., "depth ∈ [0.5, 0.8] ∧ attn:mlp = 3:1")
        # Currently not implemented

    def _detect_architectural_motifs(self, graph, circuit_nodes: List) -> Set[URIRef]:
        """
        Derive OWL motif URIs from topology-based _detect_motifs() output.

        This replaces the old aggregate-statistics heuristics.  Motif URIs
        are derived from the same topology walk used by _detect_motifs so
        the OWL triples and the ilp_arch ILP predicates stay aligned.

        Mapping (topology type → OWL class in mechint namespace):
            scaffold_event_entity  → ScaffoldEventEntity
            induction_head         → InductionHead
            token_copy             → TokenCopy
            ff_chain               → FeedForwardChain
            broadcast_head         → BroadcastHead
            attn_mlp_attn          → AttnMlpAttn
        """
        _TYPE_TO_OWL = {
            'scaffold_event_entity': self.namespace.ScaffoldEventEntity,
            'induction_head':        self.namespace.InductionHead,
            'token_copy':            self.namespace.TokenCopy,
            'ff_chain':              self.namespace.FeedForwardChain,
            'broadcast_head':        self.namespace.BroadcastHead,
            'attn_mlp_attn':         self.namespace.AttnMlpAttn,
        }

        raw_motifs = self._detect_motifs(graph)      # topology-based
        motif_uris: Set[URIRef] = set()
        for m in raw_motifs:
            uri = _TYPE_TO_OWL.get(m['type'])
            if uri:
                motif_uris.add(uri)

        if not motif_uris:
            motif_uris.add(self.namespace.Motif)     # generic fallback

        return motif_uris

    # ========================================
    # UTILITY METHODS
    # ========================================

    def _make_uri(self, local_name: str) -> URIRef:
        """Create URI in mechint namespace"""
        local_name = local_name.replace(" ", "_").replace(":", "_").replace(".", "_")
        return URIRef(f"{self.base_uri}#{local_name}")

    def _save_graph(self, rdf_graph: RDFGraph, experiment_id: str, step: int) -> Path:
        """Save to Turtle file"""
        filename = f"{experiment_id}_step_{step}.ttl"
        filepath = self.output_dir / filename

        rdf_graph.serialize(destination=str(filepath), format='turtle')

        return filepath

    # ========================================
    # step 1: ADD LEARNED SPECIFICATION
    # ========================================

    def add_learned_architectural_signature(
            self,
            circuit_ttl_path: str,
            signature_text: str,
            confidence: float,
    ) -> str:
        """
        Write ILP_arch learned τ_arch into an existing circuit TTL.

        Args:
            circuit_ttl_path: Path to circuit TTL file (already exported)
            signature_text: FOL signature rule from ILParch.learn_signature()
            confidence: arch confidence score

        Returns:
            Updated file path
        """
        rdf_graph = RDFGraph()
        rdf_graph.parse(circuit_ttl_path, format='turtle')
        rdf_graph.bind("mechint", self.namespace)

        # Find circuit URI
        circuit_uri = None
        for s in rdf_graph.subjects(RDF.type, self.namespace.Circuit):
            circuit_uri = s
            break
        if not circuit_uri:
            raise ValueError(f"No circuit found in {circuit_ttl_path}")

        # Find or create the ArchitecturalSignature node
        sig_uri = None
        for s in rdf_graph.subjects(RDF.type, self.namespace.ArchitecturalSignature):
            sig_uri = s
            break
        if sig_uri is None:
            # Create one if _add_architectural_layer didn't run (e.g. test context)
            sig_uri = self._make_uri(f"arch_sig_{circuit_uri.split('#')[-1]}")
            rdf_graph.add((sig_uri, RDF.type, self.namespace.ArchitecturalSignature))
            rdf_graph.add((circuit_uri, self.namespace.hasArchitecturalSignature, sig_uri))

        # Write learned τ_arch
        rdf_graph.add((sig_uri, self.namespace.signatureText, Literal(signature_text)))
        rdf_graph.add((sig_uri, self.namespace.confidence,
                       Literal(float(confidence), datatype=XSD.float)))
        rdf_graph.add((sig_uri, self.namespace.learnedBy, Literal("ILP_arch")))

        rdf_graph.serialize(destination=circuit_ttl_path, format='turtle')

        print(f"✓ Added arch signature: confidence={confidence:.3f}, sig={signature_text[:60]}...")
        return circuit_ttl_path

    def add_learned_specification(
            self,
            circuit_ttl_path: str,
            spec_text: str,
            predicates: str,
            confidence: float,
            ilp_signature: Dict[str, Any],
    ) -> str:
        """
        Add ILP_sem learned specification to existing circuit.

        Args:
            circuit_ttl_path: Path to circuit TTL file
            spec_text: FOL clause text
            predicates: Comma-separated predicates
            confidence: ILP confidence
            ilp_signature: Dict with n_clauses, avg_depth, etc.

        Returns:
            Updated file path
        """
        # Load existing
        rdf_graph = RDFGraph()
        rdf_graph.parse(circuit_ttl_path, format='turtle')
        rdf_graph.bind("mechint", self.namespace)

        # Find circuit
        circuit_uri = None
        for s in rdf_graph.subjects(RDF.type, self.namespace.Circuit):
            circuit_uri = s
            break

        if not circuit_uri:
            raise ValueError(f"No circuit in {circuit_ttl_path}")

        # Create spec
        spec_uri = self._make_uri(f"spec_{circuit_uri.split('#')[-1]}")
        rdf_graph.add((spec_uri, RDF.type, self.namespace.LearnedSpecification))
        rdf_graph.add((spec_uri, RDF.type, self.namespace.SemanticSpec))

        # Link
        rdf_graph.add((circuit_uri, self.namespace.hasSemanticSpec, spec_uri))

        # Properties
        rdf_graph.add((spec_uri, self.namespace.specText, Literal(spec_text)))

        # Store BOTH ways for compatibility:
        # 1. String (backward compatible)
        rdf_graph.add((spec_uri, self.namespace.predicates, Literal(predicates)))

        # 2. Structured (for set operations - Jaccard, θ-subsumption)
        self._add_structured_predicates(rdf_graph, spec_uri, predicates)

        rdf_graph.add((spec_uri, self.namespace.confidence,
                       Literal(float(confidence), datatype=XSD.float)))
        rdf_graph.add((spec_uri, self.namespace.learnedBy, Literal("ILP_sem")))

        # ILP signature
        if "n_clauses" in ilp_signature:
            rdf_graph.add((spec_uri, self.namespace.numClauses,
                           Literal(ilp_signature["n_clauses"], datatype=XSD.integer)))
        if "avg_depth" in ilp_signature:
            rdf_graph.add((spec_uri, self.namespace.avgClauseDepth,
                           Literal(float(ilp_signature["avg_depth"]), datatype=XSD.float)))
        if "avg_length" in ilp_signature:
            rdf_graph.add((spec_uri, self.namespace.avgClauseLength,
                           Literal(float(ilp_signature["avg_length"]), datatype=XSD.float)))
        if "complexity" in ilp_signature:
            rdf_graph.add((spec_uri, self.namespace.complexity,
                           Literal(float(ilp_signature["complexity"]), datatype=XSD.float)))

        # Infer computational pattern from learned predicates
        # This is the CORRECT place to assign patterns - after ILP learns them
        pattern = self._infer_pattern_from_predicates(predicates)
        rdf_graph.add((circuit_uri, self.namespace.hasComputationalPattern, pattern))

        # Save
        rdf_graph.serialize(destination=circuit_ttl_path, format='turtle')

        print(f"✓ Added spec: confidence={confidence:.3f}, predicates={predicates}, pattern={pattern.split('#')[-1]}")

        return circuit_ttl_path

    def _infer_pattern_from_predicates(self, predicates: str) -> URIRef:
        """
        Infer computational pattern from learned predicates.

        This implements the SPARQL rule :
        "hasComputationalPattern some BindingPattern ← predicates contains 'binds'"

        Args:
            predicates: Comma-separated predicate string (e.g., "binds, scaffold, compatible")

        Returns:
            URI of inferred computational pattern

        Note: This inference is based on LEARNED predicates from ILP_sem,
        not from task family labels (which would be heuristic guessing).
        """
        pred_lower = predicates.lower()

        # Pattern inference rules (from learned evidence)
        if 'binds' in pred_lower or 'binding' in pred_lower:
            return self.namespace.BindingPattern
        elif 'compares' in pred_lower or 'comparison' in pred_lower or 'greater' in pred_lower:
            return self.namespace.ComparisonPattern
        elif 'selects' in pred_lower or 'selection' in pred_lower or 'choose' in pred_lower:
            return self.namespace.SelectionPattern
        elif 'inhibits' in pred_lower or 'inhibition' in pred_lower or 'suppress' in pred_lower:
            return self.namespace.InhibitionPattern
        elif 'copies' in pred_lower or 'copy' in pred_lower or 'duplicate' in pred_lower:
            return self.namespace.CopyPattern
        elif 'extracts' in pred_lower or 'extraction' in pred_lower:
            return self.namespace.ExtractionPattern
        else:
            # Unknown pattern - use base class
            return self.namespace.ComputationalPattern

    def _add_structured_predicates(self, rdf_graph: RDFGraph, spec_uri: URIRef, predicates: str):
        """
        Add structured predicate set storage

        Enables efficient SPARQL set operations:
        - Jaccard similarity: |P1 ∩ P2| / |P1 ∪ P2|
        - θ-subsumption: P_A ⊆ P_B

        Args:
            rdf_graph: RDF graph to add to
            spec_uri: Specification URI
            predicates: Comma-separated predicate string (e.g., "binds, scaffold, compatible")
        """
        if not predicates or predicates.strip() == "":
            return

        # Parse predicate string
        predicate_list = [p.strip() for p in predicates.split(',') if p.strip()]

        for pred_name in predicate_list:
            # Create unique predicate URI
            pred_uri = self._make_uri(f"pred_{pred_name}_{hash(spec_uri) % 10000}")

            # Add predicate instance
            rdf_graph.add((pred_uri, RDF.type, self.namespace.Predicate))
            rdf_graph.add((pred_uri, self.namespace.predicateName, Literal(pred_name)))

            # Infer arity from predicate name (heuristic - could be improved with ILP metadata)
            arity = self._infer_predicate_arity(pred_name)
            if arity > 0:
                rdf_graph.add((pred_uri, self.namespace.predicateArity,
                              Literal(arity, datatype=XSD.integer)))

            # Link to specification (multi-valued property)
            rdf_graph.add((spec_uri, self.namespace.hasPredicateInSet, pred_uri))

    def _infer_predicate_arity(self, pred_name: str) -> int:
        """
        Infer predicate arity from name (heuristic).

        Args:
            pred_name: Predicate name (e.g., "binds", "scaffold")

        Returns:
            Arity (number of arguments), or -1 if unknown

        Note: This is a heuristic. Ideally, arity would come from ILP metadata.
        """
        # Common predicate arities (can be extended)
        arity_map = {
            'binds': 3,        # binds(circuit, entity, role)
            'scaffold': 2,     # scaffold(structure, marker)
            'compatible': 2,   # compatible(entity1, entity2)
            'selects': 2,      # selects(circuit, entity)
            'compares': 3,     # compares(circuit, entity1, entity2)
            'inhibits': 2,     # inhibits(circuit, entity)
            'copies': 2,       # copies(circuit, entity)
            'extracts': 2,     # extracts(circuit, feature)
        }

        # Check if predicate name matches known pattern
        pred_lower = pred_name.lower()
        for known_pred, arity in arity_map.items():
            if known_pred in pred_lower:
                return arity

        # Default: unknown arity
        return -1  # -1 indicates unknown


    # ── patch_kappa_tau ──────────────────────────────────────────────────────

    def patch_kappa_tau(
            self,
            circuit_ttl_path: str,
            operation_type: Optional[str] = None,
            logical_category: Optional[str] = None,
    ) -> str:
        """
        Add/update operationType (κ) and logicalCategory (τ) triples on an
        existing circuit TTL.  Called after analyze_and_export() when the
        installed analyzer.analyze() does not accept extra_metadata.

        Safe to call with None values — only emits triples for non-None inputs.
        """
        if not operation_type and not logical_category:
            return circuit_ttl_path

        rdf_graph = RDFGraph()
        rdf_graph.parse(circuit_ttl_path, format="turtle")
        rdf_graph.bind("mechint", self.namespace)

        circuit_uri = None
        for s in rdf_graph.subjects(RDF.type, self.namespace.Circuit):
            circuit_uri = s
            break
        if circuit_uri is None:
            return circuit_ttl_path   # nothing to patch

        if operation_type:
            rdf_graph.set((circuit_uri, self.namespace.operationType,
                           Literal(operation_type)))
        if logical_category:
            rdf_graph.set((circuit_uri, self.namespace.logicalCategory,
                           Literal(logical_category)))

        rdf_graph.serialize(destination=circuit_ttl_path, format="turtle")
        return circuit_ttl_path


# ========================================
# CONVENIENCE FUNCTIONS (module-level)
# ========================================

def export_eap_circuit_to_owl(
        graph,
        experiment_id: str,
        task_family: str,
        role_name: str,
        model_name: str,
        step: int = 0,
        eap_results: Dict[str, Any] = None,
        config: Dict[str, Any] = None,
        output_dir: str = "./data/experiments",
        operation_type: Optional[str] = None,   # κ: e.g. "binding", "selection"
        logical_category: Optional[str] = None, # τ: "PREDICATE" | "ROLE" | "OPERATOR"
) -> str:
    """
    Convenience function: export EAP circuit to OWL in one call.

    Example:
         owl_path = export_eap_circuit_to_owl(
             graph=circuit_graph,
             experiment_id="semantic_roles_pythia70m",
             task_family="SemanticRoles",
             role_name="INSTRUMENT",
             model_name="pythia-70m",
             step=500,
             eap_results={"faithfulness": 0.89, ...},
             config={"method": "EAP-IG-inputs", "top_k": 50},
             operation_type="binding",      # κ
             logical_category="ROLE",       # τ
         )
    """
    exporter = OWLExporter(output_dir=output_dir)

    return exporter.export_circuit(
        graph=graph,
        experiment_id=experiment_id,
        step=step,
        config=config or {},
        metadata={"model_name": model_name},
        eap_results=eap_results or {},
        task_family=task_family,
        role_or_phenomenon=role_name,
        operation_type=operation_type,
        logical_category=logical_category,
    )