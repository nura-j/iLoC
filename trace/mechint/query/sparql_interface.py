"""
SPARQL Query Interface - Updated for 4-Layer Schema
Provides query methods for:
- Layer 0: Provenance queries (tasks, experiments, models)
- Layer 1: Structural queries (components, edges, performance)
- Layer 2: Semantic queries (learned specs, patterns) - RQ1, RQ2
- Layer 3: Architectural queries (motifs, depths, signatures) - RQ3
"""

import rdflib
from rdflib import Graph as RDFGraph, Namespace, URIRef  # , Literal
from rdflib.namespace import RDF, RDFS, XSD
from pathlib import Path
from typing import List, Dict, Any, Optional, Union #, Set, Tuple
from trace.mechint.storage.pattern_recognition import PatternRecognizer

import warnings

warnings.filterwarnings('ignore')


class CircuitQueryEngine:
    """
    SPARQL-based query interface for mechanistic interpretability database
    Updated for 4-layer ontology schema
    """

    def __init__(self, data_dir: str = "./data/experiments"):
        self.data_dir = Path(data_dir)
        self.graph = RDFGraph()

        # Set up namespaces
        self.mechint_ns = Namespace("http://purl.org/mechint#")
        self.graph.bind("mechint", self.mechint_ns)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("xsd", XSD)
        self.pattern_recognizer = PatternRecognizer()
        self.loaded_experiments = []

    def load_experiments(self, experiment_ids: Optional[List[str]] = None):
        """Load circuit data from experiments"""
        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist")
            return False

        if experiment_ids is None:
            # Load all .ttl files recursively
            ttl_files = list(self.data_dir.rglob("*.ttl"))
        else:
            ttl_files = []
            for exp_id in experiment_ids:
                exp_dir = self.data_dir / exp_id
                if exp_dir.exists():
                    ttl_files.extend(exp_dir.rglob("*.ttl"))

        loaded_count = 0
        failed_count = 0

        for ttl_file in ttl_files:
            try:
                temp_graph = RDFGraph()
                temp_graph.parse(ttl_file, format='turtle')

                if len(temp_graph) > 0:
                    self.graph += temp_graph
                    loaded_count += 1
                    print(f" Loaded {ttl_file.name} ({len(temp_graph)} triples)")
                else:
                    print(f" Empty: {ttl_file.name}")

            except Exception as e:
                failed_count += 1
                print(f"X Failed {ttl_file.name}: {str(e)[:100]}")

        print(f"\n✓ Loaded {loaded_count} files, {failed_count} failed")
        print(f"✓ Total triples: {len(self.graph)}")

        return loaded_count > 0

    # ========================================
    # LAYER 0: PROVENANCE QUERIES
    # ========================================

    def find_circuits_by_task(self, task_family: str) -> List[Dict]:
        """Find all circuits for a specific task family"""
        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?circuit ?roleName ?modelName ?step ?faithfulness WHERE {{
            ?circuit a mechint:Circuit .
            ?circuit mechint:belongsToTaskFamily mechint:{task_family} .
            OPTIONAL {{ ?circuit mechint:roleName ?roleName }}
            OPTIONAL {{ ?circuit mechint:fromModel ?model .
                       ?model mechint:modelName ?modelName }}
            OPTIONAL {{ ?circuit mechint:atCheckpoint ?checkpoint .
                       ?checkpoint mechint:step ?step }}
            OPTIONAL {{ ?circuit mechint:faithfulness ?faithfulness }}
        }}
        ORDER BY ?roleName ?step
        """
        return self._execute_select_query(query)

    def find_circuits_by_role(self, role_name: str) -> List[Dict]:
        """Find all circuits for a specific role/phenomenon"""
        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?circuit ?taskFamily ?modelName ?step ?faithfulness WHERE {{
            ?circuit a mechint:Circuit .
            ?circuit mechint:roleName "{role_name}" .
            OPTIONAL {{ ?circuit mechint:belongsToTaskFamily ?taskFamily }}
            OPTIONAL {{ ?circuit mechint:fromModel ?model .
                       ?model mechint:modelName ?modelName }}
            OPTIONAL {{ ?circuit mechint:atCheckpoint ?checkpoint .
                       ?checkpoint mechint:step ?step }}
            OPTIONAL {{ ?circuit mechint:faithfulness ?faithfulness }}
        }}
        ORDER BY ?modelName ?step
        """
        return self._execute_select_query(query)

    def get_experiment_summary(self) -> List[Dict]:
        """Get overview of all experiments"""
        query = """
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?taskFamily ?roleName (COUNT(?circuit) as ?numCircuits) 
               (AVG(?faithfulness) as ?avgFaithfulness) 
        WHERE {
            ?circuit a mechint:Circuit .
            ?circuit mechint:belongsToTaskFamily ?taskFamily .
            OPTIONAL { ?circuit mechint:roleName ?roleName }
            OPTIONAL { ?circuit mechint:faithfulness ?faithfulness }
        }
        GROUP BY ?taskFamily ?roleName
        ORDER BY ?taskFamily ?roleName
        """
        return self._execute_select_query(query)

    # ========================================
    # LAYER 1: STRUCTURAL QUERIES
    # ========================================

    def find_high_faithfulness_circuits(self, min_faithfulness: float = 0.8) -> List[Dict]:
        """Find circuits with high faithfulness scores"""
        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?circuit ?taskFamily ?roleName ?faithfulness ?causalEffect 
               (COUNT(?node) as ?numNodes) (COUNT(?edge) as ?numEdges)
        WHERE {{
            ?circuit a mechint:Circuit .
            ?circuit mechint:faithfulness ?faithfulness .
            FILTER(?faithfulness >= {min_faithfulness})

            OPTIONAL {{ ?circuit mechint:belongsToTaskFamily ?taskFamily }}
            OPTIONAL {{ ?circuit mechint:roleName ?roleName }}
            OPTIONAL {{ ?circuit mechint:causalEffect ?causalEffect }}
            OPTIONAL {{ ?circuit mechint:hasMemberNode ?node }}
            OPTIONAL {{ ?circuit mechint:hasMemberEdge ?edge }}
        }}
        GROUP BY ?circuit ?taskFamily ?roleName ?faithfulness ?causalEffect
        ORDER BY DESC(?faithfulness)
        """
        return self._execute_select_query(query)

    def get_circuit_components(self, circuit_uri: str) -> List[Dict]:
        """Get all components (nodes) in a circuit"""
        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?node ?address ?layer ?head ?relativeDepth WHERE {{
            <{circuit_uri}> mechint:hasMemberNode ?node .
            ?node mechint:address ?address .
            OPTIONAL {{ ?node mechint:inLayer ?layer }}
            OPTIONAL {{ ?node mechint:head ?head }}
            OPTIONAL {{ ?node mechint:hasRelativeDepth ?relativeDepth }}
        }}
        ORDER BY ?layer ?head
        """
        return self._execute_select_query(query)

    def get_circuit_edges(self, circuit_uri: str) -> List[Dict]:
        """Get all edges in a circuit"""
        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?edge ?srcAddr ?tgtAddr ?score ?rank WHERE {{
            <{circuit_uri}> mechint:hasMemberEdge ?edge .
            ?edge mechint:hasSource ?src .
            ?edge mechint:hasTarget ?tgt .
            ?src mechint:address ?srcAddr .
            ?tgt mechint:address ?tgtAddr .
            ?edge mechint:score ?score .
            OPTIONAL {{ ?edge mechint:rank ?rank }}
        }}
        ORDER BY ?rank
        """
        return self._execute_select_query(query)

    # ========================================
    # LAYER 2: SEMANTIC QUERIES (RQ1 & RQ2)
    # ========================================

    def find_circuits_with_learned_specs(self) -> List[Dict]:
        """Find circuits that have learned ILP specifications """
        query = """
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?circuit ?roleName ?specText ?predicates ?confidence WHERE {
            ?circuit a mechint:Circuit .
            ?circuit mechint:hasSemanticSpec ?spec .
            ?spec mechint:specText ?specText .
            ?spec mechint:predicates ?predicates .
            ?spec mechint:confidence ?confidence .
            OPTIONAL { ?circuit mechint:roleName ?roleName }
        }
        ORDER BY DESC(?confidence)
        """
        return self._execute_select_query(query)

    def find_circuits_by_predicate(self, predicate: str) -> List[Dict]:
        """Find circuits whose learned specs use a specific predicate"""
        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?circuit ?roleName ?predicates ?confidence WHERE {{
            ?circuit a mechint:Circuit .
            ?circuit mechint:hasSemanticSpec ?spec .
            ?spec mechint:predicates ?predicates .
            ?spec mechint:confidence ?confidence .
            FILTER(CONTAINS(LCASE(?predicates), "{predicate.lower()}"))
            OPTIONAL {{ ?circuit mechint:roleName ?roleName }}
        }}
        ORDER BY DESC(?confidence)
        """
        return self._execute_select_query(query)

    def compute_predicate_jaccard_similarity(
            self,
            circuit_uri1: str,
            circuit_uri2: str
    ) -> float:
        """
        Compute Jaccard similarity between two circuits' predicates (RQ2)

        Returns:
            Jaccard similarity score [0, 1]
        """
        # Get predicates for both circuits
        query1 = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?predicates WHERE {{
            <{circuit_uri1}> mechint:hasSemanticSpec ?spec .
            ?spec mechint:predicates ?predicates .
        }}
        """
        query2 = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?predicates WHERE {{
            <{circuit_uri2}> mechint:hasSemanticSpec ?spec .
            ?spec mechint:predicates ?predicates .
        }}
        """

        result1 = self._execute_select_query(query1)
        result2 = self._execute_select_query(query2)

        if not result1 or not result2:
            return 0.0

        # Parse predicate lists
        preds1 = set(p.strip() for p in result1[0]['predicates'].split(','))
        preds2 = set(p.strip() for p in result2[0]['predicates'].split(','))

        # Compute Jaccard
        intersection = len(preds1 & preds2)
        union = len(preds1 | preds2)

        return intersection / union if union > 0 else 0.0

    def find_similar_circuits(
            self,
            circuit_uri: str,
            min_jaccard: float = 0.15
    ) -> List[Dict]:
        """
        Find circuits similar to given circuit (RQ2 - coherence analysis)

        Args:
            circuit_uri: URI of reference circuit
            min_jaccard: Minimum Jaccard similarity threshold

        Returns:
            List of similar circuits with similarity scores
        """
        # Get reference predicates
        query_ref = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?predicates WHERE {{
            <{circuit_uri}> mechint:hasSemanticSpec ?spec .
            ?spec mechint:predicates ?predicates .
        }}
        """
        result_ref = self._execute_select_query(query_ref)

        if not result_ref:
            return []

        ref_preds = set(p.strip() for p in result_ref[0]['predicates'].split(','))

        # Get all other circuits with specs
        query_all = """
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?circuit ?roleName ?predicates WHERE {
            ?circuit a mechint:Circuit .
            ?circuit mechint:hasSemanticSpec ?spec .
            ?spec mechint:predicates ?predicates .
            OPTIONAL { ?circuit mechint:roleName ?roleName }
        }
        """
        all_circuits = self._execute_select_query(query_all)

        # Compute similarities
        similar = []
        for circ in all_circuits:
            if circ['circuit'] == circuit_uri:
                continue  # Skip self

            circ_preds = set(p.strip() for p in circ['predicates'].split(','))
            intersection = len(ref_preds & circ_preds)
            union = len(ref_preds | circ_preds)
            jaccard = intersection / union if union > 0 else 0.0

            if jaccard >= min_jaccard:
                similar.append({
                    'circuit': circ['circuit'],
                    'roleName': circ.get('roleName'),
                    'jaccard_similarity': jaccard,
                    'shared_predicates': ', '.join(ref_preds & circ_preds),
                    'num_shared': intersection
                })

        # Sort by similarity
        similar.sort(key=lambda x: x['jaccard_similarity'], reverse=True)

        return similar

    def compute_predicate_jaccard_similarity_structurely(
            self,
            circuit_uri1: str,
            circuit_uri2: str
    ) -> float:
        pass

    def compute_task_coherence_matrix(self, task_family: str) -> Dict[str, Any]:
        """
        Compute pairwise coherence matrix for a task family (RQ2)

        Returns:
            Dict with coherence matrix and statistics
        """
        # Get all circuits in task family with specs
        circuits = self.find_circuits_by_task(task_family)
        # Build set of circuit URIs that have a learned spec (one query, not N)
        spec_circuit_uris = {
            c['circuit'] for c in self.find_circuits_with_learned_specs()
        }
        circuits_with_specs = [
            c for c in circuits
            if c['circuit'] in spec_circuit_uris
        ]

        if len(circuits_with_specs) < 2:
            return {'error': 'Need at least 2 circuits with specifications'}

        # Compute pairwise Jaccard
        matrix = {}
        for i, c1 in enumerate(circuits_with_specs):
            for j, c2 in enumerate(circuits_with_specs):
                if i <= j:  # Upper triangle + diagonal
                    jaccard = self.compute_predicate_jaccard_similarity(
                        c1['circuit'], c2['circuit']
                    )
                    key = f"{c1['roleName']}_{c2['roleName']}"
                    matrix[key] = jaccard

        # Compute statistics
        jaccards = [v for k, v in matrix.items() if 'circuit' not in k]
        stats = {
            'mean_jaccard': sum(jaccards) / len(jaccards) if jaccards else 0,
            'min_jaccard': min(jaccards) if jaccards else 0,
            'max_jaccard': max(jaccards) if jaccards else 0,
            'num_comparisons': len(jaccards)
        }

        return {
            'task_family': task_family,
            'matrix': matrix,
            'statistics': stats,
            'circuits': [c['roleName'] for c in circuits_with_specs]
        }

    # ========================================
    # LAYER 3: ARCHITECTURAL QUERIES (RQ3)
    # ========================================

    def find_circuits_by_relative_depth(
            self,
            target_depth: float,
            tolerance: float = 0.1
    ) -> List[Dict]:
        """Find circuits with similar relative depth (for transfer - RQ3)"""
        min_depth = target_depth - tolerance
        max_depth = target_depth + tolerance

        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?circuit ?roleName ?depthMean ?componentRatio WHERE {{
            ?circuit a mechint:Circuit .
            ?circuit mechint:depthMean ?depthMean .
            FILTER(?depthMean >= {min_depth} && ?depthMean <= {max_depth})
            OPTIONAL {{ ?circuit mechint:roleName ?roleName }}
            OPTIONAL {{ ?circuit mechint:componentRatio ?componentRatio }}
        }}
        ORDER BY ABS(?depthMean - {target_depth})
        """
        return self._execute_select_query(query)

    def find_circuits_by_motif(self, motif_name: str) -> List[Dict]:
        """Find circuits with a specific motif"""
        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?circuit ?roleName ?depthMean ?componentRatio WHERE {{
            ?circuit a mechint:Circuit .
            ?circuit mechint:hasMotif mechint:{motif_name} .
            OPTIONAL {{ ?circuit mechint:roleName ?roleName }}
            OPTIONAL {{ ?circuit mechint:depthMean ?depthMean }}
            OPTIONAL {{ ?circuit mechint:componentRatio ?componentRatio }}
        }}
        """
        return self._execute_select_query(query)

    def find_transfer_candidates(
            self,
            source_circuit_uri: str,
            depth_tolerance:   float = 0.15,
            size_tolerance:    int   = 15,
            require_same_motifs: bool = True,
            delta_ilp:         float = 0.30,
    ) -> List[Dict]:
        """
        Find candidate circuits for transfer from source (paper §3.6 Steps 1–2).

        Uses the full S_circuit(Cα) = {depth_mean, depth_min, depth_max,
        component_ratio, num_attn, num_mlp, motifs} for filtering, not
        just depth.  Motif matching uses Jaccard overlap rather than exact
        set equality so partial matches are ranked rather than discarded.

        Parameters
        ----------
        source_circuit_uri : URI of the source circuit in this graph
        depth_tolerance    : ±tolerance on depthMean (relative, 0–1)
        size_tolerance     : ±tolerance on total node count
        require_same_motifs: if True, require motif Jaccard > 0 (at least
                             one shared motif); if False, accept any depth match
        delta_ilp          : δ_ILP for ILP signature distance filter
        """
        # ── Fetch full S_circuit for source ──────────────────────────────────
        query_source = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?depthMean ?depthMin ?depthMax ?componentRatio
               ?numAttn ?numMlp ?motif WHERE {{
            <{source_circuit_uri}> mechint:depthMean ?depthMean .
            OPTIONAL {{ <{source_circuit_uri}> mechint:depthMin ?depthMin }}
            OPTIONAL {{ <{source_circuit_uri}> mechint:depthMax ?depthMax }}
            OPTIONAL {{ <{source_circuit_uri}> mechint:componentRatio ?componentRatio }}
            OPTIONAL {{ <{source_circuit_uri}> mechint:numAttentionHeads ?numAttn }}
            OPTIONAL {{ <{source_circuit_uri}> mechint:numMLPBlocks ?numMlp }}
            OPTIONAL {{ <{source_circuit_uri}> mechint:hasMotif ?motif }}
        }}
        """
        src_rows = self._execute_select_query(query_source)
        if not src_rows:
            return []

        depth      = float(src_rows[0].get('depthMean', 0.5) or 0.5)
        depth_min  = float(src_rows[0].get('depthMin',  0.0) or 0.0)
        depth_max  = float(src_rows[0].get('depthMax',  1.0) or 1.0)
        ratio      = src_rows[0].get('componentRatio', '') or ''
        n_attn     = int(src_rows[0].get('numAttn', 0) or 0)
        n_mlp      = int(src_rows[0].get('numMlp',  0) or 0)
        src_size   = n_attn + n_mlp
        src_motifs = {str(r['motif']) for r in src_rows if r.get('motif')}

        # ── Fetch all candidates within depth window ───────────────────────
        min_d = depth - depth_tolerance
        max_d = depth + depth_tolerance

        query_candidates = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT DISTINCT ?circuit ?roleName ?modelName
                        ?depthMean ?componentRatio ?numAttn ?numMlp WHERE {{
            ?circuit a mechint:Circuit .
            ?circuit mechint:depthMean ?depthMean .
            FILTER(?depthMean >= {min_d} && ?depthMean <= {max_d})
            FILTER(?circuit != <{source_circuit_uri}>)
            OPTIONAL {{ ?circuit mechint:roleName ?roleName }}
            OPTIONAL {{ ?circuit mechint:componentRatio ?componentRatio }}
            OPTIONAL {{ ?circuit mechint:numAttentionHeads ?numAttn }}
            OPTIONAL {{ ?circuit mechint:numMLPBlocks ?numMlp }}
            OPTIONAL {{ ?circuit mechint:fromModel ?model .
                       ?model mechint:modelName ?modelName }}
        }}
        ORDER BY ABS(?depthMean - {depth})
        LIMIT 50
        """
        raw = self._execute_select_query(query_candidates)

        # ── Fetch motifs for each candidate ────────────────────────────────
        cand_uris = list({r['circuit'] for r in raw if r.get('circuit')})
        cand_motifs: dict = {}
        if cand_uris:
            values_block = " ".join(f"<{u}>" for u in cand_uris)
            motif_q = f"""
            PREFIX mechint: <http://purl.org/mechint#>
            SELECT ?circuit ?motif WHERE {{
                VALUES ?circuit {{ {values_block} }}
                ?circuit mechint:hasMotif ?motif .
            }}
            """
            for row in self._execute_select_query(motif_q):
                uri = str(row.get('circuit', ''))
                m   = str(row.get('motif', ''))
                cand_motifs.setdefault(uri, set()).add(m)

        # ── Score and filter ───────────────────────────────────────────────
        candidates = []
        for r in raw:
            uri        = str(r.get('circuit', ''))
            c_depth    = float(r.get('depthMean', 0.5) or 0.5)
            c_n_attn   = int(r.get('numAttn', 0) or 0)
            c_n_mlp    = int(r.get('numMlp',  0) or 0)
            c_size     = c_n_attn + c_n_mlp
            c_ratio    = r.get('componentRatio', '') or ''
            c_motifs   = cand_motifs.get(uri, set())

            # Depth diff
            depth_diff = abs(c_depth - depth)

            # Size diff
            size_diff  = abs(c_size - src_size)

            # Motif Jaccard
            union = src_motifs | c_motifs
            inter = src_motifs & c_motifs
            motif_jaccard = len(inter) / len(union) if union else 1.0

            # Apply filters
            if size_diff > size_tolerance:
                continue
            if require_same_motifs and src_motifs and motif_jaccard == 0:
                continue

            # Composite similarity score  (higher = better match)
            ratio_bonus   = 0.10 if c_ratio == ratio else 0.0
            similarity    = (
                (1.0 - depth_diff / depth_tolerance) * 0.50 +
                (1.0 - size_diff  / max(size_tolerance, 1)) * 0.20 +
                motif_jaccard * 0.20 +
                ratio_bonus   * 0.10
            )

            candidates.append({
                'circuit':          uri,
                'roleName':         r.get('roleName'),
                'modelName':        r.get('modelName'),
                'depthMean':        c_depth,
                'componentRatio':   c_ratio,
                'numAttn':          c_n_attn,
                'numMlp':           c_n_mlp,
                'motifs':           list(c_motifs),
                # Scoring details
                'depth_diff':       round(depth_diff, 4),
                'size_diff':        size_diff,
                'motif_jaccard':    round(motif_jaccard, 3),
                'similarity_score': round(similarity, 4),
                # Source reference
                'source_depth':     depth,
                'source_ratio':     ratio,
                'source_motifs':    list(src_motifs),
            })

        candidates.sort(key=lambda c: -c['similarity_score'])
        return candidates[:20]

    def get_architectural_signatures(self) -> List[Dict]:
        """Get architectural signatures for all circuits"""
        query = """
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?circuit ?roleName ?depthMin ?depthMax ?depthMean 
               ?componentRatio ?numAttentionHeads ?numMLPBlocks
        WHERE {
            ?circuit a mechint:Circuit .
            OPTIONAL { ?circuit mechint:roleName ?roleName }
            OPTIONAL { ?circuit mechint:depthMin ?depthMin }
            OPTIONAL { ?circuit mechint:depthMax ?depthMax }
            OPTIONAL { ?circuit mechint:depthMean ?depthMean }
            OPTIONAL { ?circuit mechint:componentRatio ?componentRatio }
            OPTIONAL { ?circuit mechint:numAttentionHeads ?numAttentionHeads }
            OPTIONAL { ?circuit mechint:numMLPBlocks ?numMLPBlocks }
        }
        ORDER BY ?depthMean
        """
        return self._execute_select_query(query)

    # ========================================
    # DATABASE STATISTICS
    # ========================================

    def get_database_stats(self) -> Dict[str, int]:
        """Get comprehensive database statistics"""
        stats = {
            'total_triples': len(self.graph),
            'loaded_experiments': len(self.loaded_experiments)
        }

        # Count entities
        entity_counts = {
            'circuits': 'mechint:Circuit',
            'nodes': 'mechint:Component',
            'edges': 'mechint:Edge',
            'learned_specs': 'mechint:LearnedSpecification',
            'architectural_signatures': 'mechint:ArchitecturalSignature',
        }

        for entity_name, entity_type in entity_counts.items():
            query = f"""
            PREFIX mechint: <http://purl.org/mechint#>
            SELECT (COUNT(DISTINCT ?x) as ?count) WHERE {{
                ?x a {entity_type} .
            }}
            """
            result = self._execute_select_query(query)
            stats[f'num_{entity_name}'] = result[0]['count'] if result else 0

        return stats

    # ========================================
    # UTILITY METHODS
    # ========================================

    def _clean_uri(self, uri_ref: Union[rdflib.URIRef, str, None]) -> str:
        """Extract clean name from URIRef"""
        if uri_ref is None:
            return "None"

        if isinstance(uri_ref, rdflib.URIRef):
            uri_str = str(uri_ref)
            if '#' in uri_str:
                return uri_str.split('#')[-1]
            elif '/' in uri_str:
                return uri_str.split('/')[-1]

        return str(uri_ref)

    def _execute_select_query(self, query: str) -> List[Dict]:
        """Execute SPARQL SELECT query and return results"""
        try:
            results = self.graph.query(query)
            processed_results = []

            for row in results:
                row_dict = {}
                if hasattr(row, 'asdict'):
                    items = row.asdict().items()
                else:
                    items = [(str(var), row[var]) for var in results.vars]

                for var, value in items:
                    if value is None:
                        row_dict[var] = None
                    elif isinstance(value, rdflib.Literal):
                        try:
                            row_dict[var] = value.toPython()
                        except:
                            row_dict[var] = str(value)
                    elif isinstance(value, rdflib.URIRef):
                        row_dict[var] = self._clean_uri(value)
                    else:
                        row_dict[var] = str(value)

                processed_results.append(row_dict)

            return processed_results

        except Exception as e:
            print(f"Query failed: {e}")
            print(f"Query: {query[:200]}...")
            return []

    def find_minimal_circuits(self, min_pvalue: float = 0.05) -> List[Dict]:
        """
        Find circuits that are ε-minimal and statistically significant.

        Uses NEW properties from Ppaper
        - epsilonMinimal (bool)
        - permutationPValue (float)

        Args:
            min_pvalue: Maximum p-value threshold (default 0.05 for α=0.05)

        Returns:
            List of validated circuits with statistics
        """
        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>

        SELECT ?circuit ?epsilon ?pvalue ?causal ?faithful ?taskAcc WHERE {{
            ?circuit a mechint:Circuit .

            ?circuit mechint:epsilonMinimal ?epsilon .
            ?circuit mechint:permutationPValue ?pvalue .

            # Performance metrics
            OPTIONAL {{ ?circuit mechint:causalEffect ?causal }}
            OPTIONAL {{ ?circuit mechint:faithfulness ?faithful }}
            OPTIONAL {{ ?circuit mechint:taskAccuracy ?taskAcc }}

            # Filter for statistical significance
            FILTER(?epsilon = true && ?pvalue < {min_pvalue})
        }}
        ORDER BY ?pvalue
        """

        return self._execute_select_query(query)

    def get_validation_statistics(self, circuit_uri: str) -> Dict:
        """
        Get complete validation statistics κ for a circuit.

        Returns κ = ⟨Acc_T, Δ_T, confidence, ε-minimality, p-value⟩
        """
        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>

        SELECT ?accT ?deltaT ?confidence ?epsilon ?pvalue WHERE {{
            <{circuit_uri}> mechint:taskAccuracy ?accT .
            <{circuit_uri}> mechint:causalEffect ?deltaT .
            OPTIONAL {{ <{circuit_uri}> mechint:epsilonMinimal ?epsilon }}
            OPTIONAL {{ <{circuit_uri}> mechint:permutationPValue ?pvalue }}

            # Confidence from semantic spec (if learned)
            OPTIONAL {{
                <{circuit_uri}> mechint:hasSemanticSpec ?spec .
                ?spec mechint:confidence ?confidence .
            }}
        }}
        """

        results = self._execute_select_query(query)
        return results[0] if results else {}

    def get_complete_metrics(self, circuit_uri: str) -> Dict:
        """

        Returns all performance metrics with defaults
        """
        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>

        SELECT * WHERE {{
            <{circuit_uri}> a mechint:Circuit .

            # Core performance (always present)
            OPTIONAL {{ <{circuit_uri}> mechint:baselinePerformance ?baseline }}
            OPTIONAL {{ <{circuit_uri}> mechint:circuitPerformance ?circuit }}
            OPTIONAL {{ <{circuit_uri}> mechint:ablatedPerformance ?ablated }}

            # Faithfulness
            OPTIONAL {{ <{circuit_uri}> mechint:faithfulness ?faithful }}
            OPTIONAL {{ <{circuit_uri}> mechint:faithfulnessSimple ?faithfulSimple }}

            # Causal metrics
            OPTIONAL {{ <{circuit_uri}> mechint:causalEffect ?causal }}
            OPTIONAL {{ <{circuit_uri}> mechint:performanceDrop ?drop }}
            OPTIONAL {{ <{circuit_uri}> mechint:performanceDropPct ?dropPct }}

            # Validation statistics κ (NEW)
            OPTIONAL {{ <{circuit_uri}> mechint:taskAccuracy ?taskAcc }}
            OPTIONAL {{ <{circuit_uri}> mechint:epsilonMinimal ?epsilon }}
            OPTIONAL {{ <{circuit_uri}> mechint:permutationPValue ?pvalue }}
        }}
        """

        results = self._execute_select_query(query)
        return results[0] if results else {}

    # ========================================
    # STRUCTURED PREDICATE QUERIES
    # ========================================

    def find_specs_with_predicate(self, predicate_name: str) -> List[Dict]:
        """
        Find specifications that use a specific predicate.

        Uses NEW structured P_H storage from paper

        Args:
            predicate_name: Name of predicate (e.g., "binds", "scaffold")

        Returns:
            List of specifications using that predicate
        """
        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>

        SELECT ?spec ?circuit ?specText ?confidence WHERE {{
            ?spec a mechint:SemanticSpec .
            ?spec mechint:hasPredicateInSet ?pred .
            ?pred mechint:predicateName "{predicate_name}" .

            # Get specification details
            ?circuit mechint:hasSemanticSpec ?spec .
            ?spec mechint:specText ?specText .
            OPTIONAL {{ ?spec mechint:confidence ?confidence }}
        }}
        ORDER BY DESC(?confidence)
        """

        return self._execute_select_query(query)

    def get_predicate_usage_stats(self) -> Dict[str, int]:
        """
        Get statistics on predicate usage across all specifications.

        Returns:
            Dict mapping predicate names to usage counts
        """
        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>

        SELECT ?predName (COUNT(?spec) as ?count) WHERE {{
            ?spec mechint:hasPredicateInSet ?pred .
            ?pred mechint:predicateName ?predName .
        }}
        GROUP BY ?predName
        ORDER BY DESC(?count)
        """

        results = self._execute_select_query(query)
        return {str(r['predName']): int(r['count']) for r in results}

    def compute_jaccard_similarity_query(self, spec1_uri: str, spec2_uri: str) -> float:
        """
        Compute Jaccard similarity using SPARQL .

        Jaccard(P1, P2) = |P1 ∩ P2| / |P1 ∪ P2|
        """
        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>

        SELECT 
            (COUNT(DISTINCT ?p1) as ?size1)
            (COUNT(DISTINCT ?p2) as ?size2)
            (COUNT(DISTINCT ?shared) as ?intersection)
        WHERE {{
            # Get predicates from spec1
            <{spec1_uri}> mechint:hasPredicateInSet ?pred1 .
            ?pred1 mechint:predicateName ?name1 .
            BIND(?pred1 as ?p1)

            # Get predicates from spec2
            <{spec2_uri}> mechint:hasPredicateInSet ?pred2 .
            ?pred2 mechint:predicateName ?name2 .
            BIND(?pred2 as ?p2)

            # Find shared predicates by name
            OPTIONAL {{
                <{spec1_uri}> mechint:hasPredicateInSet ?s1 .
                ?s1 mechint:predicateName ?sharedName .
                <{spec2_uri}> mechint:hasPredicateInSet ?s2 .
                ?s2 mechint:predicateName ?sharedName .
                BIND(?s1 as ?shared)
            }}
        }}
        """

        result = self._execute_select_query(query)
        if result:
            size1 = int(result[0]['size1'])
            size2 = int(result[0]['size2'])
            intersection = int(result[0]['intersection'])
            union = size1 + size2 - intersection
            return intersection / union if union > 0 else 0.0
        return 0.0

    # ========================================
    # ENHANCED QUALITY QUERIES
    # ========================================

    def find_high_quality_circuits(
            self,
            min_faithfulness: float = 0.8,
            min_causal_effect: float = 0.05,
            max_pvalue: float = 0.05,
            require_validated: bool = True
    ) -> List[Dict]:
        """
        Find high-quality circuits based on multiple criteria.

        Args:
            min_faithfulness: Minimum faithfulness threshold
            min_causal_effect: Minimum causal effect (Δ_T)
            max_pvalue: Maximum p-value for significance
            require_validated: Whether to require ε-minimality

        Returns:
            List of high-quality circuits with all metrics
        """
        epsilon_filter = "FILTER(?epsilon = true)" if require_validated else ""

        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>

        SELECT ?circuit ?faithful ?causal ?pvalue 
               ?baseline ?circuitPerf ?ablated
               ?epsilon ?taskAcc
        WHERE {{
            ?circuit a mechint:Circuit .

            # Core metrics
            ?circuit mechint:faithfulness ?faithful .
            ?circuit mechint:causalEffect ?causal .
            ?circuit mechint:baselinePerformance ?baseline .
            ?circuit mechint:circuitPerformance ?circuitPerf .

            # Optional new metrics
            OPTIONAL {{ ?circuit mechint:ablatedPerformance ?ablated }}
            OPTIONAL {{ ?circuit mechint:permutationPValue ?pvalue }}
            OPTIONAL {{ ?circuit mechint:epsilonMinimal ?epsilon }}
            OPTIONAL {{ ?circuit mechint:taskAccuracy ?taskAcc }}

            # Quality filters
            FILTER(?faithful >= {min_faithfulness})
            FILTER(?causal >= {min_causal_effect})
            FILTER(!bound(?pvalue) || ?pvalue < {max_pvalue})
            {epsilon_filter}
        }}
        ORDER BY DESC(?faithful) DESC(?causal)
        """

        return self._execute_select_query(query)

    # ========================================
    # PATTERN RECOGNITION INTEGRATION
    # ========================================

    def analyze_patterns(self) -> Dict[str, Any]:
        """
        Run pattern recognition on loaded circuits.

        Uses PatternRecognizer to extract patterns from OWL data.

        Returns:
            Comprehensive pattern analysis report
        """
        print("\nRunning pattern recognition on loaded circuits...")
        patterns = self.pattern_recognizer.analyze_all_patterns(self.graph)

        print("\n" + "=" * 60)
        print("PATTERN RECOGNITION RESULTS")
        print("=" * 60)

        # Component patterns
        if 'component_patterns' in patterns:
            comp = patterns['component_patterns']
            print(f"\nComponent Patterns ({comp['total_circuits']} circuits):")
            print(f"  Avg components per circuit: {comp['avg_components_per_circuit']:.1f}")
            print(f"  Most frequent components:")
            for item, count in comp['frequent_components'][:5]:
                print(f"    {item}: {count}")

        # Architectural patterns
        if 'architectural_patterns' in patterns:
            arch = patterns['architectural_patterns']
            print(f"\nArchitectural Patterns:")
            print(f"  Common ratios:")
            for ratio, count in arch['common_ratios'][:5]:
                print(f"    {ratio}: {count} circuits")
            print(f"  Common motifs:")
            for motif, count in arch['common_motifs']:
                print(f"    {motif}: {count} circuits")

        # Performance patterns
        if 'performance_patterns' in patterns:
            perf = patterns['performance_patterns']
            tiers = perf['performance_tiers']
            print(f"\nPerformance Distribution:")
            print(f"  High (≥0.8): {tiers['high']} circuits")
            print(f"  Medium (0.5-0.8): {tiers['medium']} circuits")
            print(f"  Low (<0.5): {tiers['low']} circuits")

            if tiers['high'] > 0:
                high = perf['high_performer_characteristics']
                print(f"  High performer characteristics:")
                print(f"    Avg depth: {high['avg_depth']:.3f}")
                print(f"    Avg causal effect: {high['avg_causal_effect']:.4f}")

        # Task patterns
        if 'task_patterns' in patterns:
            print(f"\nTask-Specific Patterns:")
            for task, data in patterns['task_patterns'].items():
                if data['num_circuits'] > 0:
                    print(f"  {task}: {data['num_circuits']} circuits")
                    print(f"    Avg depth: {data['avg_depth']:.3f}")
                    print(f"    Avg faithfulness: {data['avg_faithfulness']:.3f}")
                    if data['common_ratios']:
                        top_ratio = data['common_ratios'][0]
                        print(f"    Most common ratio: {top_ratio[0]} ({top_ratio[1]} circuits)")

        return patterns

    def find_pattern_based_recommendations(
            self,
            target_task: str,
            min_circuits: int = 3
    ) -> Dict[str, Any]:
        """
        Get architectural recommendations based on pattern recognition.

        Analyzes successful circuits for a task and recommends:
        - Best component ratio
        - Optimal depth range
        - Useful motifs

        Args:
            target_task: Task family to analyze
            min_circuits: Minimum circuits needed for reliable patterns

        Returns:
            Dict with recommendations
        """
        patterns = self.pattern_recognizer.analyze_all_patterns(self.graph)

        if target_task not in patterns.get('task_patterns', {}):
            return {'error': f'No data for task: {target_task}'}

        task_data = patterns['task_patterns'][target_task]

        if task_data['num_circuits'] < min_circuits:
            return {
                'error': f'Insufficient data (need {min_circuits}, have {task_data["num_circuits"]})'
            }

        recommendations = {
            'task': target_task,
            'based_on_circuits': task_data['num_circuits'],
            'recommended_depth': task_data['avg_depth'],
            'expected_faithfulness': task_data['avg_faithfulness'],
            'recommended_ratios': task_data['common_ratios'],
            'confidence': 'HIGH' if task_data['num_circuits'] >= 10 else 'MEDIUM'
        }

        return recommendations

    # ========================================
    # COMBINED QUERIES (Pattern + Comparison + Transfer)
    # ========================================

    def find_similar_circuits_comprehensive(
            self,
            source_uri: str,
            use_patterns: bool = True,
            use_comparison: bool = True,
            use_transfer: bool = True
    ) -> List[Dict]:
        """
        Find similar circuits using all available methods.

        Combines:
        - Pattern recognition (structural similarity)
        - Comparison module (semantic similarity via Jaccard)
        - Transfer module (architectural similarity)

        Args:
            source_uri: Source circuit URI
            use_patterns: Include pattern-based similarity
            use_comparison: Include semantic spec comparison
            use_transfer: Include architectural transfer matching

        Returns:
            Ranked list of similar circuits with multiple similarity scores
        """
        from trace.mechint.storage.comparison import SpecificationComparator
        from trace.mechint.storage.transfer import TransferMatcher

        results = []

        # Get all candidate circuits
        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT DISTINCT ?circuit WHERE {{
            ?circuit a mechint:Circuit .
            FILTER(?circuit != <{source_uri}>)
        }}
        """

        candidates = self._execute_select_query(query)

        # Initialize modules
        comp = SpecificationComparator() if use_comparison else None
        matcher = TransferMatcher() if use_transfer else None

        for cand in candidates:
            cand_uri = str(cand['circuit'])
            scores = {'circuit': cand_uri}

            # Semantic similarity (via specs)
            if use_comparison and comp:
                try:
                    # Get specs
                    src_spec_q = f'SELECT ?spec WHERE {{ <{source_uri}> mechint:hasSemanticSpec ?spec }}'
                    cand_spec_q = f'SELECT ?spec WHERE {{ <{cand_uri}> mechint:hasSemanticSpec ?spec }}'

                    src_spec = self._execute_select_query(src_spec_q)
                    cand_spec = self._execute_select_query(cand_spec_q)

                    if src_spec and cand_spec:
                        jaccard = comp.compute_jaccard_similarity(
                            self.graph,
                            src_spec[0]['spec'],
                            cand_spec[0]['spec']
                        )
                        scores['semantic_similarity'] = jaccard
                except:
                    scores['semantic_similarity'] = None

            # Architectural similarity (via transfer)
            if use_transfer and matcher:
                try:
                    src_features = matcher.get_architectural_features(self.graph, URIRef(source_uri))
                    cand_features = matcher.get_architectural_features(self.graph, URIRef(cand_uri))

                    arch_sim = matcher.compute_architectural_similarity(src_features, cand_features)
                    scores['architectural_similarity'] = arch_sim
                except:
                    scores['architectural_similarity'] = None

            # Combined score
            valid_scores = [v for v in [
                scores.get('semantic_similarity'),
                scores.get('architectural_similarity')
            ] if v is not None]

            if valid_scores:
                scores['combined_score'] = sum(valid_scores) / len(valid_scores)
                results.append(scores)

        # Sort by combined score
        results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)

        return results

    # ========================================
    # REPORTING
    # ========================================

    def generate_circuit_report(self, circuit_uri: str) -> str:
        """
        Generate comprehensive report for a circuit.

        Includes:
        - Validation statistics
        - Semantic specification (if exists)
        - Pattern analysis
        - Similar circuits

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append(f"CIRCUIT REPORT: {circuit_uri}")
        report.append("=" * 60)

        # Metrics
        metrics = self.get_complete_metrics(circuit_uri)
        report.append("\nPERFORMANCE METRICS:")
        report.append(f"  Baseline: {metrics.get('baseline', 'N/A')}")
        report.append(f"  Circuit: {metrics.get('circuit', 'N/A')}")
        report.append(f"  Faithfulness: {metrics.get('faithful', 'N/A')}")
        report.append(f"  Causal Effect: {metrics.get('causal', 'N/A')}")

        # Validation
        validation = self.get_validation_statistics(circuit_uri)
        if validation:
            report.append("\nVALIDATION STATISTICS (κ):")
            report.append(f"  Task Accuracy (Acc_T): {validation.get('accT', 'N/A')}")
            report.append(f"  Causal Effect (Δ_T): {validation.get('deltaT', 'N/A')}")
            report.append(f"  ε-minimal: {validation.get('epsilon', 'N/A')}")
            report.append(f"  p-value: {validation.get('pvalue', 'N/A')}")
            report.append(f"  Confidence: {validation.get('confidence', 'N/A')}")

        # Specification
        spec_query = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?specText ?confidence WHERE {{
            <{circuit_uri}> mechint:hasSemanticSpec ?spec .
            OPTIONAL {{ ?spec mechint:specText ?specText }}
            OPTIONAL {{ ?spec mechint:confidence ?confidence }}
        }}
        """
        spec = self._execute_select_query(spec_query)
        if spec:
            report.append("\nSEMANTIC SPECIFICATION:")
            report.append(f"  {spec[0].get('specText', 'N/A')}")
            report.append(f"  Confidence: {spec[0].get('confidence', 'N/A')}")

        return "\n".join(report)


# ========================================
# CONVENIENCE FUNCTIONS
# ========================================

def quick_query(data_dir: str = "./data/experiments", task_family: str = None) -> Dict[str, Any]:
    """Quick database query for debugging"""
    engine = CircuitQueryEngine(data_dir)

    if not engine.load_experiments():
        return {'error': 'No experiments loaded'}

    stats = engine.get_database_stats()

    if task_family:
        circuits = engine.find_circuits_by_task(task_family)
        return {
            'database_stats': stats,
            'circuits': circuits,
            'num_circuits': len(circuits)
        }

    return {
        'database_stats': stats,
        'experiment_summary': engine.get_experiment_summary()
    }
