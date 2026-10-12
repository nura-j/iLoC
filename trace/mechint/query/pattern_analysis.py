# trace/mechint/query/pattern_analysis.py

"""
Advanced analysis utilities for mechanistic interpretability database
Includes pattern mining, ILP reasoning, and aggregate analysis functions
"""

from typing import Dict, List, Any
from collections import defaultdict
from pathlib import Path
import json

from rdflib import Graph as RDFGraph, Namespace, Literal
from rdflib.namespace import RDF, RDFS, XSD


class CircuitPatternMiner:
    """Mine common structural patterns from circuits"""

    def __init__(self, query_engine):
        self.engine = query_engine
        self.mechint_ns = Namespace("http://purl.org/mechint#")

    def find_attention_head_patterns(self) -> Dict[str, Any]:
        """Identify common attention head usage patterns"""
        query = """
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?layer ?head (COUNT(DISTINCT ?run) as ?appearances) 
               (AVG(?score) as ?avgScore) WHERE {
            ?edge a mechint:CircuitEdge .
            ?edge mechint:hasTarget ?node .
            ?edge mechint:inCircuit true .
            ?edge mechint:score ?score .
            ?edge mechint:inRun ?run .

            ?node mechint:aboutComponent ?comp .
            ?comp a mechint:AttentionHead .
            ?comp mechint:layer ?layer .
            ?comp mechint:head ?head .
        }
        GROUP BY ?layer ?head
        ORDER BY DESC(?appearances)
        """

        results = self.engine._execute_select_query(query)

        # Identify "core" attention heads (appear in >75% of circuits)
        if not results:
            return {'core_heads': [], 'peripheral_heads': [], 'stats': {}}

        max_appearances = max(r['appearances'] for r in results if r.get('appearances'))
        threshold = max_appearances * 0.75

        core_heads = [r for r in results if r.get('appearances', 0) >= threshold]
        peripheral_heads = [r for r in results if r.get('appearances', 0) < threshold]

        return {
            'core_heads': core_heads,
            'peripheral_heads': peripheral_heads,
            'stats': {
                'total_unique_heads': len(results),
                'core_head_count': len(core_heads),
                'peripheral_head_count': len(peripheral_heads),
                'max_appearances': max_appearances
            }
        }

    def find_mlp_patterns(self) -> Dict[str, Any]:
        """Identify MLP layer usage patterns"""
        query = """
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?layer (COUNT(DISTINCT ?run) as ?appearances) 
               (AVG(?score) as ?avgScore) WHERE {
            ?edge a mechint:CircuitEdge .
            ?edge mechint:hasTarget ?node .
            ?edge mechint:inCircuit true .
            ?edge mechint:score ?score .
            ?edge mechint:inRun ?run .

            ?node mechint:aboutComponent ?comp .
            ?comp a mechint:MLP .
            ?comp mechint:layer ?layer .
        }
        GROUP BY ?layer
        ORDER BY ?layer
        """

        return self.engine._execute_select_query(query)

    def find_skip_connections(self) -> List[Dict]:
        """Find skip connections (edges that skip layers)"""
        query = """
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?srcLayer ?tgtLayer ?skipDistance 
               (COUNT(?edge) as ?frequency) 
               (AVG(?score) as ?avgScore) WHERE {
            ?edge a mechint:CircuitEdge .
            ?edge mechint:hasSource ?srcNode .
            ?edge mechint:hasTarget ?tgtNode .
            ?edge mechint:inCircuit true .
            ?edge mechint:score ?score .

            ?srcNode mechint:aboutComponent ?srcComp .
            ?tgtNode mechint:aboutComponent ?tgtComp .

            ?srcComp mechint:layer ?srcLayer .
            ?tgtComp mechint:layer ?tgtLayer .

            BIND((?tgtLayer - ?srcLayer) as ?skipDistance)
            FILTER(?skipDistance > 1)
        }
        GROUP BY ?srcLayer ?tgtLayer ?skipDistance
        ORDER BY DESC(?frequency)
        """

        return self.engine._execute_select_query(query)

    def find_qkv_patterns(self) -> Dict[str, List]:
        """Analyze Q/K/V edge type patterns"""
        query = """
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?edgeType ?srcLayer ?tgtLayer 
               (COUNT(?edge) as ?frequency) 
               (AVG(?score) as ?avgScore) WHERE {
            ?edge a mechint:CircuitEdge .
            ?edge mechint:hasSource ?srcNode .
            ?edge mechint:hasTarget ?tgtNode .
            ?edge mechint:inCircuit true .
            ?edge mechint:score ?score .
            ?edge mechint:hasEdgeType ?edgeType .

            ?srcNode mechint:aboutComponent ?srcComp .
            ?tgtNode mechint:aboutComponent ?tgtComp .

            OPTIONAL { ?srcComp mechint:layer ?srcLayer }
            OPTIONAL { ?tgtComp mechint:layer ?tgtLayer }
        }
        GROUP BY ?edgeType ?srcLayer ?tgtLayer
        ORDER BY DESC(?frequency)
        """

        results = self.engine._execute_select_query(query)

        # Group by edge type
        by_type = defaultdict(list)
        for r in results:
            edge_type = r.get('edgeType', 'Unknown')
            by_type[edge_type].append(r)

        return dict(by_type)

    def find_circuit_motifs(self, min_support: int = 2) -> List[Dict]:
        """Find common subgraph motifs (3-node patterns)"""
        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?pattern (COUNT(*) as ?support) WHERE {{
            ?e1 a mechint:CircuitEdge .
            ?e1 mechint:hasSource ?n1 .
            ?e1 mechint:hasTarget ?n2 .
            ?e1 mechint:inCircuit true .

            ?e2 a mechint:CircuitEdge .
            ?e2 mechint:hasSource ?n2 .
            ?e2 mechint:hasTarget ?n3 .
            ?e2 mechint:inCircuit true .

            ?n1 mechint:aboutComponent ?c1 .
            ?n2 mechint:aboutComponent ?c2 .
            ?n3 mechint:aboutComponent ?c3 .

            BIND(CONCAT(
                STRAFTER(STR(?c1), "#"), " -> ",
                STRAFTER(STR(?c2), "#"), " -> ",
                STRAFTER(STR(?c3), "#")
            ) as ?pattern)
        }}
        GROUP BY ?pattern
        HAVING (COUNT(*) >= {min_support})
        ORDER BY DESC(?support)
        """

        return self.engine._execute_select_query(query)

    def analyze_layer_wise_patterns(self) -> List[Dict]:
        """Analyze patterns within specific layers"""
        query = """
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?layer 
               (COUNT(DISTINCT ?node) as ?numNodes)
               (COUNT(DISTINCT ?edge) as ?numEdges)
               (AVG(?score) as ?avgScore)
        WHERE {
            ?edge a mechint:CircuitEdge .
            ?edge mechint:hasTarget ?node .
            ?edge mechint:inCircuit true .
            ?edge mechint:score ?score .

            ?node mechint:aboutComponent ?comp .
            ?comp mechint:layer ?layer .
        }
        GROUP BY ?layer
        ORDER BY ?layer
        """

        results = self.engine._execute_select_query(query)

        # Compute layer importance score
        if not results:
            return []

        max_nodes = max([r.get('numNodes', 1) for r in results])
        max_edges = max([r.get('numEdges', 1) for r in results])

        for r in results:
            node_norm = r.get('numNodes', 0) / max_nodes
            edge_norm = r.get('numEdges', 0) / max_edges
            avg_score = r.get('avgScore', 0) if r.get('avgScore') else 0

            r['importance_score'] = (node_norm * 0.3 + edge_norm * 0.3 + avg_score * 0.4)

        return results

    def find_hub_components(self, min_degree: int = 5) -> List[Dict]:
        """Find components that serve as hubs (high connectivity)"""
        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?component ?layer ?head 
               (COUNT(?edge) as ?degree)
               (AVG(?score) as ?avgScore)
        WHERE {{
            ?edge a mechint:CircuitEdge .
            ?edge mechint:inCircuit true .
            ?edge mechint:score ?score .

            {{
                ?edge mechint:hasTarget ?node .
            }} UNION {{
                ?edge mechint:hasSource ?node .
            }}

            ?node mechint:aboutComponent ?component .
            OPTIONAL {{ ?component mechint:layer ?layer }}
            OPTIONAL {{ ?component mechint:head ?head }}
        }}
        GROUP BY ?component ?layer ?head
        HAVING (COUNT(?edge) >= {min_degree})
        ORDER BY DESC(?degree)
        """

        return self.engine._execute_select_query(query)


class CircuitCompositionAnalyzer:
    """Analyze circuit composition and compute aggregate statistics"""

    def __init__(self, query_engine):
        self.engine = query_engine

    def compute_circuit_complexity_metrics(self) -> List[Dict]:
        """Compute complexity metrics for each circuit"""
        query = """
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?run ?experiment ?step 
               ?numEdges ?numNodes
               ?faithfulness
               (?numEdges / ?numNodes as ?edgeDensity)
        WHERE {
            ?run a mechint:Run .
            ?run mechint:ofExperiment ?experiment .
            ?run mechint:step ?step .
            ?run mechint:numEdgesInCircuit ?numEdges .
            ?run mechint:numNodesInCircuit ?numNodes .
            ?run mechint:faithfulness ?faithfulness .
            FILTER(?numNodes > 0)
        }
        ORDER BY ?experiment ?step
        """

        results = self.engine._execute_select_query(query)

        # Add complexity score (weighted combination)
        for r in results:
            if r.get('numEdges') and r.get('numNodes') and r.get('faithfulness'):
                edge_density = r.get('edgeDensity', 0)
                faithfulness = r.get('faithfulness', 0.01)
                faithfulness_inv = 1.0 / max(faithfulness, 0.01)
                size_norm = (r['numEdges'] + r['numNodes']) / 100.0

                r['complexity_score'] = (
                        edge_density * 0.4 +
                        faithfulness_inv * 0.3 +
                        size_norm * 0.3
                )

        return results

    def analyze_circuit_efficiency(self) -> List[Dict]:
        """Compute efficiency: faithfulness per edge"""
        query = """
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?run ?experiment ?step 
               ?faithfulness ?numEdges
               (?faithfulness / ?numEdges as ?efficiency)
        WHERE {
            ?run a mechint:Run .
            ?run mechint:ofExperiment ?experiment .
            ?run mechint:step ?step .
            ?run mechint:faithfulness ?faithfulness .
            ?run mechint:numEdgesInCircuit ?numEdges .
            FILTER(?numEdges > 0)
        }
        ORDER BY DESC(?efficiency)
        """

        return self.engine._execute_select_query(query)

    def find_minimal_circuits(self, min_faithfulness: float = 0.8) -> List[Dict]:
        """Find circuits with high faithfulness but minimal size"""
        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?run ?experiment ?step 
               ?faithfulness ?numEdges ?numNodes
        WHERE {{
            ?run a mechint:Run .
            ?run mechint:ofExperiment ?experiment .
            ?run mechint:step ?step .
            ?run mechint:faithfulness ?faithfulness .
            ?run mechint:numEdgesInCircuit ?numEdges .
            ?run mechint:numNodesInCircuit ?numNodes .
            FILTER(?faithfulness >= {min_faithfulness})
        }}
        ORDER BY ?numEdges ?numNodes
        LIMIT 10
        """

        return self.engine._execute_select_query(query)

    def compare_early_vs_late_circuits(self, mid_step: int = 500) -> Dict[str, Any]:
        """Compare early training vs late training circuits"""
        early_query = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT (AVG(?faithfulness) as ?avgFaith)
               (AVG(?numEdges) as ?avgEdges)
               (AVG(?numNodes) as ?avgNodes)
               (COUNT(?run) as ?count)
        WHERE {{
            ?run a mechint:Run .
            ?run mechint:step ?step .
            ?run mechint:faithfulness ?faithfulness .
            ?run mechint:numEdgesInCircuit ?numEdges .
            ?run mechint:numNodesInCircuit ?numNodes .
            FILTER(?step < {mid_step})
        }}
        """

        late_query = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT (AVG(?faithfulness) as ?avgFaith)
               (AVG(?numEdges) as ?avgEdges)
               (AVG(?numNodes) as ?avgNodes)
               (COUNT(?run) as ?count)
        WHERE {{
            ?run a mechint:Run .
            ?run mechint:step ?step .
            ?run mechint:faithfulness ?faithfulness .
            ?run mechint:numEdgesInCircuit ?numEdges .
            ?run mechint:numNodesInCircuit ?numNodes .
            FILTER(?step >= {mid_step})
        }}
        """

        early = self.engine._execute_select_query(early_query)
        late = self.engine._execute_select_query(late_query)

        early_data = early[0] if early else {}
        late_data = late[0] if late else {}

        return {
            'early': early_data,
            'late': late_data,
            'comparison': {
                'faith_improvement': (
                        late_data.get('avgFaith', 0) - early_data.get('avgFaith', 0)
                ),
                'size_change_edges': (
                        late_data.get('avgEdges', 0) - early_data.get('avgEdges', 0)
                ),
                'size_change_nodes': (
                        late_data.get('avgNodes', 0) - early_data.get('avgNodes', 0)
                ),
            }
        }

    def find_growth_patterns(self) -> Dict[str, Any]:
        """Analyze how circuits grow over training"""
        query = """
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?experiment ?step ?numEdges ?numNodes ?faithfulness
        WHERE {
            ?run a mechint:Run .
            ?run mechint:ofExperiment ?experiment .
            ?run mechint:step ?step .
            ?run mechint:numEdgesInCircuit ?numEdges .
            ?run mechint:numNodesInCircuit ?numNodes .
            ?run mechint:faithfulness ?faithfulness .
        }
        ORDER BY ?experiment ?step
        """

        results = self.engine._execute_select_query(query)

        # Group by experiment and compute growth rates
        by_experiment = defaultdict(list)
        for r in results:
            exp = r.get('experiment', 'unknown')
            by_experiment[exp].append(r)

        growth_patterns = {}
        for exp, steps in by_experiment.items():
            if len(steps) < 2:
                continue

            # Compute growth rate
            first = steps[0]
            last = steps[-1]

            edge_growth = (last.get('numEdges', 0) - first.get('numEdges', 0)) / len(steps)
            node_growth = (last.get('numNodes', 0) - first.get('numNodes', 0)) / len(steps)
            faith_growth = (last.get('faithfulness', 0) - first.get('faithfulness', 0)) / len(steps)

            growth_patterns[exp] = {
                'edge_growth_rate': edge_growth,
                'node_growth_rate': node_growth,
                'faithfulness_growth_rate': faith_growth,
                'steps_tracked': len(steps)
            }

        return growth_patterns


class SemanticRuleAnalyzer:
    """Analyze semantic rules and role allocation in circuits"""

    def __init__(self, query_engine):
        self.engine = query_engine

    def compare_role_circuits(self, role_a: str, role_b: str) -> Dict[str, Any]:
        """Compare circuits for two different semantic roles"""
        # Make roles lowercase for matching
        role_a_lower = role_a.lower()
        role_b_lower = role_b.lower()

        query_a = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?sourceAddr ?targetAddr WHERE {{
            ?circuit a mechint:Circuit .
            ?circuit mechint:discoveredIn ?exp .
            FILTER(CONTAINS(LCASE(STR(?exp)), "{role_a_lower}"))
            
            ?circuit mechint:hasMemberEdge ?edge .
            ?edge mechint:hasSource ?srcNode .
            ?edge mechint:hasTarget ?tgtNode .
            ?srcNode mechint:address ?sourceAddr .
            ?tgtNode mechint:address ?targetAddr .
        }}
        """

        query_b = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?sourceAddr ?targetAddr WHERE {{
            ?circuit a mechint:Circuit .
            ?circuit mechint:discoveredIn ?exp .
            FILTER(CONTAINS(LCASE(STR(?exp)), "{role_b_lower}"))
            
            ?circuit mechint:hasMemberEdge ?edge .
            ?edge mechint:hasSource ?srcNode .
            ?edge mechint:hasTarget ?tgtNode .
            ?srcNode mechint:address ?sourceAddr .
            ?tgtNode mechint:address ?targetAddr .
        }}
        """

        edges_a = set((r['sourceAddr'], r['targetAddr'])
                      for r in self.engine._execute_select_query(query_a)
                      if r.get('sourceAddr') and r.get('targetAddr'))
        edges_b = set((r['sourceAddr'], r['targetAddr'])
                      for r in self.engine._execute_select_query(query_b)
                      if r.get('sourceAddr') and r.get('targetAddr'))

        shared = edges_a & edges_b
        unique_a = edges_a - edges_b
        unique_b = edges_b - edges_a

        jaccard = len(shared) / len(edges_a | edges_b) if (edges_a | edges_b) else 0

        return {
            'role_a': role_a,
            'role_b': role_b,
            'shared_edges': len(shared),
            'unique_to_a': len(unique_a),
            'unique_to_b': len(unique_b),
            'jaccard_similarity': jaccard,
            'total_edges_a': len(edges_a),
            'total_edges_b': len(edges_b),
            'shared_edge_list': list(shared)[:10],  # Sample
            'unique_a_list': list(unique_a)[:10],
            'unique_b_list': list(unique_b)[:10],
        }

    def find_role_specific_components(self, role: str) -> List[Dict]:
        """Find components that appear only in circuits for a specific role"""
        # Make role lowercase for matching
        role_lower = role.lower()

        query = f"""
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?nodeName ?layer ?head (COUNT(DISTINCT ?circuit) as ?frequency) WHERE {{
            ?circuit a mechint:Circuit .
            ?circuit mechint:discoveredIn ?exp .
            FILTER(CONTAINS(LCASE(STR(?exp)), "{role_lower}"))
            
            ?circuit mechint:hasComponent ?node .
            ?node mechint:address ?nodeName .
            OPTIONAL {{ ?node mechint:inLayer ?layer }}
            OPTIONAL {{ ?node mechint:head ?head }}
        }}
        GROUP BY ?nodeName ?layer ?head
        ORDER BY DESC(?frequency)
        """

        results = self.engine._execute_select_query(query)

        # Add component name for better readability
        for r in results:
            if not r.get('component'):
                r['component'] = r.get('nodeName', 'unknown')

        return results

    def analyze_role_layer_distribution(self, roles: List[str]) -> Dict[str, Any]:
        """Analyze which layers are used by different roles"""
        layer_usage = {}

        for role in roles:
            role_lower = role.lower()
            query = f"""
            PREFIX mechint: <http://purl.org/mechint#>
            SELECT ?layer (COUNT(DISTINCT ?node) as ?count) WHERE {{
                ?circuit a mechint:Circuit .
                ?circuit mechint:discoveredIn ?exp .
                FILTER(CONTAINS(LCASE(STR(?exp)), "{role_lower}"))
                
                ?circuit mechint:hasComponent ?node .
                ?node mechint:inLayer ?layer .
            }}
            GROUP BY ?layer
            ORDER BY ?layer
            """

            layer_usage[role] = self.engine._execute_select_query(query)

        return layer_usage


class ILPReasoner:
    """Inductive Logic Programming for circuit pattern learning"""

    def __init__(self, query_engine):
        self.engine = query_engine
        self.learned_rules = []

    def learn_composition_rules(self) -> List[Dict[str, Any]]:
        """Learn rules about how circuits compose"""
        query = """
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?run1 ?run2 ?sharedNode WHERE {
            ?run1 a mechint:Run .
            ?run2 a mechint:Run .
            FILTER(?run1 != ?run2)

            ?circuit1 mechint:inRun ?run1 .
            ?circuit2 mechint:inRun ?run2 .

            ?circuit1 mechint:hasMemberNode ?sharedNode .
            ?circuit2 mechint:hasMemberNode ?sharedNode .
        }
        """

        results = self.engine._execute_select_query(query)

        # Group by run pairs
        pairs = defaultdict(set)
        for r in results:
            run1 = r.get('run1')
            run2 = r.get('run2')
            if run1 and run2:
                pair = tuple(sorted([run1, run2]))
                pairs[pair].add(r.get('sharedNode'))

        composition_rules = []
        for (run1, run2), shared_nodes in pairs.items():
            if len(shared_nodes) >= 2:  # Significant overlap
                composition_rules.append({
                    'rule_type': 'composition',
                    'circuit_1': run1,
                    'circuit_2': run2,
                    'shared_components': list(shared_nodes)[:10],  # Limit for JSON
                    'overlap_count': len(shared_nodes),
                    'confidence': min(len(shared_nodes) / 10.0, 1.0)
                })

        self.learned_rules.extend(composition_rules)
        return composition_rules

    def infer_component_specialization(self) -> List[Dict]:
        """Infer which components specialize in which tasks"""
        query = """
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?component ?layer ?head 
               (COUNT(DISTINCT ?run) as ?appearances)
               (AVG(?faithfulness) as ?avgFaithfulness)
        WHERE {
            ?edge a mechint:CircuitEdge .
            ?edge mechint:hasTarget ?node .
            ?edge mechint:inCircuit true .
            ?edge mechint:inRun ?run .

            ?run mechint:faithfulness ?faithfulness .
            ?node mechint:aboutComponent ?component .
            OPTIONAL { ?component mechint:layer ?layer }
            OPTIONAL { ?component mechint:head ?head }
        }
        GROUP BY ?component ?layer ?head
        HAVING (COUNT(DISTINCT ?run) > 3)
        ORDER BY DESC(?appearances)
        """

        results = self.engine._execute_select_query(query)

        specializations = []
        for r in results:
            avg_faith = r.get('avgFaithfulness', 0)
            if avg_faith and avg_faith > 0.7:
                appearances = r.get('appearances', 0)
                specializations.append({
                    'component': r.get('component'),
                    'layer': r.get('layer'),
                    'head': r.get('head'),
                    'specialization_confidence': min(appearances / 10.0, 1.0),
                    'avg_faithfulness': avg_faith,
                    'appearances': appearances,
                    'rule': f"Component {r.get('component')} is specialized (appears in {appearances} high-quality circuits)"
                })

        self.learned_rules.extend(specializations)
        return specializations

    def learn_prerequisite_patterns(self) -> List[Dict]:
        """Learn which components are prerequisites for others"""
        query = """
        PREFIX mechint: <http://purl.org/mechint#>
        SELECT ?comp1 ?comp2 (COUNT(*) as ?cooccurrence) WHERE {
            ?edge1 mechint:inCircuit true .
            ?edge1 mechint:hasTarget ?node1 .
            ?node1 mechint:aboutComponent ?comp1 .
            ?edge1 mechint:inRun ?run .

            ?edge2 mechint:inCircuit true .
            ?edge2 mechint:hasTarget ?node2 .
            ?node2 mechint:aboutComponent ?comp2 .
            ?edge2 mechint:inRun ?run .

            FILTER(?comp1 != ?comp2)
        }
        GROUP BY ?comp1 ?comp2
        HAVING (COUNT(*) > 5)
        ORDER BY DESC(?cooccurrence)
        """

        results = self.engine._execute_select_query(query)

        prerequisites = []
        for r in results:
            cooccurrence = r.get('cooccurrence', 0)
            if cooccurrence > 5:
                prerequisites.append({
                    'prerequisite': r.get('comp1'),
                    'dependent': r.get('comp2'),
                    'cooccurrence_count': cooccurrence,
                    'confidence': min(cooccurrence / 20.0, 1.0),
                    'rule': f"{r.get('comp1')} often appears with {r.get('comp2')}"
                })

        self.learned_rules.extend(prerequisites)
        return prerequisites

    def export_learned_rules(self, output_path: str):
        """Export learned rules to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                'total_rules': len(self.learned_rules),
                'rules': self.learned_rules
            }, f, indent=2, default=str)

        print(f"Exported {len(self.learned_rules)} learned rules to {output_path}")


def run_comprehensive_analysis(data_dir: str = "./data/experiments",
                               output_dir: str = "./analysis_results") -> Dict[str, Any]:
    """Run all analysis pipelines and save results"""

    # Import here to avoid circular dependencies
    try:
        from ..query.sparql_interface import CircuitQueryEngine
    except ImportError:
        # Fallback for standalone usage
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from ..query.sparql_interface import CircuitQueryEngine

    # Initialize
    engine = CircuitQueryEngine(data_dir)
    if not engine.load_experiments():
        print("ERROR: No experiments loaded. Cannot perform analysis.")
        return {}

    pattern_miner = CircuitPatternMiner(engine)
    composition_analyzer = CircuitCompositionAnalyzer(engine)
    semantic_analyzer = SemanticRuleAnalyzer(engine)
    ilp_reasoner = ILPReasoner(engine)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("COMPREHENSIVE CIRCUIT ANALYSIS")
    print("=" * 60)

    # 1. Pattern mining
    print("\n[1/4] Mining circuit patterns...")
    try:
        attention_patterns = pattern_miner.find_attention_head_patterns()
        mlp_patterns = pattern_miner.find_mlp_patterns()
        skip_connections = pattern_miner.find_skip_connections()
        qkv_patterns = pattern_miner.find_qkv_patterns()
        motifs = pattern_miner.find_circuit_motifs(min_support=2)
        layer_patterns = pattern_miner.analyze_layer_wise_patterns()
        hub_components = pattern_miner.find_hub_components(min_degree=5)
        print(f"  ✓ Found {len(motifs)} motifs, {len(hub_components)} hubs")
    except Exception as e:
        print(f"  ✗ Pattern mining failed: {e}")
        attention_patterns = mlp_patterns = skip_connections = {}
        qkv_patterns = motifs = layer_patterns = hub_components = []

    # 2. Composition analysis
    print("[2/4] Analyzing circuit composition...")
    try:
        complexity_metrics = composition_analyzer.compute_circuit_complexity_metrics()
        efficiency = composition_analyzer.analyze_circuit_efficiency()
        minimal_circuits = composition_analyzer.find_minimal_circuits()
        early_late_comparison = composition_analyzer.compare_early_vs_late_circuits()
        growth_patterns = composition_analyzer.find_growth_patterns()
        print(f"  ✓ Analyzed {len(complexity_metrics)} circuits")
    except Exception as e:
        print(f"  ✗ Composition analysis failed: {e}")
        complexity_metrics = efficiency = minimal_circuits = []
        early_late_comparison = growth_patterns = {}

    # 3. ILP reasoning
    print("[3/4] Learning compositional rules...")
    try:
        composition_rules = ilp_reasoner.learn_composition_rules()
        specializations = ilp_reasoner.infer_component_specialization()
        prerequisites = ilp_reasoner.learn_prerequisite_patterns()
        print(f"  ✓ Learned {len(composition_rules)} composition rules")
        print(f"  ✓ Identified {len(specializations)} specialized components")
        print(f"  ✓ Found {len(prerequisites)} prerequisite patterns")
    except Exception as e:
        print(f"  ✗ ILP reasoning failed: {e}")
        composition_rules = specializations = prerequisites = []

    # 4. Database statistics
    print("[4/4] Computing database statistics...")
    try:
        db_stats = engine.get_database_stats()
        experiment_summary = engine.get_experiment_summary()
        print(f"  ✓ Total triples: {db_stats.get('total_triples', 0)}")
        print(f"  ✓ Experiments: {db_stats.get('loaded_experiments', 0)}")
    except Exception as e:
        print(f"  ✗ Statistics computation failed: {e}")
        db_stats = experiment_summary = {}

    # Compile results
    results = {
        'metadata': {
            'data_dir': str(data_dir),
            'output_dir': str(output_dir),
            'timestamp': str(Path(__file__).stat().st_mtime) if Path(__file__).exists() else None
        },
        'database_stats': db_stats,
        'experiment_summary': experiment_summary,
        'patterns': {
            'attention_heads': attention_patterns,
            'mlp_layers': mlp_patterns,
            'skip_connections': skip_connections[:20] if skip_connections else [],
            'qkv_patterns': {k: v[:10] for k, v in qkv_patterns.items()} if isinstance(qkv_patterns, dict) else {},
            'motifs': motifs[:20] if motifs else [],
            'layer_patterns': layer_patterns,
            'hub_components': hub_components[:20] if hub_components else [],
        },
        'composition': {
            'complexity_metrics': complexity_metrics[:10] if complexity_metrics else [],
            'efficiency': efficiency[:10] if efficiency else [],
            'minimal_circuits': minimal_circuits,
            'early_vs_late': early_late_comparison,
            'growth_patterns': growth_patterns,
        },
        'learned_rules': {
            'composition_rules': composition_rules[:10] if composition_rules else [],
            'specializations': specializations[:20] if specializations else [],
            'prerequisites': prerequisites[:20] if prerequisites else [],
        }
    }

    # Export main results
    try:
        output_file = output_path / "comprehensive_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Analysis complete. Results saved to {output_file}")
    except Exception as e:
        print(f"\n✗ Failed to save results: {e}")

    # Export ILP rules separately
    try:
        ilp_reasoner.export_learned_rules(str(output_path / "learned_rules.json"))
    except Exception as e:
        print(f"✗ Failed to export ILP rules: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Motifs discovered: {len(motifs) if motifs else 0}")
    print(f"Composition rules: {len(composition_rules) if composition_rules else 0}")
    print(f"Specialized components: {len(specializations) if specializations else 0}")
    print(f"Hub components: {len(hub_components) if hub_components else 0}")
    print(f"Prerequisite patterns: {len(prerequisites) if prerequisites else 0}")
    print("=" * 60)

    return results


def quick_analysis(data_dir: str = "./data/experiments") -> Dict[str, Any]:
    """Quick analysis for debugging - just basic stats"""
    try:
        from ..query.sparql_interface import CircuitQueryEngine
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from ..query.sparql_interface import CircuitQueryEngine

    engine = CircuitQueryEngine(data_dir)
    if not engine.load_experiments():
        return {'error': 'No experiments loaded'}

    stats = engine.get_database_stats()
    summary = engine.get_experiment_summary()
    high_faith = engine.find_high_faithfulness_circuits(min_faithfulness=0.8)

    return {
        'database_stats': stats,
        'experiment_summary': summary,
        'high_faithfulness_circuits': high_faith[:5]
    }


# Convenience function for exporting to various formats
def export_analysis_results(results: Dict[str, Any],
                            output_dir: str,
                            formats: List[str] = ['json', 'summary']) -> List[str]:
    """
    Export analysis results to multiple formats

    Args:
        results: Analysis results dictionary
        output_dir: Output directory
        formats: List of formats ['json', 'summary', 'csv']

    Returns:
        List of created file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    created_files = []

    # JSON export
    if 'json' in formats:
        json_path = output_path / "analysis_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        created_files.append(str(json_path))

    # Text summary
    if 'summary' in formats:
        summary_path = output_path / "analysis_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MECHANISTIC INTERPRETABILITY ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            # Database stats
            if 'database_stats' in results:
                f.write("DATABASE STATISTICS\n")
                f.write("-" * 40 + "\n")
                for key, value in results['database_stats'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            # Pattern summary
            if 'patterns' in results:
                f.write("DISCOVERED PATTERNS\n")
                f.write("-" * 40 + "\n")
                patterns = results['patterns']
                if isinstance(patterns.get('attention_heads'), dict):
                    f.write(
                        f"  Core attention heads: {patterns['attention_heads'].get('stats', {}).get('core_head_count', 0)}\n")
                f.write(f"  Circuit motifs: {len(patterns.get('motifs', []))}\n")
                f.write(f"  Hub components: {len(patterns.get('hub_components', []))}\n")
                f.write("\n")

            # Learned rules summary
            if 'learned_rules' in results:
                f.write("LEARNED RULES\n")
                f.write("-" * 40 + "\n")
                rules = results['learned_rules']
                f.write(f"  Composition rules: {len(rules.get('composition_rules', []))}\n")
                f.write(f"  Specializations: {len(rules.get('specializations', []))}\n")
                f.write(f"  Prerequisites: {len(rules.get('prerequisites', []))}\n")

        created_files.append(str(summary_path))

    # CSV export for tabular data
    if 'csv' in formats:
        try:
            import pandas as pd

            # Export efficiency data
            if 'composition' in results and 'efficiency' in results['composition']:
                csv_path = output_path / "circuit_efficiency.csv"
                df = pd.DataFrame(results['composition']['efficiency'])
                df.to_csv(csv_path, index=False)
                created_files.append(str(csv_path))
        except ImportError:
            print("Warning: pandas not available, skipping CSV export")

    print(f"Exported analysis results to {len(created_files)} files")
    return created_files
