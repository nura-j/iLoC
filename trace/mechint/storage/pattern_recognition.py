"""
Pattern Recognition from OWL

Extracts patterns from existing OWL circuits without requiring ILP.
Uses structural analysis, statistical patterns, and heuristic rules.

This is a lightweight alternative to full ILP that: (I'm goning to change this)
- Analyzes existing circuit data
- Finds recurring structural patterns
- Identifies component usage patterns
- Detects architectural motifs
- Groups similar circuits
"""

from typing import List, Dict, Any
from rdflib import Graph, Namespace
from collections import defaultdict, Counter
import numpy as np


class PatternRecognizer:
    """
    Extract patterns from OWL circuits without ILP.

    Uses:
    - Structural analysis (components, edges, depth)
    - Statistical patterns (frequencies, correlations)
    - Heuristic rules (if-then patterns)
    - Clustering (group similar circuits)
    """

    def __init__(self, namespace_uri: str = "http://purl.org/mechint#"):
        self.MECHINT = Namespace(namespace_uri)

    def load_graph(self, ttl_path: str) -> Graph:
        """Load RDF graph from Turtle file"""
        g = Graph()
        g.parse(ttl_path, format='turtle')
        return g

    # ========================================
    # STRUCTURAL PATTERNS
    # ========================================

    def find_component_patterns(self, graph: Graph) -> Dict[str, Any]:
        """
        Find common component usage patterns.

        Identifies:
        - Which components appear together frequently
        - Component type distributions (attn vs MLP)
        - Depth distributions

        Returns:
            Dict with component patterns
        """
        # Get all circuits
        query = f"""
        PREFIX mechint: <{self.MECHINT}>
        SELECT ?circuit ?component WHERE {{
            ?circuit a mechint:Circuit .
            ?circuit mechint:hasComponent ?component .
        }}
        """

        results = graph.query(query)

        # Group components by circuit
        circuit_components = defaultdict(list)
        for row in results:
            circuit_components[str(row.circuit)].append(str(row.component))

        # Find frequent component sets
        component_counter = Counter()
        for circuit, components in circuit_components.items():
            # Count individual components
            for comp in components:
                component_counter[comp] += 1

            # Count pairs
            for i, comp1 in enumerate(components):
                for comp2 in components[i + 1:]:
                    pair = tuple(sorted([comp1, comp2]))
                    component_counter[pair] += 1

        # Extract patterns
        frequent_components = [
            (comp, count) for comp, count in component_counter.most_common(20)
            if not isinstance(comp, tuple)
        ]

        frequent_pairs = [
            (comp, count) for comp, count in component_counter.most_common(20)
            if isinstance(comp, tuple)
        ]

        return {
            'total_circuits': len(circuit_components),
            'frequent_components': frequent_components,
            'frequent_pairs': frequent_pairs,
            'avg_components_per_circuit': np.mean([len(c) for c in circuit_components.values()]),
        }

    def find_edge_patterns(self, graph: Graph) -> Dict[str, Any]:
        """
        Find common edge patterns.

        Identifies:
        - Frequently connected component pairs
        - Edge type distributions
        - Path patterns (A→B→C)

        Returns:
            Dict with edge patterns
        """
        query = f"""
        PREFIX mechint: <{self.MECHINT}>
        SELECT ?circuit ?edge ?source ?target WHERE {{
            ?circuit mechint:hasMemberEdge ?edge .
            ?edge mechint:hasSource ?source .
            ?edge mechint:hasTarget ?target .
        }}
        """

        results = graph.query(query)

        # Count connection patterns
        connection_counter = Counter()
        circuit_edges = defaultdict(list)

        for row in results:
            source_name = self._extract_component_name(str(row.source))
            target_name = self._extract_component_name(str(row.target))

            connection = (source_name, target_name)
            connection_counter[connection] += 1
            circuit_edges[str(row.circuit)].append(connection)

        # Find frequent connections
        frequent_connections = connection_counter.most_common(20)

        # Find paths (A→B→C)
        path_counter = Counter()
        for circuit, edges in circuit_edges.items():
            # Build adjacency
            adjacency = defaultdict(list)
            for src, tgt in edges:
                adjacency[src].append(tgt)

            # Find 2-hop paths
            for src, targets in adjacency.items():
                for mid in targets:
                    if mid in adjacency:
                        for end in adjacency[mid]:
                            path = (src, mid, end)
                            path_counter[path] += 1

        frequent_paths = path_counter.most_common(10)

        return {
            'frequent_connections': frequent_connections,
            'frequent_paths': frequent_paths,
            'total_edges': sum(connection_counter.values()),
        }

    def find_depth_patterns(self, graph: Graph) -> Dict[str, Any]:
        """
        Find depth distribution patterns.

        Identifies:
        - Common depth ranges
        - Depth clustering
        - Layer preferences

        Returns:
            Dict with depth patterns
        """
        query = f"""
        PREFIX mechint: <{self.MECHINT}>
        SELECT ?circuit ?depthMin ?depthMax ?depthMean WHERE {{
            ?circuit a mechint:Circuit .
            OPTIONAL {{ ?circuit mechint:depthMin ?depthMin }}
            OPTIONAL {{ ?circuit mechint:depthMax ?depthMax }}
            OPTIONAL {{ ?circuit mechint:depthMean ?depthMean }}
        }}
        """

        results = graph.query(query)

        depths = []
        for row in results:
            if row.depthMean:
                depths.append(float(row.depthMean))

        if not depths:
            return {'error': 'No depth data found'}

        # Cluster depths
        depth_ranges = self._cluster_values(depths, n_bins=5)

        return {
            'depth_distribution': {
                'min': min(depths),
                'max': max(depths),
                'mean': np.mean(depths),
                'std': np.std(depths),
            },
            'depth_ranges': depth_ranges,
        }

    # ========================================
    # ARCHITECTURAL PATTERNS
    # ========================================

    def find_architectural_patterns(self, graph: Graph) -> Dict[str, Any]:
        """
        Find architectural patterns across circuits.

        Identifies:
        - Common component ratios (attn:mlp)
        - Motif distributions
        - Size patterns

        Returns:
            Dict with architectural patterns
        """
        query = f"""
        PREFIX mechint: <{self.MECHINT}>
        SELECT ?circuit ?ratio ?numAttn ?numMlp ?motif WHERE {{
            ?circuit a mechint:Circuit .
            OPTIONAL {{ ?circuit mechint:componentRatio ?ratio }}
            OPTIONAL {{ ?circuit mechint:numAttentionHeads ?numAttn }}
            OPTIONAL {{ ?circuit mechint:numMLPBlocks ?numMlp }}
            OPTIONAL {{ ?circuit mechint:hasMotif ?motif }}
        }}
        """

        results = graph.query(query)

        ratio_counter = Counter()
        motif_counter = Counter()
        sizes = []

        for row in results:
            if row.ratio:
                ratio_counter[str(row.ratio)] += 1
            if row.motif:
                motif_name = str(row.motif).split('#')[-1]
                motif_counter[motif_name] += 1
            if row.numAttn and row.numMlp:
                total = int(row.numAttn) + int(row.numMlp)
                sizes.append(total)

        return {
            'common_ratios': ratio_counter.most_common(10),
            'common_motifs': motif_counter.most_common(10),
            'size_distribution': {
                'mean': np.mean(sizes) if sizes else 0,
                'std': np.std(sizes) if sizes else 0,
                'range': (min(sizes), max(sizes)) if sizes else (0, 0),
            },
        }

    # ========================================
    # PERFORMANCE PATTERNS
    # ========================================

    def find_performance_patterns(self, graph: Graph) -> Dict[str, Any]:
        """
        Find patterns in circuit performance.

        Identifies:
        - High-performing circuit characteristics
        - Correlation between features and performance
        - Quality clusters

        Returns:
            Dict with performance patterns
        """
        query = f"""
        PREFIX mechint: <{self.MECHINT}>
        SELECT ?circuit ?faithful ?causal ?ratio ?depthMean WHERE {{
            ?circuit a mechint:Circuit .
            OPTIONAL {{ ?circuit mechint:faithfulness ?faithful }}
            OPTIONAL {{ ?circuit mechint:causalEffect ?causal }}
            OPTIONAL {{ ?circuit mechint:componentRatio ?ratio }}
            OPTIONAL {{ ?circuit mechint:depthMean ?depthMean }}
        }}
        """

        results = graph.query(query)

        # Group by performance level
        high_performing = []
        medium_performing = []
        low_performing = []

        for row in results:
            if not row.faithful:
                continue

            faithful = float(row.faithful)
            circuit_data = {
                'circuit': str(row.circuit),
                'faithfulness': faithful,
                'causal': float(row.causal) if row.causal else 0,
                'ratio': str(row.ratio) if row.ratio else 'unknown',
                'depth': float(row.depthMean) if row.depthMean else 0,
            }

            if faithful >= 0.8:
                high_performing.append(circuit_data)
            elif faithful >= 0.5:
                medium_performing.append(circuit_data)
            else:
                low_performing.append(circuit_data)

        # Extract patterns from high performers
        high_perf_ratios = Counter(c['ratio'] for c in high_performing)
        high_perf_depths = [c['depth'] for c in high_performing if c['depth'] > 0]

        return {
            'performance_tiers': {
                'high': len(high_performing),
                'medium': len(medium_performing),
                'low': len(low_performing),
            },
            'high_performer_characteristics': {
                'common_ratios': high_perf_ratios.most_common(5),
                'avg_depth': np.mean(high_perf_depths) if high_perf_depths else 0,
                'avg_causal_effect': np.mean([c['causal'] for c in high_performing]),
            },
        }

    # ========================================
    # TASK-SPECIFIC PATTERNS
    # ========================================

    def find_task_patterns(self, graph: Graph) -> Dict[str, Any]:
        """
        Find patterns specific to task families.

        Identifies:
        - Which architectural patterns work for which tasks
        - Task-specific component preferences
        - Cross-task commonalities

        Returns:
            Dict with task-specific patterns
        """
        query = f"""
        PREFIX mechint: <{self.MECHINT}>
        SELECT ?circuit ?taskFamily ?role ?ratio ?depthMean ?faithful WHERE {{
            ?circuit a mechint:Circuit .
            OPTIONAL {{ ?circuit mechint:belongsToTaskFamily ?taskFam .
                       BIND(STRAFTER(STR(?taskFam), "#") as ?taskFamily) }}
            OPTIONAL {{ ?circuit mechint:roleName ?role }}
            OPTIONAL {{ ?circuit mechint:componentRatio ?ratio }}
            OPTIONAL {{ ?circuit mechint:depthMean ?depthMean }}
            OPTIONAL {{ ?circuit mechint:faithfulness ?faithful }}
        }}
        """

        results = graph.query(query)

        # Group by task
        task_data = defaultdict(lambda: {
            'circuits': [],
            'ratios': Counter(),
            'depths': [],
            'faithfulness_scores': [],
        })

        for row in results:
            task = str(row.taskFamily) if row.taskFamily else 'unknown'

            task_data[task]['circuits'].append(str(row.circuit))
            if row.ratio:
                task_data[task]['ratios'][str(row.ratio)] += 1
            if row.depthMean:
                task_data[task]['depths'].append(float(row.depthMean))
            if row.faithful:
                task_data[task]['faithfulness_scores'].append(float(row.faithful))

        # Summarize patterns per task
        task_patterns = {}
        for task, data in task_data.items():
            task_patterns[task] = {
                'num_circuits': len(data['circuits']),
                'common_ratios': data['ratios'].most_common(3),
                'avg_depth': np.mean(data['depths']) if data['depths'] else 0,
                'avg_faithfulness': np.mean(data['faithfulness_scores']) if data['faithfulness_scores'] else 0,
            }

        return task_patterns

    # ========================================
    # SIMILARITY CLUSTERING
    # ========================================

    def find_circuit_clusters(
            self,
            graph: Graph,
            n_clusters: int = 5
    ) -> Dict[str, Any]:
        """
        Cluster circuits by similarity.

        Uses architectural features to group similar circuits.

        Returns:
            Dict with cluster assignments and characteristics
        """
        # Get circuit features
        query = f"""
        PREFIX mechint: <{self.MECHINT}>
        SELECT ?circuit ?depthMean ?numAttn ?numMlp ?faithful WHERE {{
            ?circuit a mechint:Circuit .
            OPTIONAL {{ ?circuit mechint:depthMean ?depthMean }}
            OPTIONAL {{ ?circuit mechint:numAttentionHeads ?numAttn }}
            OPTIONAL {{ ?circuit mechint:numMLPBlocks ?numMlp }}
            OPTIONAL {{ ?circuit mechint:faithfulness ?faithful }}
        }}
        """

        results = graph.query(query)

        # Build feature matrix
        circuits = []
        features = []

        for row in results:
            circuit_uri = str(row.circuit)

            # Extract features (use defaults if missing)
            depth = float(row.depthMean) if row.depthMean else 0.5
            num_attn = int(row.numAttn) if row.numAttn else 0
            num_mlp = int(row.numMlp) if row.numMlp else 0
            faithful = float(row.faithful) if row.faithful else 0.5

            circuits.append(circuit_uri)
            features.append([depth, num_attn, num_mlp, faithful])

        if not features:
            return {'error': 'No circuits with features found'}

        # Simple k-means clustering
        features_array = np.array(features)

        # Normalize features
        features_norm = (features_array - features_array.mean(axis=0)) / (features_array.std(axis=0) + 1e-8)

        # Cluster
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(n_clusters, len(circuits)), random_state=42)
        cluster_labels = kmeans.fit_predict(features_norm)

        # Group circuits by cluster
        clusters = defaultdict(list)
        for circuit, label in zip(circuits, cluster_labels):
            clusters[int(label)].append(circuit)

        # Characterize clusters
        cluster_chars = {}
        for label, cluster_circuits in clusters.items():
            indices = [i for i, c in enumerate(circuits) if c in cluster_circuits]
            cluster_features = features_array[indices]

            cluster_chars[f"cluster_{label}"] = {
                'size': len(cluster_circuits),
                'circuits': cluster_circuits[:5],  # Sample
                'avg_depth': float(cluster_features[:, 0].mean()),
                'avg_attn': float(cluster_features[:, 1].mean()),
                'avg_mlp': float(cluster_features[:, 2].mean()),
                'avg_faithfulness': float(cluster_features[:, 3].mean()),
            }

        return cluster_chars

    # ========================================
    # COMPREHENSIVE ANALYSIS
    # ========================================

    def analyze_all_patterns(self, graph: Graph) -> Dict[str, Any]:
        """
        Run all pattern recognition analyses.

        Returns:
            Complete pattern analysis report
        """
        print("Analyzing patterns from OWL data...")

        analyses = {}

        print("  1/6 Component patterns...")
        analyses['component_patterns'] = self.find_component_patterns(graph)

        print("  2/6 Edge patterns...")
        analyses['edge_patterns'] = self.find_edge_patterns(graph)

        print("  3/6 Depth patterns...")
        analyses['depth_patterns'] = self.find_depth_patterns(graph)

        print("  4/6 Architectural patterns...")
        analyses['architectural_patterns'] = self.find_architectural_patterns(graph)

        print("  5/6 Performance patterns...")
        analyses['performance_patterns'] = self.find_performance_patterns(graph)

        print("  6/6 Task patterns...")
        analyses['task_patterns'] = self.find_task_patterns(graph)

        print("\n✓ Pattern analysis complete")

        return analyses

    # ========================================
    # HELPER METHODS
    # ========================================

    def _extract_component_name(self, uri: str) -> str:
        """Extract component name from URI"""
        # Extract last part after # or /
        if '#' in uri:
            return uri.split('#')[-1]
        elif '/' in uri:
            return uri.split('/')[-1]
        return uri

    def _cluster_values(self, values: List[float], n_bins: int = 5) -> List[Dict]:
        """Cluster continuous values into bins"""
        if not values:
            return []

        min_val = min(values)
        max_val = max(values)
        bin_size = (max_val - min_val) / n_bins

        bins = []
        for i in range(n_bins):
            bin_min = min_val + i * bin_size
            bin_max = bin_min + bin_size
            count = sum(1 for v in values if bin_min <= v < bin_max)

            bins.append({
                'range': (bin_min, bin_max),
                'count': count,
                'percentage': count / len(values) * 100,
            })

        return bins

