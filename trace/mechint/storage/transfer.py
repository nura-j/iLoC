"""
Transfer Matching Module

Implements transfer candidate search and validation (§2.6)

This module provides:
1. Candidate search via architectural similarity
2. Transfer validation (causal effect preservation)
3. Transfer success metrics

"""

from typing import List, Dict, Any, Tuple
from rdflib import Graph, URIRef, Namespace
import numpy as np


class TransferMatcher:
    """
    Find and validate transfer candidates based on architectural signatures.
    
    Transfer hypothesis (Paper: 2.6): Circuits with similar architectural signatures
    should transfer (preserve causal effect Δ_T) across tasks/models.
    """
    
    def __init__(self, namespace_uri: str = "http://purl.org/mechint#"):
        """
        Initialize transfer matcher.
        
        Args:
            namespace_uri: Mechint ontology namespace
        """
        self.MECHINT = Namespace(namespace_uri)
    
    def load_graph(self, ttl_path: str) -> Graph:
        """Load RDF graph from Turtle file"""
        g = Graph()
        g.parse(ttl_path, format='turtle')
        return g
    
    def get_architectural_features(self, graph: Graph, circuit_uri: URIRef) -> Dict[str, Any]:
        """
        Extract architectural features from circuit.
        
        Args:
            graph: RDF graph
            circuit_uri: Circuit URI
            
        Returns:
            Dict with architectural features:
                - depth_min: float (ρ_min)
                - depth_max: float (ρ_max)
                - depth_mean: float (ρ̄)
                - component_ratio: str (e.g., "3:1")
                - num_attention_heads: int
                - num_mlp_blocks: int
                - motifs: List[str]
        """
        features = {}
        
        # Depth features
        features['depth_min'] = float(graph.value(circuit_uri, self.MECHINT.depthMin) or -1)
        features['depth_max'] = float(graph.value(circuit_uri, self.MECHINT.depthMax) or -1)
        features['depth_mean'] = float(graph.value(circuit_uri, self.MECHINT.depthMean) or -1)
        
        # Component counts
        features['num_attention_heads'] = int(graph.value(circuit_uri, self.MECHINT.numAttentionHeads) or 0)
        features['num_mlp_blocks'] = int(graph.value(circuit_uri, self.MECHINT.numMLPBlocks) or 0)
        
        # Component ratio
        features['component_ratio'] = str(graph.value(circuit_uri, self.MECHINT.componentRatio) or "")
        
        # Motifs
        motif_uris = list(graph.objects(circuit_uri, self.MECHINT.hasMotif))
        features['motifs'] = [str(m).split('#')[-1] for m in motif_uris]
        
        return features
    
    def compute_architectural_similarity(
        self,
        features1: Dict[str, Any],
        features2: Dict[str, Any],
        depth_weight: float = 0.4,
        ratio_weight: float = 0.3,
        motif_weight: float = 0.3
    ) -> float:
        """
        Compute architectural similarity between two circuits.
        
        Args:
            features1: Features from first circuit
            features2: Features from second circuit
            depth_weight: Weight for depth similarity
            ratio_weight: Weight for ratio similarity
            motif_weight: Weight for motif similarity
            
        Returns:
            Similarity score in [0, 1]
        """
        # Depth similarity (using mean depth)
        if features1['depth_mean'] >= 0 and features2['depth_mean'] >= 0:
            depth_diff = abs(features1['depth_mean'] - features2['depth_mean'])
            depth_sim = max(0, 1 - depth_diff)  # 1 if same, 0 if differ by 1+
        else:
            depth_sim = 0.0
        
        # Ratio similarity (exact match)
        ratio_sim = 1.0 if features1['component_ratio'] == features2['component_ratio'] else 0.0
        
        # Motif similarity (Jaccard of motif sets)
        motifs1 = set(features1['motifs'])
        motifs2 = set(features2['motifs'])
        if motifs1 or motifs2:
            motif_sim = len(motifs1 & motifs2) / len(motifs1 | motifs2)
        else:
            motif_sim = 1.0  # Both empty = similar
        
        # Weighted combination
        similarity = (
            depth_weight * depth_sim +
            ratio_weight * ratio_sim +
            motif_weight * motif_sim
        )
        
        return similarity
    
    def find_transfer_candidates(
        self,
        graph: Graph,
        source_circuit_uri: URIRef,
        min_similarity: float = 0.7,
        max_candidates: int = 20,
        depth_tolerance: float = 0.1,
        require_same_ratio: bool = True
    ) -> List[Tuple[URIRef, float, Dict[str, Any]]]:
        """
        Find transfer candidates for a source circuit.
        
        Implements §2.6 Step 2: Candidate generation via architectural similarity.
        
        Args:
            graph: RDF graph
            source_circuit_uri: Source circuit URI
            min_similarity: Minimum architectural similarity threshold
            max_candidates: Maximum number of candidates to return
            depth_tolerance: Maximum allowed depth difference
            require_same_ratio: Whether to require exact component ratio match
            
        Returns:
            List of (candidate_uri, similarity, features) tuples sorted by similarity
            
        Example:
             matcher = TransferMatcher()
             g = matcher.load_graph("circuits.ttl")
             candidates = matcher.find_transfer_candidates(g, source_uri)
             for cand_uri, sim, feat in candidates:
                 print(f"{cand_uri}: similarity={sim:.3f}")
        """
        # Get source features
        source_features = self.get_architectural_features(graph, source_circuit_uri)
        
        # Get all circuits
        query = f"""
        PREFIX mechint: <{self.MECHINT}>
        
        SELECT DISTINCT ?circuit WHERE {{
            ?circuit a mechint:Circuit .
            FILTER(?circuit != <{source_circuit_uri}>)
        }}
        """
        
        results = graph.query(query)
        all_circuits = [row.circuit for row in results]
        
        # Score candidates
        candidates = []
        for cand_uri in all_circuits:
            cand_features = self.get_architectural_features(graph, cand_uri)
            
            # Filter by depth tolerance
            if source_features['depth_mean'] >= 0 and cand_features['depth_mean'] >= 0:
                depth_diff = abs(source_features['depth_mean'] - cand_features['depth_mean'])
                if depth_diff > depth_tolerance:
                    continue
            
            # Filter by ratio if required
            if require_same_ratio:
                if source_features['component_ratio'] != cand_features['component_ratio']:
                    continue
            
            # Compute similarity
            similarity = self.compute_architectural_similarity(source_features, cand_features)
            
            if similarity >= min_similarity:
                candidates.append((cand_uri, similarity, cand_features))
        
        # Sort by similarity descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:max_candidates]
    
    def validate_transfer(
        self,
        graph: Graph,
        source_uri: URIRef,
        target_uri: URIRef,
        causal_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Validate if transfer is successful (paper 2.6 Step 3).
        
        Transfer is successful if |Δ_src - Δ_tgt| < threshold
        
        Args:
            graph: RDF graph
            source_uri: Source circuit URI
            target_uri: Target circuit URI
            causal_threshold: Maximum allowed causal effect difference
            
        Returns:
            Dict with validation results:
                - success: bool
                - delta_src: float
                - delta_tgt: float
                - delta_diff: float
                - preservation_ratio: float (Δ_tgt / Δ_src)
        """
        # Get causal effects
        delta_src = float(graph.value(source_uri, self.MECHINT.causalEffect) or 0.0)
        delta_tgt = float(graph.value(target_uri, self.MECHINT.causalEffect) or 0.0)
        
        # Compute difference
        delta_diff = abs(delta_src - delta_tgt)
        
        # Check success
        success = delta_diff < causal_threshold
        
        # Preservation ratio
        preservation_ratio = delta_tgt / delta_src if delta_src > 0 else 0.0
        
        return {
            'success': success,
            'delta_src': delta_src,
            'delta_tgt': delta_tgt,
            'delta_diff': delta_diff,
            'preservation_ratio': preservation_ratio,
        }
    
    def find_and_validate_transfers(
        self,
        graph: Graph,
        source_uri: URIRef,
        min_similarity: float = 0.7,
        causal_threshold: float = 0.1,
        max_candidates: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find candidates and validate transfers in one call.
        
        Args:
            graph: RDF graph
            source_uri: Source circuit
            min_similarity: Architectural similarity threshold
            causal_threshold: Causal preservation threshold
            max_candidates: Maximum candidates to check
            
        Returns:
            List of dicts with candidate info and validation results
        """
        # Find candidates
        candidates = self.find_transfer_candidates(
            graph, source_uri,
            min_similarity=min_similarity,
            max_candidates=max_candidates
        )
        
        # Validate each
        results = []
        for cand_uri, similarity, features in candidates:
            validation = self.validate_transfer(
                graph, source_uri, cand_uri,
                causal_threshold=causal_threshold
            )
            
            results.append({
                'candidate_uri': cand_uri,
                'architectural_similarity': similarity,
                'features': features,
                'transfer_validation': validation,
                'transfer_success': validation['success'],
            })
        
        return results
    
    def compute_transfer_statistics(
        self,
        graph: Graph,
        min_similarity: float = 0.7,
        causal_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Compute transfer statistics across all circuits.
        
        Args:
            graph: RDF graph
            min_similarity: Architectural similarity threshold
            causal_threshold: Causal preservation threshold
            
        Returns:
            Dict with statistics:
                - total_circuits: int
                - total_pairs: int
                - successful_transfers: int
                - success_rate: float
                - mean_preservation: float
        """
        # Get all circuits
        query = f"""
        PREFIX mechint: <{self.MECHINT}>
        
        SELECT DISTINCT ?circuit WHERE {{
            ?circuit a mechint:Circuit .
        }}
        """
        
        results = graph.query(query)
        all_circuits = list([row.circuit for row in results])
        
        total_circuits = len(all_circuits)
        successful = 0
        total_pairs = 0
        preservation_ratios = []
        
        # Check all pairs
        for i, src in enumerate(all_circuits):
            for tgt in all_circuits[i+1:]:
                src_features = self.get_architectural_features(graph, src)
                tgt_features = self.get_architectural_features(graph, tgt)
                
                similarity = self.compute_architectural_similarity(src_features, tgt_features)
                
                if similarity >= min_similarity:
                    total_pairs += 1
                    validation = self.validate_transfer(graph, src, tgt, causal_threshold)
                    
                    if validation['success']:
                        successful += 1
                    
                    if validation['preservation_ratio'] > 0:
                        preservation_ratios.append(validation['preservation_ratio'])
        
        success_rate = successful / total_pairs if total_pairs > 0 else 0.0
        mean_preservation = np.mean(preservation_ratios) if preservation_ratios else 0.0
        
        return {
            'total_circuits': total_circuits,
            'total_pairs': total_pairs,
            'successful_transfers': successful,
            'success_rate': success_rate,
            'mean_preservation': mean_preservation,
        }


    
    # print("Transfer module ready for use!")
