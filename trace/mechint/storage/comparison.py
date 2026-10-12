"""
Specification Comparison Module

Implements comparison operations for semantic specifications (paper 2.5)

This module provides:
1. Jaccard similarity via SPARQL (using structured P_H)
2. θ-subsumption checking
3. Predicate overlap computation
4. Specification equivalence checking

"""

from typing import Set, List, Dict, Any, Tuple
from rdflib import Graph, URIRef, Namespace


class SpecificationComparator:
    """
    Compare semantic specifications using structured predicate sets.
    
    Uses the structured P_H storage from Phase 2 Issue #8 to enable
    efficient set operations via SPARQL.
    """
    
    def __init__(self, namespace_uri: str = "http://purl.org/mechint#"):
        """
        Initialize comparator.
        
        Args:
            namespace_uri: Mechint ontology namespace
        """
        self.MECHINT = Namespace(namespace_uri)
    
    def load_graph(self, ttl_path: str) -> Graph:
        """Load RDF graph from Turtle file"""
        g = Graph()
        g.parse(ttl_path, format='turtle')
        return g
    
    def get_predicate_names(self, graph: Graph, spec_uri: URIRef) -> Set[str]:
        """
        Get predicate names from structured P_H storage.
        
        Args:
            graph: RDF graph
            spec_uri: Specification URI
            
        Returns:
            Set of predicate names (e.g., {'binds', 'scaffold', 'compatible'})
        """
        # Query structured predicate storage
        query = f"""
        PREFIX mechint: <{self.MECHINT}>
        
        SELECT ?predName WHERE {{
            <{spec_uri}> mechint:hasPredicateInSet ?pred .
            ?pred mechint:predicateName ?predName .
        }}
        """
        
        results = graph.query(query)
        return {str(row.predName) for row in results}
    
    def get_predicate_names_from_string(self, graph: Graph, spec_uri: URIRef) -> Set[str]:
        """
        Fallback: Get predicates from string storage.
        
        Args:
            graph: RDF graph
            spec_uri: Specification URI
            
        Returns:
            Set of predicate names parsed from comma-separated string
        """
        pred_string = graph.value(spec_uri, self.MECHINT.predicates)
        if pred_string:
            return {p.strip() for p in str(pred_string).split(',') if p.strip()}
        return set()
    
    def compute_jaccard_similarity(
        self,
        graph: Graph,
        spec1_uri: URIRef,
        spec2_uri: URIRef,
        use_structured: bool = True
    ) -> float:
        """
        Compute Jaccard similarity: Jaccard(P1, P2) = |P1 ∩ P2| / |P1 ∪ P2|
        
        Args:
            graph: RDF graph
            spec1_uri: First specification URI
            spec2_uri: Second specification URI
            use_structured: Use structured P_H (faster) vs string fallback
            
        Returns:
            Jaccard similarity score in [0, 1]
            
        Example:
            comp = SpecificationComparator()
            g = comp.load_graph("spec.ttl")
            sim = comp.compute_jaccard_similarity(g, spec1, spec2)
            print(f"Similarity: {sim:.3f}")
        """
        if use_structured:
            preds1 = self.get_predicate_names(graph, spec1_uri)
            preds2 = self.get_predicate_names(graph, spec2_uri)
            
            # Fallback to string if structured not available
            if not preds1:
                preds1 = self.get_predicate_names_from_string(graph, spec1_uri)
            if not preds2:
                preds2 = self.get_predicate_names_from_string(graph, spec2_uri)
        else:
            preds1 = self.get_predicate_names_from_string(graph, spec1_uri)
            preds2 = self.get_predicate_names_from_string(graph, spec2_uri)
        
        # Compute set operations
        intersection = preds1 & preds2
        union = preds1 | preds2
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def check_theta_subsumption(
        self,
        graph: Graph,
        spec_a_uri: URIRef,
        spec_b_uri: URIRef,
        use_structured: bool = True
    ) -> bool:
        """
        Check if spec_a subsumes spec_b via θ-subsumption.
        
        θ-subsumption: spec_a subsumes spec_b if predicates(spec_a) ⊆ predicates(spec_b)
        
        Interpretation: spec_a is more GENERAL than spec_b
        
        Args:
            graph: RDF graph
            spec_a_uri: Potentially more general specification
            spec_b_uri: Potentially more specific specification
            use_structured: Use structured P_H vs string fallback
            
        Returns:
            True if spec_a subsumes spec_b (spec_a is more general)
            
        Example:
             # spec_a: {binds, scaffold}
             # spec_b: {binds, scaffold, compatible}
             subsumes = comp.check_theta_subsumption(g, spec_a, spec_b)
             assert subsumes == True  # spec_a is more general
        """
        if use_structured:
            preds_a = self.get_predicate_names(graph, spec_a_uri)
            preds_b = self.get_predicate_names(graph, spec_b_uri)
            
            if not preds_a:
                preds_a = self.get_predicate_names_from_string(graph, spec_a_uri)
            if not preds_b:
                preds_b = self.get_predicate_names_from_string(graph, spec_b_uri)
        else:
            preds_a = self.get_predicate_names_from_string(graph, spec_a_uri)
            preds_b = self.get_predicate_names_from_string(graph, spec_b_uri)
        
        # Check subset relation: P_a ⊆ P_b
        return preds_a.issubset(preds_b)
    
    def compute_predicate_overlap(
        self,
        graph: Graph,
        spec1_uri: URIRef,
        spec2_uri: URIRef,
        use_structured: bool = True
    ) -> Dict[str, Any]:
        """
        Compute detailed predicate overlap statistics.
        
        Args:
            graph: RDF graph
            spec1_uri: First specification
            spec2_uri: Second specification
            use_structured: Use structured P_H
            
        Returns:
            Dict with keys:
                - shared: Set[str] (predicates in both)
                - only_in_1: Set[str] (unique to spec1)
                - only_in_2: Set[str] (unique to spec2)
                - overlap_count: int
                - jaccard: float
                - dice: float (Dice coefficient)
        """
        if use_structured:
            preds1 = self.get_predicate_names(graph, spec1_uri)
            preds2 = self.get_predicate_names(graph, spec2_uri)
            
            if not preds1:
                preds1 = self.get_predicate_names_from_string(graph, spec1_uri)
            if not preds2:
                preds2 = self.get_predicate_names_from_string(graph, spec2_uri)
        else:
            preds1 = self.get_predicate_names_from_string(graph, spec1_uri)
            preds2 = self.get_predicate_names_from_string(graph, spec2_uri)
        
        shared = preds1 & preds2
        only_1 = preds1 - preds2
        only_2 = preds2 - preds1
        union = preds1 | preds2
        
        # Jaccard
        jaccard = len(shared) / len(union) if union else 0.0
        
        # Dice coefficient: 2|P1 ∩ P2| / (|P1| + |P2|)
        dice = 2 * len(shared) / (len(preds1) + len(preds2)) if (preds1 or preds2) else 0.0
        
        return {
            'shared': shared,
            'only_in_1': only_1,
            'only_in_2': only_2,
            'overlap_count': len(shared),
            'size_1': len(preds1),
            'size_2': len(preds2),
            'jaccard': jaccard,
            'dice': dice,
        }
    
    def check_equivalence(
        self,
        graph: Graph,
        spec1_uri: URIRef,
        spec2_uri: URIRef,
        use_structured: bool = True
    ) -> bool:
        """
        Check if two specifications are equivalent (same predicate set).
        
        Args:
            graph: RDF graph
            spec1_uri: First specification
            spec2_uri: Second specification
            use_structured: Use structured P_H
            
        Returns:
            True if P1 == P2
        """
        if use_structured:
            preds1 = self.get_predicate_names(graph, spec1_uri)
            preds2 = self.get_predicate_names(graph, spec2_uri)
            
            if not preds1:
                preds1 = self.get_predicate_names_from_string(graph, spec1_uri)
            if not preds2:
                preds2 = self.get_predicate_names_from_string(graph, spec2_uri)
        else:
            preds1 = self.get_predicate_names_from_string(graph, spec1_uri)
            preds2 = self.get_predicate_names_from_string(graph, spec2_uri)
        
        return preds1 == preds2
    
    def find_similar_specifications(
        self,
        graph: Graph,
        spec_uri: URIRef,
        min_similarity: float = 0.5,
        max_results: int = 10,
        use_structured: bool = True
    ) -> List[Tuple[URIRef, float]]:
        """
        Find specifications similar to a given specification.
        
        Args:
            graph: RDF graph
            spec_uri: Query specification
            min_similarity: Minimum Jaccard threshold
            max_results: Maximum number of results
            use_structured: Use structured P_H
            
        Returns:
            List of (spec_uri, similarity) tuples, sorted by similarity descending
        """
        # Get all specifications
        query = f"""
        PREFIX mechint: <{self.MECHINT}>
        
        SELECT DISTINCT ?spec WHERE {{
            ?spec a mechint:SemanticSpec .
            FILTER(?spec != <{spec_uri}>)
        }}
        """
        
        results = graph.query(query)
        all_specs = [row.spec for row in results]
        
        # Compute similarities
        similarities = []
        for other_spec in all_specs:
            sim = self.compute_jaccard_similarity(graph, spec_uri, other_spec, use_structured)
            if sim >= min_similarity:
                similarities.append((other_spec, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:max_results]
    
    def compare_all_pairs(
        self,
        graph: Graph,
        min_similarity: float = 0.0,
        use_structured: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Compare all pairs of specifications in the graph.
        
        Args:
            graph: RDF graph
            min_similarity: Only return pairs with similarity >= threshold
            use_structured: Use structured P_H
            
        Returns:
            List of comparison dicts sorted by similarity descending
        """
        # Get all specifications
        query = f"""
        PREFIX mechint: <{self.MECHINT}>
        
        SELECT DISTINCT ?spec WHERE {{
            ?spec a mechint:SemanticSpec .
        }}
        """
        
        results = graph.query(query)
        all_specs = list([row.spec for row in results])
        
        # Compare all pairs
        comparisons = []
        for i, spec1 in enumerate(all_specs):
            for spec2 in all_specs[i+1:]:
                overlap = self.compute_predicate_overlap(graph, spec1, spec2, use_structured)
                
                if overlap['jaccard'] >= min_similarity:
                    comparisons.append({
                        'spec1': spec1,
                        'spec2': spec2,
                        'jaccard': overlap['jaccard'],
                        'dice': overlap['dice'],
                        'shared_predicates': overlap['shared'],
                        'overlap_count': overlap['overlap_count'],
                    })
        
        # Sort by Jaccard descending
        comparisons.sort(key=lambda x: x['jaccard'], reverse=True)
        
        return comparisons


# # Example usage and testing
# if __name__ == "__main__":
#     """
#     Example of how to use the comparison module
#     """
#     # Initialize comparator
#     comp = SpecificationComparator()
    
    # Load graph
    # g = comp.load_graph("circuit_with_specs.ttl")
    
    
    # print("Comparison module ready for use!")
