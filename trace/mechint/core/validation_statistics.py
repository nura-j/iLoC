"""
Validation Statistics Module

Implements validation components for κ = ⟨Acc_T, Δ_T, confidence, ε-minimality⟩

This module provides:
1. ε-minimality validation via greedy edge ablation
2. Permutation test for statistical significance
3. Integration with EAP analyzer

"""

from typing import Callable, Dict, Any, Optional
import numpy as np
from tqdm import tqdm


def validate_epsilon_minimality(
    graph,  # EAP Graph
    model,  # Transformer model
    dataloader,  # DataLoader
    metric_fn: Callable,  # Evaluation metric function
    epsilon: float = 0.05,
    track_last_token_only: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify circuit is ε-minimal via greedy edge ablation.
    
    A circuit is ε-minimal if removing ANY edge causes performance to drop
    by more than ε. This validates that all edges are essential.
    
    Args:
        graph: EAP Graph with circuit edges
        model: Transformer model
        dataloader: Data loader for evaluation
        metric_fn: Function to compute task accuracy
        epsilon: Threshold for essential edges (default 0.05 = 5%)
        track_last_token_only: Whether to evaluate only last token
        verbose: Whether to show progress bar
        
    Returns:
        Dict with keys:
            - epsilon_minimal: bool (True if circuit is ε-minimal)
            - circuit_performance: float (baseline circuit performance)
            - non_essential_edges: List[str] (edges that can be removed)
            - min_drop: float (minimum performance drop across all edges)
            - max_drop: float (maximum performance drop)
            - mean_drop: float (mean performance drop)
            
    Example:
       result = validate_epsilon_minimality(graph, model, loader, metric)
        if result['epsilon_minimal']:
             print("Circuit is ε-minimal - all edges essential")
        else:
             print(f"Non-essential edges: {result['non_essential_edges']}")
    """
    from .eap_evaluate import evaluate_graph
    
    # Evaluate baseline circuit performance
    circuit_edges = [e for e in graph.edges.values() if e.in_graph]
    
    if len(circuit_edges) == 0:
        return {
            'epsilon_minimal': True,  # Empty circuit is trivially minimal
            'circuit_performance': 0.0,
            'non_essential_edges': [],
            'min_drop': 0.0,
            'max_drop': 0.0,
            'mean_drop': 0.0,
        }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ε-Minimality Validation (ε = {epsilon})")
        print(f"{'='*60}")
        print(f"Circuit has {len(circuit_edges)} edges")
    
    # Baseline circuit performance
    circuit_perf = evaluate_graph(
        model, graph, dataloader, metric_fn,
        track_last_token_only=track_last_token_only
    ).mean().item()
    
    if verbose:
        print(f"Baseline circuit performance: {circuit_perf:.4f}")
        print(f"Testing edge removal (threshold: {epsilon:.4f})...")
    
    # Test each edge
    non_essential_edges = []
    performance_drops = []
    
    iterator = tqdm(circuit_edges) if verbose else circuit_edges
    
    for edge in iterator:
        if not edge.in_graph:
            continue
            
        # Temporarily remove edge
        edge.in_graph = False
        
        # Evaluate reduced circuit
        reduced_perf = evaluate_graph(
            model, graph, dataloader, metric_fn,
            track_last_token_only=track_last_token_only
        ).mean().item()
        
        # Restore edge
        edge.in_graph = True
        
        # Compute drop
        perf_drop = circuit_perf - reduced_perf
        performance_drops.append(perf_drop)
        
        # Check if edge is essential
        if perf_drop <= epsilon:
            edge_name = f"{edge.parent.name} → {edge.child.name}"
            non_essential_edges.append(edge_name)
            if verbose:
                print(f"  Non-essential edge: {edge_name} (drop: {perf_drop:.4f})")
    
    # Compute statistics
    is_minimal = len(non_essential_edges) == 0
    min_drop = min(performance_drops) if performance_drops else 0.0
    max_drop = max(performance_drops) if performance_drops else 0.0
    mean_drop = np.mean(performance_drops) if performance_drops else 0.0
    
    if verbose:
        print(f"\n{'='*60}")
        if is_minimal:
            print(f" Circuit is ε-minimal (all {len(circuit_edges)} edges essential)")
        else:
            print(f"X Circuit is NOT ε-minimal ({len(non_essential_edges)} non-essential edges)")
        print(f"Performance drop range: [{min_drop:.4f}, {max_drop:.4f}]")
        print(f"Mean drop: {mean_drop:.4f}")
        print(f"{'='*60}\n")
    
    return {
        'epsilon_minimal': is_minimal,
        'circuit_performance': circuit_perf,
        'non_essential_edges': non_essential_edges,
        'min_drop': min_drop,
        'max_drop': max_drop,
        'mean_drop': mean_drop,
    }


def permutation_test(
    graph,  # EAP Graph
    model,  # Transformer model
    dataloader,  # DataLoader
    metric_fn: Callable,
    baseline_performance: float,
    n_permutations: int = 1000,
    track_last_token_only: bool = False,
    verbose: bool = True,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Compute p-value via permutation test for statistical significance.
    
    Tests null hypothesis: "The circuit's performance is due to random edge selection"
    
    Args:
        graph: EAP Graph with circuit edges
        model: Transformer model
        dataloader: Data loader for evaluation
        metric_fn: Function to compute task accuracy
        baseline_performance: Full model performance (for Δ_T computation)
        n_permutations: Number of random circuits to test (default 1000)
        track_last_token_only: Whether to evaluate only last token
        verbose: Whether to show progress bar
        seed: Random seed for reproducibility
        
    Returns:
        Dict with keys:
            - p_value: float (probability observed Δ_T is due to chance)
            - observed_delta: float (actual circuit Δ_T)
            - null_distribution: List[float] (Δ_T values for random circuits)
            - significant: bool (True if p < 0.05)
            - percentile: float (observed Δ_T percentile in null distribution)
            
    Example:
         result = permutation_test(graph, model, loader, metric, baseline)
         if result['significant']:
             print(f"Circuit is significant (p={result['p_value']:.4f})")
         else:
             print("Circuit performance not statistically significant")
    """
    from .eap_evaluate import evaluate_graph
    import random
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Permutation Test (n={n_permutations})")
        print(f"{'='*60}")
    
    # Get circuit size
    circuit_edges = [e for e in graph.edges.values() if e.in_graph]
    n_circuit_edges = len(circuit_edges)
    
    if n_circuit_edges == 0:
        return {
            'p_value': 1.0,
            'observed_delta': 0.0,
            'null_distribution': [],
            'significant': False,
            'percentile': 0.0,
        }
    
    # Compute observed Δ_T
    circuit_perf = evaluate_graph(
        model, graph, dataloader, metric_fn,
        track_last_token_only=track_last_token_only
    ).mean().item()
    
    observed_delta = baseline_performance - circuit_perf
    
    if verbose:
        print(f"Circuit size: {n_circuit_edges} edges")
        print(f"Observed Δ_T: {observed_delta:.4f}")
        print(f"Generating null distribution...")
    
    # Generate null distribution
    null_distribution = []
    all_edges = list(graph.edges.values())
    
    iterator = tqdm(range(n_permutations)) if verbose else range(n_permutations)
    
    for _ in iterator:
        # Save current state
        original_state = {e: e.in_graph for e in all_edges}
        
        # Create random circuit of same size
        for edge in all_edges:
            edge.in_graph = False
        
        random_edges = random.sample(all_edges, n_circuit_edges)
        for edge in random_edges:
            edge.in_graph = True
        
        # Evaluate random circuit
        random_perf = evaluate_graph(
            model, graph, dataloader, metric_fn,
            track_last_token_only=track_last_token_only
        ).mean().item()
        
        # Compute Δ_T for random circuit
        random_delta = baseline_performance - random_perf
        null_distribution.append(random_delta)
        
        # Restore original state
        for edge in all_edges:
            edge.in_graph = original_state[edge]
    
    # Compute p-value
    # p-value = proportion of random circuits with Δ_T >= observed Δ_T
    p_value = sum(d >= observed_delta for d in null_distribution) / n_permutations
    
    # Compute percentile
    percentile = sum(d < observed_delta for d in null_distribution) / n_permutations * 100
    
    # Statistical significance (α = 0.05)
    significant = p_value < 0.05
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"  Observed Δ_T: {observed_delta:.4f}")
        print(f"  Null mean: {np.mean(null_distribution):.4f}")
        print(f"  Null std: {np.std(null_distribution):.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Percentile: {percentile:.1f}%")
        print(f"  Significant: {'✓ Yes' if significant else '✗ No'} (α=0.05)")
        print(f"{'='*60}\n")
    
    return {
        'p_value': p_value,
        'observed_delta': observed_delta,
        'null_distribution': null_distribution,
        'significant': significant,
        'percentile': percentile,
    }


def compute_validation_statistics(
    graph,
    model,
    dataloader,
    metric_fn: Callable,
    baseline_performance: float,
    epsilon: float = 0.05,
    n_permutations: int = 1000,
    track_last_token_only: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute complete validation statistics κ = ⟨Acc_T, Δ_T, confidence, ε-minimality⟩
    
    Note: 'confidence' comes from ILP_sem (not computed here)
    
    Args:
        graph: EAP Graph
        model: Transformer model
        dataloader: DataLoader
        metric_fn: Evaluation metric
        baseline_performance: Full model performance (Acc_T)
        epsilon: ε-minimality threshold
        n_permutations: Number for permutation test
        track_last_token_only: Evaluation mode
        verbose: Progress output
        
    Returns:
        Dict with validation statistics ready for OWL export
    """
    from .eap_evaluate import evaluate_graph
    
    # Compute circuit performance and Δ_T
    circuit_perf = evaluate_graph(
        model, graph, dataloader, metric_fn,
        track_last_token_only=track_last_token_only
    ).mean().item()
    
    causal_effect = baseline_performance - circuit_perf  # Δ_T
    
    # ε-minimality validation
    epsilon_result = validate_epsilon_minimality(
        graph, model, dataloader, metric_fn,
        epsilon=epsilon,
        track_last_token_only=track_last_token_only,
        verbose=verbose
    )
    
    # Permutation test
    perm_result = permutation_test(
        graph, model, dataloader, metric_fn,
        baseline_performance=baseline_performance,
        n_permutations=n_permutations,
        track_last_token_only=track_last_token_only,
        verbose=verbose
    )
    
    return {
        # κ components
        'task_accuracy': baseline_performance,  # Acc_T
        'causal_effect': causal_effect,  # Δ_T
        # 'confidence': <comes from ILP_sem>
        'epsilon_minimal': epsilon_result['epsilon_minimal'],
        'permutation_pvalue': perm_result['p_value'],
        
        # Additional statistics
        'epsilon_minimal_details': epsilon_result,
        'permutation_test_details': perm_result,
        'statistically_significant': perm_result['significant'],
    }

