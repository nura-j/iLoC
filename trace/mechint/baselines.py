"""
baselines.py
============
Baseline methods for comparing against the COMPASS framework.

1. random_circuit      — size-matched random subgraph
2. component_overlap   — Jaccard similarity over raw circuit nodes/edges
3. full_discovery_time — measure wall-clock time of full EAP pipeline

These serve as controls for the paper (Section 4.3).
"""

import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Random circuit generation
# ---------------------------------------------------------------------------

def random_circuit(
    model,
    num_nodes: int,
    num_edges: int,
    seed: Optional[int] = None,
) -> "Graph":
    """
    Generate a size-matched random circuit from a model.

    Creates a Graph with `num_nodes` randomly selected nodes and
    `num_edges` randomly selected edges (from the set of real edges
    connecting those nodes). This is the "random circuits" baseline
    from the paper — transfer should fail on these.

    Args:
        model:     HookedTransformer (for Graph.from_model)
        num_nodes: number of nodes to include (match target circuit)
        num_edges: number of edges to include (match target circuit)
        seed:      random seed for reproducibility

    Returns:
        Graph with random nodes/edges marked as in_graph.
    """
    from .core.eap_graph import Graph

    if seed is not None:
        random.seed(seed)

    graph = Graph.from_model(model)
    graph.reset(empty=True)

    # Collect all non-logit node names
    all_nodes = [name for name, node in graph.nodes.items()
                 if name != "logits"]

    # Sample nodes
    k = min(num_nodes, len(all_nodes))
    selected_nodes = set(random.sample(all_nodes, k))

    for name in selected_nodes:
        graph.nodes[name].in_graph = True

    # Collect all real edges between selected nodes
    candidate_edges = []
    for edge_name, edge in graph.edges.items():
        if (edge.parent.name in selected_nodes and
            edge.child.name in selected_nodes):
            candidate_edges.append(edge_name)

    # Also include edges to logits from selected nodes
    for edge_name, edge in graph.edges.items():
        if (edge.parent.name in selected_nodes and
            edge.child.name == "logits"):
            candidate_edges.append(edge_name)

    # Sample edges
    e = min(num_edges, len(candidate_edges))
    if e > 0:
        selected_edges = random.sample(candidate_edges, e)
        for edge_name in selected_edges:
            graph.edges[edge_name].in_graph = True
            # Assign random score for consistency
            graph.edges[edge_name].score = random.uniform(0.001, 0.1)

    return graph


def random_circuit_from_json(
    model,
    results_json_path: str,
    seed: Optional[int] = None,
) -> "Graph":
    """
    Generate a random circuit matched to the size of a stored circuit.

    Reads num_nodes and num_edges from the results JSON, then generates
    a random circuit of the same size.
    """
    with open(results_json_path) as f:
        rec = json.load(f)

    metrics = rec.get("layer1", {}).get("metrics", {})
    num_nodes = metrics.get("num_nodes_in_circuit", 10)
    num_edges = metrics.get("num_edges_in_circuit", 50)

    return random_circuit(model, num_nodes, num_edges, seed=seed)


def random_circuit_batch(
    model,
    results_json_path: str,
    n_samples: int = 10,
    base_seed: int = 42,
) -> List["Graph"]:
    """
    Generate multiple random circuits matched to a stored circuit's size.

    Returns a list of n_samples random Graph objects.
    """
    return [
        random_circuit_from_json(model, results_json_path, seed=base_seed + i)
        for i in range(n_samples)
    ]


# ---------------------------------------------------------------------------
# 2. Component overlap (Jaccard baseline)
# ---------------------------------------------------------------------------

def component_overlap(
    circuit_a: Dict[str, Any],
    circuit_b: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute Jaccard similarity over circuit nodes and edges.

    Args:
        circuit_a, circuit_b: layer1.circuit dicts from results JSON
            (each has "nodes" and "edges" lists)

    Returns:
        {
            "node_jaccard": float,   # |nodes_A ∩ nodes_B| / |nodes_A ∪ nodes_B|
            "edge_jaccard": float,   # same for edges
            "combined_jaccard": float # over nodes ∪ edges combined
        }
    """
    nodes_a = {n["name"] for n in circuit_a.get("nodes", [])}
    nodes_b = {n["name"] for n in circuit_b.get("nodes", [])}
    edges_a = {e["name"] for e in circuit_a.get("edges", [])}
    edges_b = {e["name"] for e in circuit_b.get("edges", [])}

    def _jaccard(a, b):
        if not a and not b:
            return 0.0
        return len(a & b) / len(a | b)

    return {
        "node_jaccard": _jaccard(nodes_a, nodes_b),
        "edge_jaccard": _jaccard(edges_a, edges_b),
        "combined_jaccard": _jaccard(nodes_a | edges_a, nodes_b | edges_b),
    }


def component_overlap_matrix(
    results_dir: str,
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """
    Compute pairwise component overlap for all circuits in a results directory.

    Returns (task_list, matrix) where matrix[t1][t2] = combined_jaccard.
    """
    circuits = {}
    for root, _, files in os.walk(results_dir):
        for fname in sorted(files):
            if not fname.endswith("_results.json"):
                continue
            path = os.path.join(root, fname)
            with open(path) as f:
                rec = json.load(f)
            task = rec.get("meta", {}).get("task", fname.replace("_results.json", ""))
            model = rec.get("meta", {}).get("model_name", "unknown")
            key = f"{model}/{task}"
            circuit = rec.get("layer1", {}).get("circuit", {})
            if circuit.get("nodes"):
                circuits[key] = circuit

    tasks = sorted(circuits.keys())
    matrix = {}
    for t1 in tasks:
        matrix[t1] = {}
        for t2 in tasks:
            if t1 == t2:
                matrix[t1][t2] = 1.0
            else:
                ov = component_overlap(circuits[t1], circuits[t2])
                matrix[t1][t2] = round(ov["combined_jaccard"], 4)

    return tasks, matrix


# ---------------------------------------------------------------------------
# 3. Full discovery timing
# ---------------------------------------------------------------------------

def time_full_discovery(
    model,
    tokenizer,
    dataset_path: str,
    task_family: str,
    role: str,
    model_name: str,
    top_k: int = 200,
    ig_steps: int = 3,
) -> Dict[str, Any]:
    """
    Run full EAP-IG discovery and measure wall-clock time.

    Returns dict with timing info and basic metrics.
    """
    from .api import analyze_and_export

    start = time.time()
    results, graph, owl = analyze_and_export(
        model=model,
        cnp_dataset_path=dataset_path,
        task_family=task_family,
        role=role,
        model_name=model_name,
        tokenizer=tokenizer,
        top_k=top_k,
        ig_steps=ig_steps,
        compute_validation=False,
        output_dir="/tmp/compass_timing",
    )
    elapsed = time.time() - start

    return {
        "elapsed_seconds": round(elapsed, 2),
        "faithfulness": results.get("faithfulness"),
        "num_nodes": results.get("num_nodes_in_circuit"),
        "num_edges": results.get("num_edges_in_circuit"),
    }


# ---------------------------------------------------------------------------
# 4. Random circuit transfer failure rate
# ---------------------------------------------------------------------------

def random_transfer_failure_rate(
    model,
    results_json_path: str,
    dataset_path: str,
    tokenizer=None,
    n_samples: int = 10,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate n random circuits and measure their AccT and ΔT.

    Random circuits should have low AccT and low ΔT, confirming that
    transfer success depends on structured correspondence.

    Returns summary stats.
    """
    from .core.eap_evaluate import evaluate_graph, evaluate_baseline
    from .core.eap_graph import Graph
    from .core.eap_cnp import cnp_metric
    from functools import partial

    with open(results_json_path) as f:
        rec = json.load(f)
    metrics = rec.get("layer1", {}).get("metrics", {})
    real_acc_t = metrics.get("AccT", 0)
    real_delta_t = metrics.get("DeltaT", 0)

    # Build dataloader
    try:
        from .core.analyzer import MechIntAnalyzer
        analyzer = MechIntAnalyzer(cnp_dataset_path=dataset_path)
        ds = analyzer.load_cnp_dataset(dataset_path, tokenizer)
        dataloader = ds.to_dataloader(10)
    except Exception as e:
        print(f"  Could not build dataloader: {e}")
        return {"error": str(e)}

    metric_fn = partial(cnp_metric, loss=False, mean=False)

    # Baseline (full model)
    baseline = evaluate_baseline(model, dataloader, metric_fn).mean().item()

    # Ablated (empty graph)
    empty = Graph.from_model(model)
    empty.reset(empty=True)
    ablated = evaluate_graph(
        model, empty, dataloader, metric_fn, skip_clean=False
    ).mean().item()

    random_acc_ts = []
    random_delta_ts = []

    for i in range(n_samples):
        rg = random_circuit_from_json(model, results_json_path, seed=base_seed + i)
        try:
            acc = evaluate_graph(
                model, rg, dataloader, metric_fn
            ).mean().item()
            # ΔT for the random circuit: how much does ablating THIS circuit matter?
            # = baseline - ablated (same for all, since we ablate everything)
            # What we really want: AccT(random_circuit) vs AccT(ablated)
            dt = acc - ablated
            random_acc_ts.append(acc)
            random_delta_ts.append(dt)
        except Exception:
            pass

    if not random_acc_ts:
        return {"error": "all random circuits failed evaluation"}

    return {
        "n_samples": len(random_acc_ts),
        "real_acc_t": real_acc_t,
        "real_delta_t": real_delta_t,
        "random_acc_t_mean": sum(random_acc_ts) / len(random_acc_ts),
        "random_acc_t_std": (sum((x - sum(random_acc_ts)/len(random_acc_ts))**2
                                 for x in random_acc_ts) / len(random_acc_ts)) ** 0.5,
        "random_delta_t_mean": sum(random_delta_ts) / len(random_delta_ts),
        "baseline": baseline,
        "ablated": ablated,
    }
