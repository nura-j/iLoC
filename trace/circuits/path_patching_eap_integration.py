"""
Complete Example: Path Patching with EAP Visualization
Shows how to discover circuits with path patching and visualize using EAP
"""

import torch
from pathlib import Path
from transformer_lens import HookedTransformer

# Import path patching
from trace.circuits import discover_complete_circuit

# Import EAP visualization

# Import adapter
from trace.circuits import (
    PathPatchingToEAPAdapter,
    create_eap_compatible_graph_data,
    convert_path_patching_results_to_eap
)


def run_path_patching_with_eap_viz(
    model: HookedTransformer,
    dataloader,
    output_dir: str = './circuit_results',
    model_name: str = 'model',
    step: int = 0,
    top_k_per_component: int = 10,
    use_cnp: bool = False
):
    """
    Complete pipeline: Path Patching -> EAP Format -> Visualization
    
    Args:
        model: HookedTransformer model
        dataloader: DataLoader with (clean, corrupted, labels)
        output_dir: Where to save results
        model_name: Name for output files
        step: Training step number
        top_k_per_component: Number of top components per type
        use_cnp: Whether to use CNP metric
    """
    
    print("="*80)
    print("PATH PATCHING CIRCUIT DISCOVERY WITH EAP VISUALIZATION")
    print("="*80)
    
    # ========================================
    # STEP 1: Run Path Patching
    # ========================================
    print("\n[1/4] Running path patching circuit discovery...")
    
    results = discover_complete_circuit(
        model=model,
        dataloader=dataloader,
        top_k_per_component=top_k_per_component,
        component_types=[
            'attention_heads',
            'mlp_layers',
            'attn_out',
            'mlp_out',
            'resid_post',
        ],
        test_faithfulness=True,
        ablation_type='mean',
        use_cnp=use_cnp,
        quiet=False
    )
    
    print(f"\n✓ Circuit discovered:")
    print(f"  - {len(results['circuit_components'])} components")
    print(f"  - Clean performance: {results['clean_performance']:.4f}")
    print(f"  - Corrupted performance: {results['corrupted_performance']:.4f}")
    
    # ========================================
    # STEP 2: Convert to EAP Format
    # ========================================
    print("\n[2/4] Converting to EAP format...")
    
    adapter = PathPatchingToEAPAdapter(model)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save as JSON (human-readable, for inspection)
    json_path = Path(output_dir) / f'{model_name}_step_{step}_circuit.json'
    adapter.save_as_eap_json(
        results['circuit_components'],
        results['effects'],
        str(json_path),
        include_all_nodes=False
    )
    
    # Save as PT (compact, for loading)
    pt_path = Path(output_dir) / f'{model_name}_step_{step}_circuit.pt'
    adapter.save_as_eap_pt(
        results['circuit_components'],
        results['effects'],
        str(pt_path)
    )
    
    print(f"✓ Saved EAP format:")
    print(f"  - JSON: {json_path}")
    print(f"  - PT: {pt_path}")
    
    # ========================================
    # STEP 3: Create Graph Data for Viz
    # ========================================
    print("\n[3/4] Creating visualization data...")
    
    graph_data = create_eap_compatible_graph_data(
        results['circuit_components'],
        results['effects'],
        model.cfg.n_layers,
        model.cfg.n_heads
    )
    
    print(f"✓ Graph data created:")
    print(f"  - {len(graph_data['nodes'])} nodes")
    print(f"  - {len(graph_data['edges'])} edges")
    
    # ========================================
    # STEP 4: Visualize with EAP
    # ========================================
    print("\n[4/4] Creating visualization...")
    
    
    plot_path = Path(output_dir) / 'circuit_graphs' / f'circuit_step_{step}.png'
    print(f"✓ Visualization saved: {plot_path}")
    
    # ========================================
    # STEP 5: Summary
    # ========================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  1. {json_path.name} - Circuit in JSON format")
    print(f"  2. {pt_path.name} - Circuit in PyTorch format")
    print(f"  3. circuit_graphs/circuit_step_{step}.png - Visualization")
    
    if 'faithfulness' in results and results['faithfulness']:
        f = results['faithfulness']
        print(f"\nFaithfulness Metrics:")
        print(f"  Necessity:   {f.get('necessity_faithfulness', 'N/A'):.4f}")
        print(f"  Sufficiency: {f.get('sufficiency_faithfulness', 'N/A'):.4f}")
        print(f"  Overall:     {f.get('overall_faithfulness', 'N/A'):.4f}")
    
    print("\n" + "="*80)
    
    return {
        'results': results,
        'json_path': json_path,
        'pt_path': pt_path,
        'plot_path': plot_path,
        'graph_data': graph_data
    }


def load_and_visualize_saved_circuit(
    json_path: str,
    output_dir: str = None,
    model_name: str = 'model',
    step: int = 0
):
    """
    Load a saved circuit and visualize it
    
    Args:
        json_path: Path to saved JSON circuit
        output_dir: Where to save visualization
        model_name: Model name for plot title
        step: Step number for title
    """
    import json
    
    # Load JSON
    with open(json_path, 'r') as f:
        eap_data = json.load(f)
    
    print(f"Loaded circuit from: {json_path}")
    print(f"  Nodes: {len(eap_data['nodes'])}")
    print(f"  Edges: {len(eap_data['edges'])}")
    
    # Convert to graph data format
    nodes = [name for name, data in eap_data['nodes'].items() if data['in_graph']]
    
    edges = []
    scores = {}
    for edge_name, edge_data in eap_data['edges'].items():
        if edge_data['in_graph']:
            # Parse edge name: "a0.h1->m1" or "m0->a1.h2<q>"
            if '<' in edge_name:
                base_name = edge_name.split('<')[0]
            else:
                base_name = edge_name
            
            src, dst = base_name.split('->')
            edges.append((src, dst))
            scores[(src, dst)] = edge_data['score']
    
    graph_data = {
        'nodes': nodes,
        'edges': edges,
        'scores': scores
    }
    
    # Visualize
    if output_dir is None:
        output_dir = Path(json_path).parent
    
    print(f"✓ Visualization saved to: {output_dir}/circuit_graphs/")


