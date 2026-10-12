"""
Adapter to convert Path Patching results to EAP Graph format
Allows path patching to use EAP visualization and OWL storage
"""

import torch
import json
from typing import Dict, List, Tuple, Optional
from transformer_lens import HookedTransformer


class PathPatchingToEAPAdapter:
    """Convert path patching circuit to EAP Graph format"""
    
    def __init__(self, model: HookedTransformer):
        self.model = model
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        
    def convert_circuit_to_eap_format(
        self,
        circuit_components: List[Dict],
        effects: Dict[str, torch.Tensor],
        include_all_nodes: bool = False
    ) -> Dict:
        """
        Convert path patching results to EAP format
        
        Args:
            circuit_components: List of {type, layer, head (optional), effect}
            effects: Dict with component effects
            include_all_nodes: Whether to include all nodes (not just circuit)
            
        Returns:
            Dictionary in EAP format compatible with Graph.from_json()
        """
        
        # Create configuration (matches TransformerLens config)
        cfg = {
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_model': self.model.cfg.d_model,
            'd_head': self.model.cfg.d_head,
            'n_ctx': self.model.cfg.n_ctx,
            'model_name': self.model.cfg.model_name,
        }
        
        # Create nodes dictionary
        nodes = {}
        edges = {}
        
        # Add input node (always in graph)
        nodes['input'] = {'in_graph': True, 'score': 0.0}
        
        # Add logits node (always in graph)
        nodes['logits'] = {'in_graph': True, 'score': 0.0}
        
        # Track which nodes are in circuit
        circuit_nodes = set(['input', 'logits'])
        
        # Process circuit components to identify nodes
        for comp in circuit_components:
            comp_type = comp['type']
            layer = comp['layer']
            
            if comp_type == 'attention_head':
                head = comp['head']
                node_name = f'a{layer}.h{head}'
                nodes[node_name] = {
                    'in_graph': True,
                    'score': float(comp['effect'])
                }
                circuit_nodes.add(node_name)
                
            elif comp_type in ['mlp_layers', 'mlp_out']:
                node_name = f'm{layer}'
                if node_name not in nodes:  # Avoid duplicates
                    nodes[node_name] = {
                        'in_graph': True,
                        'score': float(comp['effect'])
                    }
                    circuit_nodes.add(node_name)
        
        # Add all other nodes if requested (for complete graph)
        if include_all_nodes:
            # Add all attention heads
            for layer in range(self.n_layers):
                for head in range(self.n_heads):
                    node_name = f'a{layer}.h{head}'
                    if node_name not in nodes:
                        effect = effects['attention_heads'][layer, head].item()
                        nodes[node_name] = {
                            'in_graph': False,
                            'score': float(effect)
                        }
                
                # Add all MLP nodes
                node_name = f'm{layer}'
                if node_name not in nodes:
                    effect = effects['mlp_layers'][layer].item()
                    nodes[node_name] = {
                        'in_graph': False,
                        'score': float(effect)
                    }
        
        # Create edges based on circuit nodes
        # Path patching doesn't explicitly track edges, so we infer them
        # from the hierarchical structure
        
        # Rule: Input -> Layer 0 components
        for node_name in circuit_nodes:
            if node_name.startswith('a0') or node_name == 'm0':
                edges[f'input->{node_name}'] = {
                    'score': nodes[node_name]['score'],
                    'in_graph': True
                }
        
        # Rule: Layer i -> Layer i+1
        for layer in range(self.n_layers - 1):
            layer_nodes = [n for n in circuit_nodes if self._get_node_layer(n) == layer]
            next_layer_nodes = [n for n in circuit_nodes if self._get_node_layer(n) == layer + 1]
            
            for src in layer_nodes:
                for dst in next_layer_nodes:
                    # Attention heads can have q/k/v edges
                    if dst.startswith('a'):
                        # Create edges for q, k, v
                        for qkv in ['q', 'k', 'v']:
                            edge_name = f'{src}->{dst}<{qkv}>'
                            edges[edge_name] = {
                                'score': nodes[dst]['score'] / 3,  # Split score among q/k/v
                                'in_graph': True
                            }
                    else:
                        # Regular edge to MLP
                        edge_name = f'{src}->{dst}'
                        edges[edge_name] = {
                            'score': nodes[dst]['score'],
                            'in_graph': True
                        }
        
        # Rule: Last layer -> Logits
        last_layer_nodes = [n for n in circuit_nodes if self._get_node_layer(n) == self.n_layers - 1]
        for src in last_layer_nodes:
            edges[f'{src}->logits'] = {
                'score': nodes[src]['score'],
                'in_graph': True
            }
        
        return {
            'cfg': cfg,
            'nodes': nodes,
            'edges': edges
        }
    
    def _get_node_layer(self, node_name: str) -> int:
        """Extract layer number from node name"""
        if node_name == 'input':
            return -1
        elif node_name == 'logits':
            return self.n_layers
        elif node_name.startswith('a'):
            # Format: a0.h1 -> layer 0
            return int(node_name.split('.')[0][1:])
        elif node_name.startswith('m'):
            # Format: m0 -> layer 0
            return int(node_name[1:])
        else:
            return -1
    
    def save_as_eap_json(
        self,
        circuit_components: List[Dict],
        effects: Dict[str, torch.Tensor],
        filepath: str,
        include_all_nodes: bool = False
    ):
        """Save path patching circuit as EAP JSON"""
        eap_data = self.convert_circuit_to_eap_format(
            circuit_components, effects, include_all_nodes
        )
        
        with open(filepath, 'w') as f:
            json.dump(eap_data, f, indent=2)
        
        print(f"Saved EAP-compatible JSON: {filepath}")
    
    def save_as_eap_pt(
        self,
        circuit_components: List[Dict],
        effects: Dict[str, torch.Tensor],
        filepath: str
    ):
        """Save path patching circuit as EAP PT format"""
        eap_data = self.convert_circuit_to_eap_format(
            circuit_components, effects, include_all_nodes=False
        )
        
        # Convert to PT format (more compact)
        # Create tensor representations
        n_total_nodes = 1 + sum(self.n_heads for _ in range(self.n_layers)) + self.n_layers + 1
        
        nodes_in_graph = torch.zeros(n_total_nodes, dtype=torch.bool)
        nodes_scores = torch.zeros(n_total_nodes, dtype=torch.float32)
        
        # Map nodes to indices
        node_idx = 0
        node_to_idx = {'input': node_idx}
        node_idx += 1
        
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                node_name = f'a{layer}.h{head}'
                node_to_idx[node_name] = node_idx
                
                if node_name in eap_data['nodes']:
                    nodes_in_graph[node_idx] = eap_data['nodes'][node_name]['in_graph']
                    nodes_scores[node_idx] = eap_data['nodes'][node_name]['score']
                
                node_idx += 1
            
            # MLP
            node_name = f'm{layer}'
            node_to_idx[node_name] = node_idx
            
            if node_name in eap_data['nodes']:
                nodes_in_graph[node_idx] = eap_data['nodes'][node_name]['in_graph']
                nodes_scores[node_idx] = eap_data['nodes'][node_name]['score']
            
            node_idx += 1
        
        node_to_idx['logits'] = node_idx
        nodes_in_graph[node_idx] = True
        
        pt_data = {
            'cfg': eap_data['cfg'],
            'nodes_in_graph': nodes_in_graph,
            'nodes_scores': nodes_scores,
            'node_to_idx': node_to_idx,
            'edges': eap_data['edges'],  # Keep edges as dict for now
        }
        
        torch.save(pt_data, filepath)
        print(f"Saved EAP-compatible PT: {filepath}")


def create_eap_compatible_graph_data(
    circuit_components: List[Dict],
    effects: Dict[str, torch.Tensor],
    n_layers: int,
    n_heads: int
) -> Dict:
    """
    Create graph data compatible with EAPVisualizer.plot_circuit_graph()
    
    Args:
        circuit_components: List of circuit components
        effects: Component effects
        n_layers: Number of layers
        n_heads: Number of heads per layer
        
    Returns:
        Dict with 'nodes', 'edges', 'scores' for visualization
    """
    nodes = ['input']
    edges = []
    scores = {}
    
    # Extract unique nodes from circuit
    for comp in circuit_components:
        if comp['type'] == 'attention_head':
            node_name = f"a{comp['layer']}.h{comp['head']}"
        elif comp['type'] in ['mlp_layers', 'mlp_out']:
            node_name = f"m{comp['layer']}"
        else:
            continue
        
        if node_name not in nodes:
            nodes.append(node_name)
    
    nodes.append('logits')
    
    # Create edges (hierarchical structure)
    # Sort nodes by layer
    def get_layer(node_name):
        if node_name == 'input':
            return -1
        elif node_name == 'logits':
            return n_layers
        elif node_name.startswith('a'):
            return int(node_name.split('.')[0][1:])
        elif node_name.startswith('m'):
            return int(node_name[1:])
        return -1
    
    sorted_nodes = sorted(nodes, key=get_layer)
    
    # Connect consecutive layers
    for i in range(len(sorted_nodes) - 1):
        src = sorted_nodes[i]
        src_layer = get_layer(src)
        
        for j in range(i + 1, len(sorted_nodes)):
            dst = sorted_nodes[j]
            dst_layer = get_layer(dst)
            
            # Only connect adjacent layers
            if dst_layer == src_layer + 1:
                edges.append((src, dst))
                
                # Find score
                score = 0.0
                for comp in circuit_components:
                    comp_node = None
                    if comp['type'] == 'attention_head':
                        comp_node = f"a{comp['layer']}.h{comp['head']}"
                    elif comp['type'] in ['mlp_layers', 'mlp_out']:
                        comp_node = f"m{comp['layer']}"
                    
                    if comp_node == dst:
                        score = comp['effect']
                        break
                
                scores[(src, dst)] = score
    
    return {
        'nodes': nodes,
        'edges': edges,
        'scores': scores
    }


def convert_path_patching_results_to_eap(
    results: Dict,
    model: HookedTransformer,
    save_dir: str,
    model_name: str = 'model',
    step: int = 0
):
    """
    Complete conversion pipeline from path patching to EAP format

    Args:
        results: Path patching results dict
        model: HookedTransformer model
        save_dir: Directory to save outputs
        model_name: Name for saved files
        step: Training step number
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    adapter = PathPatchingToEAPAdapter(model)

    # Save as JSON
    json_path = os.path.join(save_dir, f'{model_name}_step_{step}_circuit.json')
    adapter.save_as_eap_json(
        results['circuit_components'],
        results['effects'],
        json_path,
        include_all_nodes=False
    )

    # Save as PT
    pt_path = os.path.join(save_dir, f'{model_name}_step_{step}_circuit.pt')
    adapter.save_as_eap_pt(
        results['circuit_components'],
        results['effects'],
        pt_path
    )

    # Create visualization data
    graph_data = create_eap_compatible_graph_data(
        results['circuit_components'],
        results['effects'],
        model.cfg.n_layers,
        model.cfg.n_heads
    )

    return {
        'json_path': json_path,
        'pt_path': pt_path,
        'graph_data': graph_data
    }
