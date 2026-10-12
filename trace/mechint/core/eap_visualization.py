import os
from typing import Dict, Optional, Literal, List, Tuple, Set
# import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx

# Import for compatibility with graph.py
try:
    import pygraphviz as pgv

    HAS_PYGRAPHVIZ = True
except ImportError:
    HAS_PYGRAPHVIZ = False


# Node colors by type (using ColorBrewer-inspired palette)
NODE_COLORS = {
    "input": "#FED976",  # Warm yellow
    "attn": "#A1D99B",  # Soft green (attention heads)
    "mlp": "#9ECAE1",  # Soft blue (MLP layers)
    "logits": "#C994C7",  # Soft purple (output)
    "residual": "#FDBB84",  # Orange (residual stream)
    "other": "#E0E0E0",  # Neutral gray
}

# Edge colors by importance/type
EDGE_COLORS = {
    'positive': '#2C7BB6',  # Deep blue for positive edges
    'negative': '#D7191C',  # Deep red for negative edges
    'q': '#7B3294',  # Purple for query
    'k': '#008837',  # Green for key
    'v': '#C51B7D',  # Magenta for value
}


def get_node_fill_color(name: str) -> str:
    """Get fill color for a node - compatible with graph.py"""
    node_type = classify_node_type(name)
    return NODE_COLORS.get(node_type, NODE_COLORS["other"])


def get_color(qkv: Optional[Literal['q', 'k', 'v']], score: float) -> str:
    """Get edge color - compatible with graph.py"""
    if qkv is not None:
        return EDGE_COLORS.get(qkv, EDGE_COLORS['positive'])
    elif score < 0:
        return EDGE_COLORS['negative']
    else:
        return EDGE_COLORS['positive']


def get_edge_color_and_width(score: float, qkv: Optional[Literal['q', 'k', 'v']] = None) -> Tuple[str, float]:
    """
    Get edge color and width based on score and type

    Args:
        score: Edge importance score
        qkv: Optional attention type (q/k/v)

    Returns:
        (color, width) tuple
    """
    # Width proportional to absolute score (1-5 range)
    width = min(5.0, max(1.0, abs(score) * 3))

    # Color based on type or sign
    color = get_color(qkv, score)

    return color, width


def get_edge_style(score: float) -> str:
    """Get edge line style based on importance"""
    abs_score = abs(score)
    if abs_score > 0.5:
        return 'solid'
    elif abs_score > 0.2:
        return 'dashed'
    else:
        return 'dotted'


def classify_node_type(name: str) -> str:
    """Classify node by its name pattern"""
    n = name.strip().lower()

    if n == "input":
        return "input"
    if n == "logits":
        return "logits"

    # MLP layers: m0, m1, m2, etc.
    if n.startswith("m") and n[1:].isdigit():
        return "mlp"

    # Attention heads: a0.h1, a1.h2, etc.
    if n.startswith("a") and ".h" in n:
        return "attn"

    # Residual stream: h0, h1, etc.
    if n.startswith("h") and n[1:].isdigit():
        return "residual"

    return "other"


def get_node_color(name: str) -> str:
    """Get fill color for a node"""
    node_type = classify_node_type(name)
    return NODE_COLORS.get(node_type, NODE_COLORS["other"])


class EAPVisualizer:
    """Enhanced visualizer for EAP circuit analysis"""
    def __init__(self, log_dir: str, config=None):
        self.log_dir = log_dir
        self.config = config

        # High DPI for publication quality
        self.dpi = 300

        # Font settings
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['font.size'] = 10

    def plot_circuit_graph(
            self,
            graph_data: Dict,
            step: int,
            title: Optional[str] = None,
            figsize: Tuple[int, int] = (16, 12)
    ):
        """
        Plot a circuit graph with improved aesthetics

        Args:
            graph_data: Dictionary with 'nodes', 'edges', and 'scores'
            step: Training step number
            title: Optional custom title
            figsize: Figure size in inches
        """

        # Create figure with high DPI
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)

        # Extract data
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        scores = graph_data.get('scores', {})

        if not nodes or not edges:
            print(f"Warning: Empty graph at step {step}")
            return

        # Build NetworkX graph
        G = nx.DiGraph()
        G.add_nodes_from(nodes)

        # Add edges with attributes
        for (src, dst), score in scores.items():
            if (src, dst) in edges or f"{src}->{dst}" in edges:
                G.add_edge(src, dst, weight=score)

        # Layout (hierarchical for better readability)
        pos = self._compute_layout(G, nodes)

        # Draw edges with varying styles
        self._draw_edges(G, pos, ax)

        # Draw nodes
        self._draw_nodes(G, pos, ax)

        # Add legend
        self._add_legend(ax)

        # Title and styling
        if title is None:
            title = f"Circuit Graph at Step {step}"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()

        # Save with high quality
        if self.log_dir:
            plot_dir = os.path.join(self.log_dir, 'circuit_graphs')
            os.makedirs(plot_dir, exist_ok=True)
            filename = os.path.join(plot_dir, f'circuit_step_{step}.png')
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"Saved circuit graph: {filename}")

        if self.config and self.config.show_plots:
            plt.show()
        else:
            plt.close()

    def _compute_layout(self, G: nx.DiGraph, nodes: List[str], layers: Dict[str, List[str]]= None, numLayers: int = 6) -> Dict:
        """Compute hierarchical layout for circuit graph"""

        # Classify nodes by layer
        if layers is None:
            layers = {'input': [],'output': []}
            for i in range(numLayers):
                layers[f'layer{i}'] = []

        for node in nodes:
            if node.lower() == 'input':
                layers['input'].append(node)
            elif node.lower() == 'logits':
                layers['output'].append(node)
            else:
                # Extract layer number from node name (e.g., m0, a1.h2)
                if node.startswith('m') or node.startswith('a'):
                    try:
                        layer_num = int(node[1]) if node[1].isdigit() else 0
                        layer_key = f'layer{layer_num}'
                        if layer_key in layers:
                            layers[layer_key].append(node)
                        else:
                            layers['layer0'].append(node)
                    except:
                        layers['layer0'].append(node)
                else:
                    layers['layer0'].append(node)

        # Assign positions
        pos = {}
        y_spacing = 1.5
        current_y = 0

        # for layer_name in ['input', 'layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'output']:
        for layer_name in layers.keys():
            layer_nodes = layers[layer_name]
            if not layer_nodes:
                continue

            x_spacing = 2.0
            total_width = (len(layer_nodes) - 1) * x_spacing
            start_x = -total_width / 2

            for i, node in enumerate(sorted(layer_nodes)):
                pos[node] = (start_x + i * x_spacing, current_y)

            current_y -= y_spacing

        return pos

    def _draw_edges(self, G: nx.DiGraph, pos: Dict, ax):
        """Draw edges with varying styles"""

        for src, dst, data in G.edges(data=True):
            score = data.get('weight', 0)

            # Get color, width, and style
            color, width = get_edge_color_and_width(score)
            style = get_edge_style(score)

            # Draw edge
            x_values = [pos[src][0], pos[dst][0]]
            y_values = [pos[src][1], pos[dst][1]]

            ax.plot(x_values, y_values,
                    color=color,
                    linewidth=width,
                    linestyle=style,
                    alpha=0.7,
                    zorder=1)

            # Add arrowhead
            dx = x_values[1] - x_values[0]
            dy = y_values[1] - y_values[0]
            ax.annotate('',
                        xy=(x_values[1], y_values[1]),
                        xytext=(x_values[1] - 0.15 * dx, y_values[1] - 0.15 * dy),
                        arrowprops=dict(arrowstyle='->',
                                        color=color,
                                        lw=width * 0.7,
                                        alpha=0.7),
                        zorder=1)

    def _draw_nodes(self, G: nx.DiGraph, pos: Dict, ax):
        """Draw nodes with colors and labels"""

        for node in G.nodes():
            x, y = pos[node]

            # Get node color
            color = get_node_color(node)

            # Draw node circle
            circle = plt.Circle((x, y), 0.4,
                                facecolor=color,
                                edgecolor='black',
                                linewidth=2,
                                zorder=2)
            ax.add_patch(circle)

            # Add label
            ax.text(x, y, node,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=9,
                    fontweight='bold',
                    zorder=3)

    def _add_legend(self, ax):
        """Add comprehensive legend"""

        # Node type legend
        node_legend_elements = [
            mpatches.Patch(facecolor=NODE_COLORS['input'],
                           edgecolor='black', label='Input'),
            mpatches.Patch(facecolor=NODE_COLORS['attn'],
                           edgecolor='black', label='Attention'),
            mpatches.Patch(facecolor=NODE_COLORS['mlp'],
                           edgecolor='black', label='MLP'),
            mpatches.Patch(facecolor=NODE_COLORS['logits'],
                           edgecolor='black', label='Output'),
        ]

        # Edge type legend
        edge_legend_elements = [
            Line2D([0], [0], color=EDGE_COLORS['positive'],
                   linewidth=3, label='Positive Effect'),
            Line2D([0], [0], color=EDGE_COLORS['negative'],
                   linewidth=3, label='Negative Effect'),
            Line2D([0], [0], color='gray', linewidth=3,
                   linestyle='solid', label='Strong (>0.5)'),
            Line2D([0], [0], color='gray', linewidth=2,
                   linestyle='dashed', label='Medium (0.2-0.5)'),
            Line2D([0], [0], color='gray', linewidth=1,
                   linestyle='dotted', label='Weak (<0.2)'),
        ]

        # Create legends
        node_legend = ax.legend(handles=node_legend_elements,
                                loc='upper left',
                                title='Node Types',
                                frameon=True,
                                fontsize=9)
        ax.add_artist(node_legend)

        edge_legend = ax.legend(handles=edge_legend_elements,
                                loc='upper right',
                                title='Edge Properties',
                                frameon=True,
                                fontsize=9)

    def plot_eap_evolution(self, eap_history: Dict[int, Dict], model_name: str):
        """Plot EAP metrics evolution with improved styling"""
        steps = sorted(eap_history.keys())
        if not steps:
            print("No EAP history to plot.")
            return

        # Extract results (handle both tuple and dict formats)
        results = {}
        for step in steps:
            if isinstance(eap_history[step], tuple):
                results[step] = eap_history[step][0]
            else:
                results[step] = eap_history[step]

        # Extract metrics
        faithfulness = [results[step].get('faithfulness', 0) for step in steps]
        baseline_perf = [results[step].get('baseline_performance', 0) for step in steps]
        circuit_perf = [results[step].get('circuit_performance', 0) for step in steps]

        # Create figure with high DPI
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)

        # Plot 1: Faithfulness evolution
        ax1.plot(steps, faithfulness, color='#2C7BB6', linewidth=2.5,
                 marker='o', markersize=6, label='Faithfulness')
        ax1.fill_between(steps, faithfulness, alpha=0.3, color='#2C7BB6')
        ax1.set_xlabel('Training Step', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Circuit Faithfulness', fontsize=11, fontweight='bold')
        ax1.set_title('Circuit Faithfulness Evolution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Plot 2: Performance comparison
        ax2.plot(steps, baseline_perf, color='#4DAF4A', linewidth=2.5,
                 marker='s', markersize=6, label='Baseline')
        ax2.plot(steps, circuit_perf, color='#E41A1C', linewidth=2.5,
                 marker='^', markersize=6, label='Circuit')
        ax2.set_xlabel('Training Step', fontsize=11, fontweight='bold')
        ax2.set_ylabel('CNP Performance', fontsize=11, fontweight='bold')
        ax2.set_title('Performance: Baseline vs Circuit', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', frameon=True, fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        plt.tight_layout()

        # Save with high quality
        if self.log_dir:
            plot_dir = os.path.join(self.log_dir, 'eap_plots')
            os.makedirs(plot_dir, exist_ok=True)
            filename = os.path.join(plot_dir, f'{model_name}_eap_evolution.png')
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"Saved EAP evolution plot: {filename}")

        if self.config and self.config.show_plots:
            plt.show()
        else:
            plt.close()