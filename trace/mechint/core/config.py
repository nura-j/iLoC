import os
from dataclasses import dataclass, field
from typing import Optional, List, Literal
import torch


@dataclass
class MechIntConfig:
    """
    Configuration for MechInt analyzer

    Combines EAP configuration with OWL export settings

    Args:
        method: EAP method ('EAP-IG-inputs', 'EAP-IG-logits', etc.)
        top_k: Number of top edges to keep
        ig_steps: Number of integrated gradient steps
        cnp_dataset_path: Path to counterfactual dataset
        save_owl: Whether to export circuits to OWL format
        owl_output_dir: Directory for OWL exports
        task_family: Task family name (e.g., 'SemanticRoles', 'IOI')
        role: Specific role/phenomenon (e.g., 'INSTRUMENT', 'NameMover')
        **kwargs: Additional parameters
    """

    # Core EAP parameters
    Method = Literal['EAP', 'EAP-IG-inputs', 'clean-corrupted',
        'EAP-IG-activations', 'information-flow-routes', 'exact', 'EAP-IG-inputs-vary-length',
        'path-patching','path-patching-vary-length','path-patching-standalone']
    method: Method = 'EAP-IG-inputs'
    ig_steps: int = 5
    top_k: int = 50

    # CNP dataset configuration
    cnp_dataset_path: Optional[str] = None
    intervention_type: str = 'role_cross'
    semantic_roles: List[str] = field(default_factory=lambda: ['Agent', 'Patient', 'Theme'])

    # Analysis settings
    eap_batch_size: int = 10
    # optional list of edge budgets to evaluate faithfulness at
    # (attribution is k-independent; only selection+evaluation change)
    k_sweep: Optional[List[int]] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    track_last_token_only: bool = False  # Whether to track only the last token

    # Output settings
    log_dir: Optional[str] = './analysis_results'
    save_graphs: bool = True
    save_json: bool = True
    save_pt: bool = True
    show_plots: bool = False
    plot_circuit_step_graph: bool = True

    # MechInt-specific settings
    save_owl: bool = True
    owl_output_dir: str = "./data/experiments"
    task_family: Optional[str] = None
    role: Optional[str] = None

    # Validation statistics
    compute_validation: bool = False  # Whether to compute κ statistics (slow: ~1200 evaluate_graph calls)
    epsilon: float = 0.05  # ε-minimality threshold
    n_permutations: int = 100  # Permutation test iterations
    validation_verbose: bool = True  # Show validation progress

    #ILP configs
    run_ilp_learning: bool = False  # Enable ILP
    ilp_examples_path: Optional[str] = None  # Path to examples.json

    def __post_init__(self):
        """Validate configuration"""
        valid_methods = ['EAP', 'EAP-IG-inputs', 'EAP-IG-activations', 'clean-corrupted',
                        'EAP-IG-inputs-vary-length', 'path-patching', 'path-patching-vary-length',
                        'information-flow-routes', 'exact', 'path-patching-standalone']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid EAP method: {self.method}")

        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

        # Ensure log_dir is set if save_owl is True
        if self.save_owl and not self.log_dir:
            self.log_dir = './analysis_results'
            os.makedirs(self.log_dir, exist_ok=True)