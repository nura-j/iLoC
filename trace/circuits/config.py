from dataclasses import dataclass, field
from typing import Optional, List, Literal
import torch


@dataclass
class PathPatchingConfig:
    """Configuration for Path Patching circuit discovery"""

    # Core path patching parameters
    Method = Literal[
        'path-patching',  # Standard path patching (assume input is of same length)
        'path-patching-vary-length',  # this handles variable length sequences
        'path-patching-necessity',  # Necessity test (clean -> corrupted patches)
        'path-patching-sufficiency'  # Sufficiency test (corrupted -> clean patches)
    ]
    method: Method = 'path-patching'

    # Intervention settings
    InterventionType = Literal['denoising', 'noising']
    intervention: InterventionType = 'denoising'  # (denoising: patch clean -> corrupted), (noising: patch corrupted -> clean)

    # Circuit selection
    top_k: int = 50  # Number of top edges to select for circuit if we wanted pruning

    # Batch processing
    batch_size: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Position handling
    track_last_token_only: bool = False  # Whether to only track final token position
    align_by_last_token: bool = True  # For variable length: align sequences by final token

    # Normalization and metrics
    normalize_scores: bool = True  # Normalize by (clean - corrupted) performance
    absolute_threshold: Optional[float] = None  # If set, use absolute threshold instead of top-k (|score|)

    # Performance thresholds
    min_faithfulness: float = 0.7  # Minimum faithfulness for circuit validation
    enforce_min_faithfulness: bool = False  # Enforce min faithfulness during circuit selection

    # Dataset configuration
    dataset_path: Optional[str] = None
    intervention_types: List[str] = field(default_factory=lambda: ['abstraction', 'negation', 'specification'])

    # Output settings
    log_dir: Optional[str] = './circuit_results'
    save_graphs: bool = True
    save_json: bool = True
    save_pt: bool = True
    show_plots: bool = False
    plot_circuit_graph: bool = True

    # Analysis options
    compute_necessity: bool = True  # Run necessity analysis (ablation from clean)
    compute_sufficiency: bool = True  # Run sufficiency analysis (add to ablated)
    per_layer_analysis: bool = False  # Compute layer-wise importance
    per_head_analysis: bool = False  # Compute head-wise importance

    # Memory optimization
    use_gradient_checkpointing: bool = False
    max_cache_size_gb: float = 4.0  # Maximum cache size for activations

    # Debugging and verbose output
    verbose: bool = False
    quiet: bool = False  # Suppress progress bars

    def __post_init__(self):
        """Validate configuration"""
        if self.method not in [
            'path-patching',
            'path-patching-vary-length',
            'path-patching-necessity',
            'path-patching-sufficiency'
        ]:
            raise ValueError(f"Invalid path patching method: {self.method}")

        if self.intervention not in ['denoising', 'noising']:
            raise ValueError(f"Invalid intervention type: {self.intervention}")

        if self.top_k <= 0 and self.absolute_threshold is None:
            raise ValueError("Either top_k must be positive or absolute_threshold must be set")

        if self.min_faithfulness < 0 or self.min_faithfulness > 1:
            raise ValueError("min_faithfulness must be between 0 and 1")

    def to_dict(self):
        """Convert config to dictionary for logging"""
        return {
            'method': self.method,
            'intervention': self.intervention,
            'top_k': self.top_k,
            'batch_size': self.batch_size,
            'track_last_token_only': self.track_last_token_only,
            'align_by_last_token': self.align_by_last_token,
            'normalize_scores': self.normalize_scores,
            'compute_necessity': self.compute_necessity,
            'compute_sufficiency': self.compute_sufficiency,
            'enforce_min_faithfulness': self.enforce_min_faithfulness,
            'min_faithfulness': self.min_faithfulness,
            'dataset_path': self.dataset_path,
            'intervention_types': self.intervention_types,
            'log_dir': self.log_dir,
            'save_graphs': self.save_graphs,
            'save_json': self.save_json,
            'save_pt': self.save_pt,
        }

    @classmethod
    def from_eap_config(cls, eap_config):
        """
        Create PathPatchingConfig from EAPConfig for compatibility

        Args:
            eap_config: EAPConfig instance

        Returns:
            PathPatchingConfig instance
        """
        return cls(
            method='path-patching',
            top_k=eap_config.top_k,
            batch_size=getattr(eap_config, 'eap_batch_size', 10),
            device=eap_config.device,
            track_last_token_only=getattr(eap_config, 'track_last_token_only', False),
            dataset_path=getattr(eap_config, 'cnp_dataset_path', None),
            log_dir=eap_config.log_dir,
            save_graphs=getattr(eap_config, 'save_graphs', True),
            save_json=getattr(eap_config, 'save_json', True),
            save_pt=getattr(eap_config, 'save_pt', True),
            show_plots=getattr(eap_config, 'show_plots', False),
            plot_circuit_graph=getattr(eap_config, 'plot_circuit_step_graph', True),
        )


@dataclass
class PathPatchingAnalysisConfig:
    """Extended configuration for comprehensive path patching analysis"""

    # Base configuration
    base_config: PathPatchingConfig = field(default_factory=PathPatchingConfig)

    # Comparison settings
    compare_methods: List[str] = field(default_factory=lambda: [
        'path-patching',
        'EAP-IG-inputs'
    ])

    # Statistical analysis
    num_bootstrap_samples: int = 100  # For confidence intervals
    confidence_level: float = 0.95

    # Cross-validation
    k_folds: int = 5  # For k-fold validation of circuits
    validation_split: float = 0.2

    # Ablation studies
    ablation_strategies: List[str] = field(default_factory=lambda: [
        'zero',  # Zero out activations
        'mean',  # Replace with mean
        'random'  # Replace with random samples
    ])

    # Temporal analysis
    track_evolution: bool = False  # Track circuit evolution over training
    evolution_steps: List[int] = field(default_factory=lambda: [100, 500, 1000, 5000, 10000])

    def __post_init__(self):
        """Validate extended configuration"""
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            raise ValueError("confidence_level must be between 0 and 1")

        if self.k_folds < 2:
            raise ValueError("k_folds must be at least 2")

        if self.validation_split < 0 or self.validation_split >= 1:
            raise ValueError("validation_split must be between 0 and 1")