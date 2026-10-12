from .config import PathPatchingConfig, PathPatchingAnalysisConfig
from .path_patching import (
    discover_circuit,
    path_patch_attention_heads,
    path_patch_attention_heads_vary_length,
    path_patch_residual_stream,
    necessity_test,
    sufficiency_test,
)
from .path_patching_complete import (
    discover_complete_circuit,
    path_patch_all_components,
)
from .faithfulness import (
    test_circuit_necessity,
    test_circuit_sufficiency,
    compute_faithfulness,
    create_all_components_list,
)
from .utils import (
    compute_logit_diff,
    denoising_metric,
    align_sequences_by_last_token,
)

from .path_patching_to_eap_adapter import PathPatchingToEAPAdapter, create_eap_compatible_graph_data, convert_path_patching_results_to_eap
from .path_patching_eap_integration import run_path_patching_with_eap_viz, load_and_visualize_saved_circuit
__all__ = [
    # Config
    'PathPatchingConfig',
    'PathPatchingAnalysisConfig',

    'discover_circuit',
    'discover_complete_circuit',  # Complete circuit with all components + faithfulness

    # Path patching functions
    'path_patch_attention_heads',
    'path_patch_attention_heads_vary_length',
    'path_patch_residual_stream',
    'path_patch_all_components',

    # Testing functions
    'necessity_test',
    'sufficiency_test',
    'test_circuit_necessity',
    'test_circuit_sufficiency',
    'compute_faithfulness',
    'create_all_components_list',

    # Utilities
    'compute_logit_diff',
    'denoising_metric',
    'align_sequences_by_last_token',
    'PathPatchingToEAPAdapter',
    'create_eap_compatible_graph_data',
    'run_path_patching_with_eap_viz',
    'load_and_visualize_saved_circuit',
    'convert_path_patching_results_to_eap'

]