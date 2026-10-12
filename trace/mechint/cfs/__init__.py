"""
Causal Functional Signature (CFS)

The CFS characterises what each component in a circuit computes,
grounding structural descriptions in causal behavioural evidence.
The CFS design focuses on the linguistic aspect for now.

    σ = {⟨v, δ_v, π_v⟩ : v ∈ V_C}

Two computation steps:
    1. DLA (Direct Logit Attribution) — δ_v per component
    2. Attribution-weighted attention profiles — π_v(r) per token role

Public API:
    compute_cfs(model, graph, dataset, tokenizer) -> CFSResult
    compute_dla(model, graph, examples, tokenizer) -> DLAResult
    compute_attention_profiles(model, graph, examples, tokenizer) -> AttentionProfileResult

Storage:
    save_cfs_json(cfs, path)           — standalone JSON
    save_cfs_prolog(cfs, path)         — .pl facts for ILP_arch
    add_cfs_to_ttl(ttl_path, cfs)      — Layer 2 triples in TTL
    add_cfs_to_results_json(path, cfs) — layer2.cfs in results JSON

Plotting:
    plot_dla(cfs, role, path)
    plot_attention_profiles(cfs, role, path)
    plot_cfs_summary(cfs, role, path)
    plot_dla_comparison(cfs_dict, path)
"""

from .dla import compute_dla, compute_logit_lens, DLAResult, LogitLensResult
from .attention_profile import (
    compute_attention_profiles,
    AttentionProfileResult,
    TOKEN_ROLES,
    DEFAULT_TOKEN_ROLES,
    RoleLabeller,
    linguistic_role_labeller,
    entity_heuristic_labeller,
)
from .cfs import compute_cfs, CFSResult, ComponentCFS
from .storage import (
    save_cfs_json,
    save_cfs_prolog,
    save_logit_lens_json,
    add_cfs_to_ttl,
    add_cfs_to_results_json,
)
from .plotting import (
    plot_dla,
    plot_attention_profiles,
    plot_cfs_summary,
    plot_dla_comparison,
    plot_logit_lens,
    plot_logit_lens_overlay,
)

__all__ = [
    # Computation
    "compute_cfs",
    "compute_dla",
    "compute_attention_profiles",
    # Data classes
    "CFSResult",
    "ComponentCFS",
    "DLAResult",
    "AttentionProfileResult",
    "TOKEN_ROLES",
    # Storage
    "save_cfs_json",
    "save_cfs_prolog",
    "add_cfs_to_ttl",
    "add_cfs_to_results_json",
    # Plotting
    "plot_dla",
    "plot_attention_profiles",
    "plot_cfs_summary",
    "plot_dla_comparison",
]
