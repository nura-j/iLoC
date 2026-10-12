"""
Causal Functional Signature
Combines DLA scores and attention profiles into a single signature: σ = {⟨v, δ_v, π_v⟩ : v ∈ V_C}

This is DERIVED (computed), not learned inductively. It feeds into ILP_arch as background knowledge (Layer 2 → Layer 3).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .dla import DLAResult, compute_dla
from .attention_profile import (
    AttentionProfileResult,
    TOKEN_ROLES,
    compute_attention_profiles,
)


@dataclass
class ComponentCFS:
    """CFS entry for a single component: ⟨v, δ_v, π_v⟩."""
    name: str
    dla_score: float
    attention_profile: Dict[str, float]  # role -> weight
    causally_marginal: bool

    def dominant_role(self) -> Optional[str]:
        """The role receiving the most attribution-weighted attention."""
        if not self.attention_profile:
            return None
        return max(self.attention_profile, key=self.attention_profile.get)


@dataclass
class CFSResult:
    """
    Full CFS for a circuit.

    σ = {⟨v, δ_v, π_v⟩ : v ∈ V_C}
    """
    components: Dict[str, ComponentCFS] = field(default_factory=dict)
    epsilon_dla: float = 0.05
    num_examples: int = 0

    # Raw sub-results for downstream use
    dla_result: Optional[DLAResult] = None
    attn_result: Optional[AttentionProfileResult] = None

    def causal_components(self) -> List[ComponentCFS]:
        """Components with |δ_v| >= ε_DLA, sorted by |δ_v| descending."""
        return sorted(
            [c for c in self.components.values() if not c.causally_marginal],
            key=lambda c: abs(c.dla_score),
            reverse=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict for storage in Layer 2."""
        return {
            "epsilon_dla": self.epsilon_dla,
            "num_examples": self.num_examples,
            "components": {
                name: {
                    "dla_score": comp.dla_score,
                    "causally_marginal": comp.causally_marginal,
                    "dominant_role": comp.dominant_role(),
                    "attention_profile": comp.attention_profile,
                }
                for name, comp in self.components.items()
            },
        }

    def to_prolog_facts(self) -> List[str]:
        """
        Export as Prolog-style facts for ILP_arch background knowledge:
            dla_score(v, δ).
            attends_to(v, r, p).
        """
        facts = []
        for name, comp in self.components.items():
            safe_name = name.replace(".", "_")
            facts.append(f"dla_score({safe_name}, {comp.dla_score:.6f}).")
            for role, weight in comp.attention_profile.items():
                facts.append(f"attends_to({safe_name}, {role}, {weight:.6f}).")
            if comp.causally_marginal:
                facts.append(f"causally_marginal({safe_name}).")
        return facts


def compute_cfs(
    model,
    graph,
    examples: List[Dict],
    tokenizer,
    epsilon_dla: float = 0.05,
    token_roles: Optional[tuple] = None,
    role_labeller=None,
) -> CFSResult:
    """
    Compute the full Causal Functional Signature for a circuit.

    Args:
        model: HookedTransformer (TransformerLens)
        graph: EAP Graph with .nodes dict
        examples: list of dicts with keys:
                  'clean'/'prompt' (input text)
                  'correct'/'answer' (correct completion)
        tokenizer: tokenizer
        epsilon_dla: threshold for causally marginal components
        token_roles: custom role vocabulary (default: paper's 5 linguistic roles)
        role_labeller: callable (text, token_strs) → List[str]
                       (default: spaCy dependency parser + heuristic fallback)

    Returns:
        CFSResult containing ⟨v, δ_v, π_v⟩ for each circuit component
    """
    # Step 1: DLA scores
    dla = compute_dla(model, graph, examples, tokenizer, epsilon_dla)

    # Step 2: Attribution-weighted attention profiles
    attn = compute_attention_profiles(
        model, graph, examples, tokenizer,
        dla_scores=dla.scores,
        epsilon_dla=epsilon_dla,
        token_roles=token_roles,
        role_labeller=role_labeller,
    )

    # Resolve the actual role set used (for default profile creation)
    active_roles = token_roles if token_roles is not None else TOKEN_ROLES
    if "other" not in active_roles:
        active_roles = tuple(active_roles) + ("other",)

    # Step 3: Combine into CFS
    result = CFSResult(
        epsilon_dla=epsilon_dla,
        num_examples=dla.num_examples,
        dla_result=dla,
        attn_result=attn,
    )

    # Build per-component entries
    all_names = set(dla.scores.keys()) | set(attn.profiles.keys())
    for name in all_names:
        delta_v = dla.scores.get(name, 0.0)
        profile = attn.profiles.get(name, {r: 0.0 for r in active_roles})
        marginal = abs(delta_v) < epsilon_dla

        result.components[name] = ComponentCFS(
            name=name,
            dla_score=delta_v,
            attention_profile=profile,
            causally_marginal=marginal,
        )

    return result
