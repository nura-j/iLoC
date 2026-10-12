"""
Attribution-weighted attention profiles

For each causally relevant component v, compute an attribution-weighted
attention profile: the attention distribution of v over input positions,
weighted by |δ_v|.

Input positions are labelled by a **role labeller** — a callable that maps
(text, token_strings) → list of role labels.  The default labeller uses
spaCy dependency parsing with a heuristic fallback, but tasks can supply
their own (e.g. IOI tags repeated vs. unique names).

The resulting profile: π_v : ROLE → [0, 1]

records how strongly v's causally weighted attention falls on each
token role across D_T.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Type alias for role labellers
# ---------------------------------------------------------------------------
# A RoleLabeller takes (full_text, list_of_token_strings) and returns
# a list of role labels (one per token).  Any string is a valid role label;
# unknown roles are folded into "other" during aggregation.
RoleLabeller = Callable[[str, List[str]], List[str]]


# ---------------------------------------------------------------------------
# Default token roles (paper §3.3)
# ---------------------------------------------------------------------------
DEFAULT_TOKEN_ROLES = ("subject", "object", "verb", "entity", "other")
TOKEN_ROLES = DEFAULT_TOKEN_ROLES  # backward-compatible alias


@dataclass
class AttentionProfileResult:
    """Attribution-weighted attention profiles for circuit components."""
    # component_name -> {role -> weight ∈ [0,1]}
    profiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    num_examples: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Built-in role labellers
# ═══════════════════════════════════════════════════════════════════════════

def linguistic_role_labeller(text: str, token_strs: List[str]) -> List[str]:
    """
    Default labeller: spaCy dependency parsing → {subject, object, verb, entity, other}.
    Falls back to capitalisation heuristic if spaCy is unavailable.
    """
    spacy_roles = _try_spacy_roles(text)
    if spacy_roles is not None:
        aligned = _align_roles_to_tokens(text, spacy_roles, token_strs)
        if aligned is not None:
            return aligned
    return _heuristic_roles(token_strs)


def entity_heuristic_labeller(text: str, token_strs: List[str]) -> List[str]:
    """Simple labeller: capitalised tokens → 'entity', everything else → 'other'."""
    return _heuristic_roles(token_strs)


# Backward-compatible alias
label_token_roles = linguistic_role_labeller


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers for the linguistic labeller
# ═══════════════════════════════════════════════════════════════════════════

def _try_spacy_roles(text: str) -> Optional[List[str]]:
    """Use spaCy to assign dependency-based roles. Returns None if unavailable."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        return None

    doc = nlp(text)
    roles = []
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            roles.append("subject")
        elif token.dep_ in ("dobj", "pobj", "iobj", "dative"):
            roles.append("object")
        elif token.pos_ == "VERB":
            roles.append("verb")
        elif token.ent_type_:
            roles.append("entity")
        else:
            roles.append("other")
    return roles


def _align_roles_to_tokens(
    text: str,
    word_roles: List[str],
    token_strs: List[str],
) -> Optional[List[str]]:
    """Align word-level roles to subword tokens via character offsets."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)

        char_roles = ["other"] * len(text)
        for token, role in zip(doc, word_roles):
            for i in range(token.idx, token.idx + len(token.text)):
                if i < len(text):
                    char_roles[i] = role

        aligned = []
        pos = 0
        for tok_str in token_strs:
            clean = tok_str.replace("Ġ", " ").replace("▁", " ").lstrip()
            idx = text.find(clean, pos) if clean else pos
            if idx == -1:
                idx = pos
            role = char_roles[min(idx, len(char_roles) - 1)] if char_roles else "other"
            aligned.append(role)
            pos = idx + max(len(clean), 1)
        return aligned
    except (ImportError, OSError):
        return None


def _heuristic_roles(token_strs: List[str]) -> List[str]:
    """Capitalisation heuristic: Title Case → 'entity', else → 'other'."""
    roles = []
    for tok in token_strs:
        clean = tok.replace("Ġ", "").replace("▁", "").strip()
        if clean and clean[0].isupper() and len(clean) > 1:
            roles.append("entity")
        else:
            roles.append("other")
    return roles


# ═══════════════════════════════════════════════════════════════════════════
# Main computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_attention_profiles(
    model,
    graph,
    examples: List[Dict],
    tokenizer,
    dla_scores: Optional[Dict[str, float]] = None,
    epsilon_dla: float = 0.05,
    token_roles: Optional[Sequence[str]] = None,
    role_labeller: Optional[RoleLabeller] = None,
) -> AttentionProfileResult:
    """
    Compute attribution-weighted attention profiles for attention heads.

    Args:
        model: HookedTransformer
        graph: EAP Graph with .nodes dict
        examples: list of dicts with 'clean'/'prompt' key
        tokenizer: tokenizer for decoding token strings
        dla_scores: {component_name: δ_v} from DLA computation.
                    If None, all heads are weighted equally.
        epsilon_dla: components below this threshold are skipped
        token_roles: role vocabulary to aggregate into.
                     Defaults to DEFAULT_TOKEN_ROLES.
                     Labels not in this set are folded into "other".
        role_labeller: callable (text, token_strs) → List[str].
                       Defaults to linguistic_role_labeller (spaCy + heuristic).
                       Tasks can provide their own, e.g.:
                         lambda text, toks: ["name" if t[0].isupper() else "other" for t in toks]

    Returns:
        AttentionProfileResult with π_v(r) profiles
    """
    # Resolve role vocabulary
    roles_to_use = tuple(token_roles) if token_roles is not None else DEFAULT_TOKEN_ROLES
    if "other" not in roles_to_use:
        roles_to_use = roles_to_use + ("other",)

    # Resolve labeller
    labeller = role_labeller if role_labeller is not None else linguistic_role_labeller

    result = AttentionProfileResult()

    # Identify attention heads in the circuit
    attn_heads = {}
    for name, node in graph.nodes.items():
        if not node.in_graph:
            continue
        if name.startswith("a") and ".h" in name:
            head_idx = int(name.split(".h")[1])
            attn_heads[name] = (node.layer, head_idx)

    if not attn_heads:
        return result

    # Skip causally marginal components if DLA scores provided
    if dla_scores:
        attn_heads = {
            name: (layer, head)
            for name, (layer, head) in attn_heads.items()
            if abs(dla_scores.get(name, 0.0)) >= epsilon_dla
        }

    # Accumulate role distributions
    accum = {name: defaultdict(list) for name in attn_heads}

    for ex in examples:
        prompt = ex.get("clean", ex.get("prompt", ""))
        if not prompt:
            continue

        tokens = model.to_tokens(prompt)  # [1, seq_len]
        seq_len = tokens.shape[1]

        # Decode tokens and label roles
        token_ids = tokens[0].tolist()
        token_strs = [tokenizer.decode([tid]) for tid in token_ids]
        roles = labeller(prompt, token_strs)

        # Pad/truncate to match seq_len
        if len(roles) < seq_len:
            roles = roles + ["other"] * (seq_len - len(roles))
        roles = roles[:seq_len]

        # Forward pass with cache
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Extract attention profiles for each head
        for name, (layer, head_idx) in attn_heads.items():
            pattern_key = f"blocks.{layer}.attn.hook_pattern"
            if pattern_key not in cache:
                continue
            attn_pattern = cache[pattern_key][0, head_idx, -1, :]  # [seq_len]
            attn_weights = attn_pattern.cpu().numpy()

            # Weight by |δ_v| if available
            w = abs(dla_scores.get(name, 1.0)) if dla_scores else 1.0

            # Aggregate attention by role; fold unknowns into "other"
            role_attn = defaultdict(float)
            for pos, r in enumerate(roles):
                mapped = r if r in roles_to_use else "other"
                if pos < len(attn_weights):
                    role_attn[mapped] += float(attn_weights[pos]) * w

            # Normalise to [0, 1]
            total = sum(role_attn.values())
            if total > 0:
                for r in roles_to_use:
                    accum[name][r].append(role_attn.get(r, 0.0) / total)
            else:
                for r in roles_to_use:
                    accum[name][r].append(0.0)

    # Average over examples
    for name in attn_heads:
        profile = {}
        for r in roles_to_use:
            vals = accum[name][r]
            profile[r] = float(np.mean(vals)) if vals else 0.0
        result.profiles[name] = profile

    result.num_examples = len(examples)
    return result
