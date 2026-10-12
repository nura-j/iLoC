"""
Helper utilities for path patching circuit discovery
Following TransformerLens patterns from activation_patching_in_tl_demo.py
"""

import torch
from typing import Dict, Tuple, List
from transformer_lens import utils as tl_utils


def get_hook_name(component: str, layer: int) -> str:
    """
    Get TransformerLens hook name for a component

    Args:
        component: Component name ('resid_pre', 'resid_post', 'attn_out', 'mlp_out', 'z', etc.)
        layer: Layer index

    Returns:
        Hook name string
    """
    return tl_utils.get_act_name(component, layer)


########################################
"""
Helper utilities for path patching circuit discovery
Following TransformerLens patterns from activation_patching_in_tl_demo.py
"""


def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor) -> torch.Tensor:
    """
    Extract logits at the last actual token position for each example

    Args:
        logits: [batch, seq, vocab]
        input_length: [batch] - length of each sequence

    Returns:
        [batch, vocab] - logits at last position per example
    """
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)
    logits_at_last = logits[idx, input_length - 1]  # Index each example at its own last position
    return logits_at_last


def compute_logit_diff(
        logits: torch.Tensor,
        answer_tokens: torch.Tensor,
        input_length: torch.Tensor = None,
        mean: bool = True
) -> torch.Tensor:
    """
    Compute logit difference: logit(correct) - logit(incorrect)
    Handles variable-length sequences

    Args:
        logits: Model output [batch, seq, vocab] or [batch, vocab]
        answer_tokens: Answer indices [batch, 2] where [:, 0] = correct, [:, 1] = incorrect
        input_length: [batch] - length of each sequence (required for variable length)
        mean: Whether to return mean or per-example scores

    Returns:
        Logit difference (scalar if mean=True, else [batch])
    """
    # Handle variable-length sequences
    if len(logits.shape) == 3:
        if input_length is None:
            # Fallback: assume all sequences end at same position
            logits = logits[:, -1, :]
        else:
            # Extract logits at correct position per example
            logits = get_logit_positions(logits, input_length)

    # Gather correct and incorrect logits
    correct_logits = logits.gather(dim=-1, index=answer_tokens[:, 0:1])
    incorrect_logits = logits.gather(dim=-1, index=answer_tokens[:, 1:2])

    # Compute difference
    logit_diff = (correct_logits - incorrect_logits).squeeze(-1)

    if mean:
        return logit_diff.mean()
    return logit_diff


def compute_cnp_metric(
        logits: torch.Tensor,
        answer_tokens: torch.Tensor,
        input_length: torch.Tensor,
        mean: bool = True,
        loss: bool = False
) -> torch.Tensor:
    """
    CNP metric using log probabilities (more numerically stable)

    Args:
        logits: Model output [batch, seq, vocab]
        answer_tokens: [batch, 2] where [:, 0] = correct (canonical), [:, 1] = incorrect (alternative)
        input_length: [batch] - length of each sequence
        mean: Return mean or per-example
        loss: If True, return negative (for optimization)

    Returns:
        CNP scores
    """
    # Get logits at correct positions
    target_logits = get_logit_positions(logits, input_length)

    # Convert to log probabilities
    log_probs = torch.log_softmax(target_logits, dim=-1)

    # Gather probabilities for correct/incorrect answers
    answer_tokens = answer_tokens.to(logits.device).long()
    target_probs = torch.gather(log_probs, -1, answer_tokens)

    # CNP: canonical - alternative
    cnp_scores = target_probs[:, 0] - target_probs[:, 1]

    if loss:
        cnp_scores = -cnp_scores
    if mean:
        cnp_scores = cnp_scores.mean()

    return cnp_scores


def denoising_metric(
        patched_logits: torch.Tensor,
        clean_logits: torch.Tensor,
        corrupted_logits: torch.Tensor,
        answer_tokens: torch.Tensor,
        clean_input_length: torch.Tensor = None,
        corrupted_input_length: torch.Tensor = None,
        use_cnp: bool = False
) -> torch.Tensor:
    """
    Compute denoising metric: (patched - corrupted) / (clean - corrupted)
    Handles variable-length sequences

    Args:
        patched_logits: Logits after patching
        clean_logits: Clean baseline logits
        corrupted_logits: Corrupted baseline logits
        answer_tokens: Answer token indices [batch, 2]
        clean_input_length: Length of clean sequences [batch]
        corrupted_input_length: Length of corrupted sequences [batch]
        use_cnp: Use CNP metric (log probs) instead of logit diff

    Returns:
        Normalized patching effect (scalar)
    """
    if use_cnp:
        # Use CNP metric
        if corrupted_input_length is None:
            raise ValueError("corrupted_input_length required for CNP metric")
        if clean_input_length is None:
            raise ValueError("clean_input_length required for CNP metric")

        patched_score = compute_cnp_metric(patched_logits, answer_tokens, corrupted_input_length, mean=True)
        clean_score = compute_cnp_metric(clean_logits, answer_tokens, clean_input_length, mean=True)
        corrupted_score = compute_cnp_metric(corrupted_logits, answer_tokens, corrupted_input_length, mean=True)
    else:
        # Use logit diff
        patched_score = compute_logit_diff(patched_logits, answer_tokens, corrupted_input_length, mean=True)
        clean_score = compute_logit_diff(clean_logits, answer_tokens, clean_input_length, mean=True)
        corrupted_score = compute_logit_diff(corrupted_logits, answer_tokens, corrupted_input_length, mean=True)

    # Avoid division by zero
    if abs(clean_score - corrupted_score) < 1e-6:
        return torch.tensor(0.0, device=patched_logits.device)

    effect = (patched_score - corrupted_score) / (clean_score - corrupted_score)
    return effect


def create_residual_patching_hook(
        position: int,
        clean_cache: dict,
        hook_name: str
):
    """
    Create hook that patches clean activation at specific position into residual stream

    Args:
        position: Token position to patch
        clean_cache: Cache of clean activations
        hook_name: Name of the hook point

    Returns:
        Hook function
    """

    def hook_fn(corrupted_activation, hook):
        if hook_name not in clean_cache:
            return corrupted_activation

        clean_activation = clean_cache[hook_name]
        corrupted_activation[:, position, :] = clean_activation[:, position, :]
        return corrupted_activation

    return hook_fn


def create_attention_head_patching_hook(
        head_index: int,
        clean_cache: dict,
        hook_name: str
):
    """
    Create hook that patches clean activation for specific attention head

    Args:
        head_index: Which head to patch
        clean_cache: Cache of clean activations
        hook_name: Name of the hook point

    Returns:
        Hook function
    """

    def hook_fn(corrupted_head_output, hook):
        if hook_name not in clean_cache:
            return corrupted_head_output

        clean_head_output = clean_cache[hook_name]
        # Patch this specific head
        corrupted_head_output[:, :, head_index] = clean_head_output[:, :, head_index]
        return corrupted_head_output

    return hook_fn


def create_ablation_hook(
        head_list: List[Tuple[int, int]],
        ablation_type: str = 'zero'
):
    """
    Create hook that ablates (zeros or mean-ablates) specific attention heads

    Args:
        head_list: List of (layer, head) tuples to ablate
        ablation_type: 'zero' or 'mean'

    Returns:
        Hook function
    """

    def hook_fn(activation, hook):
        # Get heads to ablate for this layer
        heads_to_ablate = [head for layer, head in head_list if layer == hook.layer()]

        if len(heads_to_ablate) == 0:
            return activation

        if ablation_type == 'zero':
            # Zero ablation
            activation[:, :, heads_to_ablate, :] = 0
        elif ablation_type == 'mean':
            # Mean ablation
            mean_act = activation.mean(dim=0, keepdim=True)
            activation[:, :, heads_to_ablate, :] = mean_act[:, :, heads_to_ablate, :]

        return activation

    return hook_fn


def align_sequences_by_last_token(
        clean_length: int,
        corrupted_length: int
) -> Dict[int, int]:
    """
    Create position mapping for sequences of different lengths
    Aligns sequences so that their final tokens correspond

    Args:
        clean_length: Length of clean sequence
        corrupted_length: Length of corrupted sequence

    Returns:
        Dictionary mapping corrupted_pos -> clean_pos

    Example:
        clean_length=5, corrupted_length=7
        Corrupted: [0, 1, 2, 3, 4, 5, 6]
        Clean:     [0, 0, 1, 2, 3, 4, 4]
    """
    if clean_length == corrupted_length:
        # Same length: direct mapping
        return {i: i for i in range(clean_length)}

    # Different lengths: proportional mapping
    position_map = {}
    for corrupted_pos in range(corrupted_length):
        # Map proportionally, ensuring last tokens align
        if corrupted_length == 1:
            clean_pos = 0
        else:
            clean_pos = int(round((corrupted_pos / (corrupted_length - 1)) * (clean_length - 1)))

        # Clamp to valid range
        clean_pos = max(0, min(clean_pos, clean_length - 1))
        position_map[corrupted_pos] = clean_pos

    return position_map


def tokenize_prompts(model, prompts):
    """
    Tokenize prompts for the model

    Args:
        model: HookedTransformer model
        prompts: List of string prompts

    Returns:
        tokens: Tokenized tensor [batch, seq_len]
    """
    if isinstance(prompts[0], str):
        tokens = model.to_tokens(prompts, prepend_bos=False)
    else:
        tokens = prompts

    return tokens.to(model.cfg.device)


def compute_input_lengths(tokens: torch.Tensor) -> torch.Tensor:
    """
    Compute actual length of each sequence (assuming padding with specific token)

    Args:
        tokens: [batch, seq_len]

    Returns:
        [batch] - length of each sequence
    """
    # Simple approach: just use sequence length dimension
    # For more sophisticated: could detect padding tokens
    batch_size, seq_len = tokens.shape
    return torch.full((batch_size,), seq_len, dtype=torch.long, device=tokens.device)
