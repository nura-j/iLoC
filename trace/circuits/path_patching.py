"""
Path Patching for Circuit Discovery
Following TransformerLens activation patching patterns
"""

import torch
from torch import Tensor
from typing import Optional, List, Tuple, Callable
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm
from functools import partial

from .utils import (
    compute_logit_diff,
    denoising_metric,
    create_residual_patching_hook,
    create_attention_head_patching_hook,
    create_ablation_hook,
    tokenize_prompts,
    get_hook_name,
    align_sequences_by_last_token,
)


def path_patch_residual_stream(
    model: HookedTransformer,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    answer_tokens: Tensor,
    clean_cache: dict,
    quiet: bool = False
) -> Tensor:
    """
    Path patch residual stream at each layer and position

    Args:
        model: HookedTransformer model
        clean_tokens: Clean input tokens [batch, seq]
        corrupted_tokens: Corrupted input tokens [batch, seq]
        answer_tokens: Answer indices [batch, 2]
        clean_cache: Cache from clean forward pass
        quiet: Suppress progress bars

    Returns:
        Patching effects [n_layers, seq_len]
    """
    n_layers = model.cfg.n_layers
    seq_len = corrupted_tokens.shape[1]
    device = model.cfg.device

    # Get baselines
    clean_logits = model(clean_tokens)
    corrupted_logits = model(corrupted_tokens)

    # Results matrix
    results = torch.zeros((n_layers, seq_len), device=device, dtype=torch.float32)

    # Patch each layer and position
    iterator = tqdm(range(n_layers), desc="Patching residual") if not quiet else range(n_layers)

    for layer in iterator:
        for pos in range(seq_len):
            hook_name = get_hook_name('resid_post', layer)
            hook_fn = create_residual_patching_hook(pos, clean_cache, hook_name)

            # Run with patch
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, hook_fn)]
            )

            # Compute effect
            effect = denoising_metric(patched_logits, clean_logits, corrupted_logits, answer_tokens)
            results[layer, pos] = effect.item()

    return results


def path_patch_attention_heads(
    model: HookedTransformer,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    answer_tokens: Tensor,
    clean_cache: dict,
    component: str = 'z',  # 'z', 'q', 'k', 'v', 'pattern'
    quiet: bool = False
) -> Tensor:
    """
    Path patch attention heads (patch clean into corrupted)

    Args:
        model: HookedTransformer model
        clean_tokens: Clean input tokens
        corrupted_tokens: Corrupted input tokens
        answer_tokens: Answer indices [batch, 2]
        clean_cache: Cache from clean forward pass
        component: Which attention component ('z', 'q', 'k', 'v', 'pattern')
        quiet: Suppress progress bars

    Returns:
        Patching effects [n_layers, n_heads]
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = model.cfg.device

    # Get baselines
    clean_logits = model(clean_tokens)
    corrupted_logits = model(corrupted_tokens)

    # Results matrix
    results = torch.zeros((n_layers, n_heads), device=device, dtype=torch.float32)

    # Patch each head
    iterator = tqdm(range(n_layers), desc=f"Patching {component}") if not quiet else range(n_layers)

    for layer in iterator:
        for head in range(n_heads):
            hook_name = get_hook_name(component, layer)
            hook_fn = create_attention_head_patching_hook(head, clean_cache, hook_name)

            # Run with patch
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, hook_fn)]
            )

            # Compute effect
            effect = denoising_metric(patched_logits, clean_logits, corrupted_logits, answer_tokens)
            results[layer, head] = effect.item()

    return results


def path_patch_attention_heads_vary_length(
    model: HookedTransformer,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    answer_tokens: Tensor,
    clean_cache: dict,
    component: str = 'z',
    quiet: bool = False
) -> Tensor:
    """
    Path patch attention heads for DIFFERENT LENGTH sequences
    Aligns sequences by their final token position

    Args:
        model: HookedTransformer model
        clean_tokens: Clean input tokens [batch, clean_len]
        corrupted_tokens: Corrupted input tokens [batch, corrupt_len]
        answer_tokens: Answer indices [batch, 2]
        clean_cache: Cache from clean forward pass
        component: Which attention component
        quiet: Suppress progress bars

    Returns:
        Patching effects [n_layers, n_heads]
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = model.cfg.device

    # Get sequence lengths
    clean_len = clean_tokens.shape[1]
    corrupt_len = corrupted_tokens.shape[1]

    # Create position mapping
    position_map = align_sequences_by_last_token(clean_len, corrupt_len)

    # Get baselines (use corrupted length for metric)
    clean_logits = model(clean_tokens)
    corrupted_logits = model(corrupted_tokens)

    # Results matrix
    results = torch.zeros((n_layers, n_heads), device=device, dtype=torch.float32)

    # Patch each head with position alignment
    iterator = tqdm(range(n_layers), desc=f"Patching {component} (var-len)") if not quiet else range(n_layers)

    for layer in iterator:
        for head in range(n_heads):
            hook_name = get_hook_name(component, layer)

            def aligned_hook_fn(corrupted_act, hook):
                """Hook that patches with position alignment"""
                if hook_name not in clean_cache:
                    return corrupted_act

                clean_act = clean_cache[hook_name]
                patched = corrupted_act.clone()

                # Patch head with position mapping
                for corrupt_pos, clean_pos in position_map.items():
                    if corrupt_pos < patched.shape[1] and clean_pos < clean_act.shape[1]:
                        patched[:, corrupt_pos, head] = clean_act[:, clean_pos, head]

                return patched

            # Run with aligned patch
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, aligned_hook_fn)]
            )

            # Compute effect
            effect = denoising_metric(patched_logits, clean_logits, corrupted_logits, answer_tokens)
            results[layer, head] = effect.item()

    return results


def necessity_test(
    model: HookedTransformer,
    tokens: Tensor,
    answer_tokens: Tensor,
    circuit_heads: List[Tuple[int, int]],
    ablation_type: str = 'mean',
    quiet: bool = False
) -> List[float]:
    """
    Test necessity by progressively ablating circuit heads from clean input

    Args:
        model: HookedTransformer model
        tokens: Clean tokens [batch, seq]
        answer_tokens: Answer indices [batch, 2]
        circuit_heads: List of (layer, head) tuples
        ablation_type: 'zero' or 'mean'
        quiet: Suppress progress bars

    Returns:
        List of logit diffs after each ablation
    """
    # Get clean baseline
    clean_logits = model(tokens)
    clean_score = compute_logit_diff(clean_logits, answer_tokens).mean().item()

    scores = [clean_score]
    ablated_heads = []

    iterator = tqdm(circuit_heads, desc="Necessity test") if not quiet else circuit_heads

    for head in iterator:
        ablated_heads.append(head)

        # Get unique layers
        head_layers = set(layer for layer, _ in ablated_heads)
        hook_names = [get_hook_name('z', layer) for layer in head_layers]
        hook_filter = lambda name: name in hook_names

        # Create ablation hook
        hook_fn = create_ablation_hook(ablated_heads, ablation_type)

        # Run with ablation
        ablated_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_filter, hook_fn)]
        )

        # Compute score
        score = compute_logit_diff(ablated_logits, answer_tokens).mean().item()
        scores.append(score)

    return scores


def sufficiency_test(
    model: HookedTransformer,
    tokens: Tensor,
    answer_tokens: Tensor,
    circuit_heads: List[Tuple[int, int]],
    all_heads: List[Tuple[int, int]],
    ablation_type: str = 'mean',
    quiet: bool = False
) -> List[float]:
    """
    Test sufficiency by progressively restoring circuit heads to ablated model

    Args:
        model: HookedTransformer model
        tokens: Clean tokens [batch, seq]
        answer_tokens: Answer indices [batch, 2]
        circuit_heads: List of (layer, head) tuples in circuit
        all_heads: List of all (layer, head) tuples
        ablation_type: 'zero' or 'mean'
        quiet: Suppress progress bars

    Returns:
        List of logit diffs as heads are restored
    """
    # Start with all heads ablated
    ablated_heads = all_heads.copy()

    # Get fully ablated baseline
    head_layers = set(layer for layer, _ in ablated_heads)
    hook_names = [get_hook_name('z', layer) for layer in head_layers]
    hook_filter = lambda name: name in hook_names
    hook_fn = create_ablation_hook(ablated_heads, ablation_type)

    ablated_logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_filter, hook_fn)]
    )
    ablated_score = compute_logit_diff(ablated_logits, answer_tokens).mean().item()

    scores = [ablated_score]

    # Progressively restore circuit heads
    iterator = tqdm(reversed(circuit_heads), desc="Sufficiency test") if not quiet else reversed(circuit_heads)

    for head in iterator:
        if head in ablated_heads:
            ablated_heads.remove(head)

        # Create hook with remaining ablated heads
        head_layers = set(layer for layer, _ in ablated_heads)
        hook_names = [get_hook_name('z', layer) for layer in head_layers]
        hook_filter = lambda name: name in hook_names
        hook_fn = create_ablation_hook(ablated_heads, ablation_type)

        # Run with partial ablation
        restored_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_filter, hook_fn)]
        )

        # Compute score
        score = compute_logit_diff(restored_logits, answer_tokens).mean().item()
        scores.append(score)

    return scores


def discover_circuit(
    model: HookedTransformer,
    dataloader: DataLoader,
    top_k: int = 20,
    component: str = 'z',
    ablation_type: str = 'mean',
    handle_var_length: bool = False,
    quiet: bool = False
) -> dict:
    """
    Complete circuit discovery pipeline with path patching

    Args:
        model: HookedTransformer model
        dataloader: DataLoader yielding (clean, corrupted, labels)
        top_k: Number of top heads to include in circuit
        component: Attention component to patch ('z', 'q', 'k', 'v')
        ablation_type: 'zero' or 'mean'
        handle_var_length: Whether to handle variable-length sequences
        quiet: Suppress progress bars

    Returns:
        Dictionary with circuit and analysis results
    """
    # Step 1: Get one batch
    print("Loading data batch...")
    clean_list, corrupted_list, labels_list = [], [], []
    for clean, corrupted, labels in dataloader:
        clean_list.extend(clean)
        corrupted_list.extend(corrupted)
        labels_list.append(labels)
        # break

    # Tokenize
    clean_tokens = tokenize_prompts(model, clean_list)
    corrupted_tokens = tokenize_prompts(model, corrupted_list)
    answer_tokens = torch.cat(labels_list, dim=0).to(model.cfg.device)

    print(f"Clean shape: {clean_tokens.shape}")
    print(f"Corrupted shape: {corrupted_tokens.shape}")
    print(f"Answers shape: {answer_tokens.shape}")

    # Step 2: Run clean forward to get cache
    print("\nRunning clean forward pass...")
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    clean_perf = compute_logit_diff(clean_logits, answer_tokens).mean().item()
    print(f"Clean performance: {clean_perf:.4f}")

    corrupted_logits = model(corrupted_tokens)
    corrupted_perf = compute_logit_diff(corrupted_logits, answer_tokens).mean().item()
    print(f"Corrupted performance: {corrupted_perf:.4f}")

    # Step 3: Path patch attention heads
    print(f"\nPatching attention {component}...")
    if handle_var_length and clean_tokens.shape[1] != corrupted_tokens.shape[1]:
        attention_effects = path_patch_attention_heads_vary_length(
            model, clean_tokens, corrupted_tokens, answer_tokens, clean_cache, component, quiet
        )
    else:
        attention_effects = path_patch_attention_heads(
            model, clean_tokens, corrupted_tokens, answer_tokens, clean_cache, component, quiet
        )

    print(f"Effects shape: {attention_effects.shape}")
    print(f"Max effect: {attention_effects.max().item():.4f}")
    print(f"Mean effect: {attention_effects.mean().item():.4f}")

    # Step 4: Select top-k heads
    flat_effects = attention_effects.flatten()
    top_k_actual = min(top_k, flat_effects.numel())
    top_k_indices = torch.topk(flat_effects, k=top_k_actual).indices

    circuit_heads = []
    print(f"\nTop {top_k_actual} heads:")
    for idx in top_k_indices:
        layer = idx.item() // model.cfg.n_heads
        head = idx.item() % model.cfg.n_heads
        effect = attention_effects[layer, head].item()
        circuit_heads.append((layer, head))
        print(f"  L{layer}H{head}: {effect:.4f}")

    # Step 5: Necessity test
    print(f"\nNecessity test (ablating {len(circuit_heads)} heads)...")
    necessity_scores = necessity_test(
        model, clean_tokens, answer_tokens, circuit_heads, ablation_type, quiet
    )

    # Step 6: Sufficiency test
    print("\nSufficiency test...")
    all_heads = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    sufficiency_scores = sufficiency_test(
        model, clean_tokens, answer_tokens, circuit_heads, all_heads, ablation_type, quiet
    )

    # Step 7: Compute faithfulness
    circuit_perf = necessity_scores[-1]
    if abs(clean_perf - corrupted_perf) > 1e-6:
        faithfulness = (clean_perf - circuit_perf) / (clean_perf - corrupted_perf)
    else:
        faithfulness = 0.0

    print(f"\n{'='*60}")
    print("CIRCUIT DISCOVERY RESULTS")
    print(f"{'='*60}")
    print(f"Clean performance:     {clean_perf:.4f}")
    print(f"Corrupted performance: {corrupted_perf:.4f}")
    print(f"Circuit performance:   {circuit_perf:.4f}")
    print(f"Faithfulness:          {faithfulness:.4f}")
    print(f"Circuit size:          {len(circuit_heads)} heads")
    print(f"{'='*60}")

    return {
        'circuit_heads': circuit_heads,
        'attention_effects': attention_effects,
        'necessity_scores': necessity_scores,
        'sufficiency_scores': sufficiency_scores,
        'clean_performance': clean_perf,
        'corrupted_performance': corrupted_perf,
        'circuit_performance': circuit_perf,
        'faithfulness': faithfulness,
        'num_heads': len(circuit_heads),
    }