import torch
from torch import Tensor
from typing import Optional, List, Tuple, Dict
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm

from trace.circuits.utils import (
    compute_logit_diff,
    denoising_metric,
    tokenize_prompts,
    get_hook_name,
    compute_input_lengths,
    align_sequences_by_last_token,
)


def path_patch_all_components(
        model: HookedTransformer,
        clean_tokens: Tensor,
        corrupted_tokens: Tensor,
        answer_tokens: Tensor,
        clean_cache: dict,
        clean_input_length: Tensor = None,
        corrupted_input_length: Tensor = None,
        use_cnp: bool = False,
        quiet: bool = False
) -> Dict[str, Tensor]:
    """
    Path patch ALL circuit components with variable-length support

    For attention heads: uses position alignment when lengths differ
    For other components: patches entire activation
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = model.cfg.device

    # Compute input lengths if not provided
    if clean_input_length is None:
        clean_input_length = compute_input_lengths(clean_tokens)
    if corrupted_input_length is None:
        corrupted_input_length = compute_input_lengths(corrupted_tokens)

    # Check if sequences have different lengths
    clean_len = clean_tokens.shape[1]
    corrupt_len = corrupted_tokens.shape[1]
    different_lengths = (clean_len != corrupt_len)

    if different_lengths and not quiet:
        print(f"⚠ Different sequence lengths detected: clean={clean_len}, corrupted={corrupt_len}")
        print("  Using position alignment for attention heads")

    # Create position mapping for variable lengths
    if different_lengths:
        position_map = align_sequences_by_last_token(clean_len, corrupt_len)
    else:
        position_map = {i: i for i in range(clean_len)}

    # Get baselines
    clean_logits = model(clean_tokens)
    corrupted_logits = model(corrupted_tokens)

    results = {
        'attention_heads': torch.zeros((n_layers, n_heads), device=device),
        'mlp_layers': torch.zeros(n_layers, device=device),
        'attn_out': torch.zeros(n_layers, device=device),
        'mlp_out': torch.zeros(n_layers, device=device),
        'resid_pre': torch.zeros(n_layers, device=device),
        'resid_mid': torch.zeros(n_layers, device=device),
        'resid_post': torch.zeros(n_layers, device=device),
    }

    # 1. Patch attention heads (with position alignment)
    if not quiet:
        print("\n1. Patching attention heads...")

    for layer in tqdm(range(n_layers), desc="Attention heads", disable=quiet):
        for head in range(n_heads):
            hook_name = get_hook_name('z', layer)

            if different_lengths:
                # Use position-aligned patching
                def head_hook(activation, hook, head_idx=head, name=hook_name, pos_map=position_map):
                    if name not in clean_cache:
                        return activation
                    clean_act = clean_cache[name]
                    patched = activation.clone()

                    # Patch each position according to alignment
                    for corrupt_pos, clean_pos in pos_map.items():
                        if corrupt_pos < patched.shape[1] and clean_pos < clean_act.shape[1]:
                            patched[:, corrupt_pos, head_idx] = clean_act[:, clean_pos, head_idx]

                    return patched
            else:
                # Same length: direct patching
                def head_hook(activation, hook, head_idx=head, name=hook_name):
                    if name not in clean_cache:
                        return activation
                    clean_act = clean_cache[name]
                    patched = activation.clone()
                    patched[:, :, head_idx] = clean_act[:, :, head_idx]
                    return patched

            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, head_hook)]
            )

            effect = denoising_metric(
                patched_logits, clean_logits, corrupted_logits, answer_tokens,
                clean_input_length, corrupted_input_length, use_cnp
            )
            results['attention_heads'][layer, head] = effect.item()

    # 2. Patch MLP layers
    if not quiet:
        print("\n2. Patching MLP layers...")

    for layer in tqdm(range(n_layers), desc="MLP layers", disable=quiet):
        hook_name = get_hook_name('mlp_out', layer)

        if different_lengths:
            # Use position-aligned patching
            def mlp_hook(activation, hook, name=hook_name, pos_map=position_map):
                if name not in clean_cache:
                    return activation
                clean_act = clean_cache[name]
                patched = activation.clone()

                # Patch each position according to alignment
                for corrupt_pos, clean_pos in pos_map.items():
                    if corrupt_pos < patched.shape[1] and clean_pos < clean_act.shape[1]:
                        patched[:, corrupt_pos, :] = clean_act[:, clean_pos, :]

                return patched
        else:
            # Same length: direct patching
            def mlp_hook(activation, hook, name=hook_name):
                if name not in clean_cache:
                    return activation
                return clean_cache[name].clone()

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, mlp_hook)]
        )

        effect = denoising_metric(
            patched_logits, clean_logits, corrupted_logits, answer_tokens,
            clean_input_length, corrupted_input_length, use_cnp
        )
        results['mlp_layers'][layer] = effect.item()
        results['mlp_out'][layer] = effect.item()

    # 3. Patch attention outputs
    if not quiet:
        print("\n3. Patching attention outputs...")

    for layer in tqdm(range(n_layers), desc="Attention outputs", disable=quiet):
        hook_name = get_hook_name('attn_out', layer)

        if different_lengths:
            # Use position-aligned patching
            def attn_out_hook(activation, hook, name=hook_name, pos_map=position_map):
                if name not in clean_cache:
                    return activation
                clean_act = clean_cache[name]
                patched = activation.clone()

                # Patch each position according to alignment
                for corrupt_pos, clean_pos in pos_map.items():
                    if corrupt_pos < patched.shape[1] and clean_pos < clean_act.shape[1]:
                        patched[:, corrupt_pos, :] = clean_act[:, clean_pos, :]

                return patched
        else:
            # Same length: direct patching
            def attn_out_hook(activation, hook, name=hook_name):
                if name not in clean_cache:
                    return activation
                return clean_cache[name].clone()

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, attn_out_hook)]
        )

        effect = denoising_metric(
            patched_logits, clean_logits, corrupted_logits, answer_tokens,
            clean_input_length, corrupted_input_length, use_cnp
        )
        results['attn_out'][layer] = effect.item()

    # 4. Patch residual stream
    if not quiet:
        print("\n4. Patching residual stream...")

    for layer in tqdm(range(n_layers), desc="Residual stream", disable=quiet):
        # resid_pre
        hook_name_pre = get_hook_name('resid_pre', layer)

        if different_lengths:
            def resid_pre_hook(activation, hook, name=hook_name_pre, pos_map=position_map):
                if name not in clean_cache:
                    return activation
                clean_act = clean_cache[name]
                patched = activation.clone()

                for corrupt_pos, clean_pos in pos_map.items():
                    if corrupt_pos < patched.shape[1] and clean_pos < clean_act.shape[1]:
                        patched[:, corrupt_pos, :] = clean_act[:, clean_pos, :]

                return patched
        else:
            def resid_pre_hook(activation, hook, name=hook_name_pre):
                if name not in clean_cache:
                    return activation
                return clean_cache[name].clone()

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name_pre, resid_pre_hook)]
        )

        effect = denoising_metric(
            patched_logits, clean_logits, corrupted_logits, answer_tokens,
            clean_input_length, corrupted_input_length, use_cnp
        )
        results['resid_pre'][layer] = effect.item()

        # resid_mid (if exists)
        hook_name_mid = get_hook_name('resid_mid', layer)

        if hook_name_mid in clean_cache:
            if different_lengths:
                def resid_mid_hook(activation, hook, name=hook_name_mid, pos_map=position_map):
                    if name not in clean_cache:
                        return activation
                    clean_act = clean_cache[name]
                    patched = activation.clone()

                    for corrupt_pos, clean_pos in pos_map.items():
                        if corrupt_pos < patched.shape[1] and clean_pos < clean_act.shape[1]:
                            patched[:, corrupt_pos, :] = clean_act[:, clean_pos, :]

                    return patched
            else:
                def resid_mid_hook(activation, hook, name=hook_name_mid):
                    if name not in clean_cache:
                        return activation
                    return clean_cache[name].clone()

            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name_mid, resid_mid_hook)]
            )

            effect = denoising_metric(
                patched_logits, clean_logits, corrupted_logits, answer_tokens,
                clean_input_length, corrupted_input_length, use_cnp
            )
            results['resid_mid'][layer] = effect.item()

        # resid_post
        hook_name_post = get_hook_name('resid_post', layer)

        if different_lengths:
            def resid_post_hook(activation, hook, name=hook_name_post, pos_map=position_map):
                if name not in clean_cache:
                    return activation
                clean_act = clean_cache[name]
                patched = activation.clone()

                for corrupt_pos, clean_pos in pos_map.items():
                    if corrupt_pos < patched.shape[1] and clean_pos < clean_act.shape[1]:
                        patched[:, corrupt_pos, :] = clean_act[:, clean_pos, :]

                return patched
        else:
            def resid_post_hook(activation, hook, name=hook_name_post):
                if name not in clean_cache:
                    return activation
                return clean_cache[name].clone()

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name_post, resid_post_hook)]
        )

        effect = denoising_metric(
            patched_logits, clean_logits, corrupted_logits, answer_tokens,
            clean_input_length, corrupted_input_length, use_cnp
        )
        results['resid_post'][layer] = effect.item()

    return results
def path_patch_all_components2(
        model: HookedTransformer,
        clean_tokens: Tensor,
        corrupted_tokens: Tensor,
        answer_tokens: Tensor,
        clean_cache: dict,
        clean_input_length: Tensor = None,
        corrupted_input_length: Tensor = None,
        use_cnp: bool = False,
        quiet: bool = False
) -> Dict[str, Tensor]:
    """
    Path patch ALL circuit components with variable-length support
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = model.cfg.device

    # Compute input lengths if not provided
    if clean_input_length is None:
        clean_input_length = compute_input_lengths(clean_tokens)
    if corrupted_input_length is None:
        corrupted_input_length = compute_input_lengths(corrupted_tokens)

    # Get baselines
    clean_logits = model(clean_tokens)
    corrupted_logits = model(corrupted_tokens)

    results = {
        'attention_heads': torch.zeros((n_layers, n_heads), device=device),
        'mlp_layers': torch.zeros(n_layers, device=device),
        'attn_out': torch.zeros(n_layers, device=device),
        'mlp_out': torch.zeros(n_layers, device=device),
        'resid_pre': torch.zeros(n_layers, device=device),
        'resid_mid': torch.zeros(n_layers, device=device),
        'resid_post': torch.zeros(n_layers, device=device),
    }


    # 2. Patch MLP layers
    if not quiet:
        print("\n2. Patching MLP layers...")

    for layer in tqdm(range(n_layers), desc="MLP layers", disable=quiet):
        hook_name = get_hook_name('mlp_out', layer)

        def mlp_hook(activation, hook, name=hook_name):
            if name not in clean_cache:
                return activation
            return clean_cache[name].clone()

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, mlp_hook)]
        )

        effect = denoising_metric(
            patched_logits, clean_logits, corrupted_logits, answer_tokens,
            clean_input_length, corrupted_input_length, use_cnp
        )
        results['mlp_layers'][layer] = effect.item()
        results['mlp_out'][layer] = effect.item()

    # 3. Patch attention outputs
    if not quiet:
        print("\n3. Patching attention outputs...")

    for layer in tqdm(range(n_layers), desc="Attention outputs", disable=quiet):
        hook_name = get_hook_name('attn_out', layer)

        def attn_out_hook(activation, hook, name=hook_name):
            if name not in clean_cache:
                return activation
            return clean_cache[name].clone()

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, attn_out_hook)]
        )

        effect = denoising_metric(
            patched_logits, clean_logits, corrupted_logits, answer_tokens,
            clean_input_length, corrupted_input_length, use_cnp
        )
        results['attn_out'][layer] = effect.item()
    # 1. Patch attention heads
    if not quiet:
        print("\n1. Patching attention heads...")

    for layer in tqdm(range(n_layers), desc="Attention heads", disable=quiet):
        for head in range(n_heads):
            hook_name = get_hook_name('z', layer)

            def head_hook(activation, hook, head_idx=head, name=hook_name):
                if name not in clean_cache:
                    return activation
                clean_act = clean_cache[name]
                patched = activation.clone()
                patched[:, :, head_idx] = clean_act[:, :, head_idx]
                return patched

            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, head_hook)]
            )

            effect = denoising_metric(
                patched_logits, clean_logits, corrupted_logits, answer_tokens,
                clean_input_length, corrupted_input_length, use_cnp
            )
            results['attention_heads'][layer, head] = effect.item()
    # 4. Patch residual stream
    if not quiet:
        print("\n4. Patching residual stream...")

    for layer in tqdm(range(n_layers), desc="Residual stream", disable=quiet):
        # resid_pre
        hook_name_pre = get_hook_name('resid_pre', layer)

        def resid_pre_hook(activation, hook, name=hook_name_pre):
            if name not in clean_cache:
                return activation
            return clean_cache[name].clone()

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name_pre, resid_pre_hook)]
        )

        effect = denoising_metric(
            patched_logits, clean_logits, corrupted_logits, answer_tokens,
            clean_input_length, corrupted_input_length, use_cnp
        )
        results['resid_pre'][layer] = effect.item()

        # resid_mid (if exists)
        hook_name_mid = get_hook_name('resid_mid', layer)

        if hook_name_mid in clean_cache:
            def resid_mid_hook(activation, hook, name=hook_name_mid):
                if name not in clean_cache:
                    return activation
                return clean_cache[name].clone()

            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name_mid, resid_mid_hook)]
            )

            effect = denoising_metric(
                patched_logits, clean_logits, corrupted_logits, answer_tokens,
                clean_input_length, corrupted_input_length, use_cnp
            )
            results['resid_mid'][layer] = effect.item()

        # resid_post
        hook_name_post = get_hook_name('resid_post', layer)

        def resid_post_hook(activation, hook, name=hook_name_post):
            if name not in clean_cache:
                return activation
            return clean_cache[name].clone()

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name_post, resid_post_hook)]
        )

        effect = denoising_metric(
            patched_logits, clean_logits, corrupted_logits, answer_tokens,
            clean_input_length, corrupted_input_length, use_cnp
        )
        results['resid_post'][layer] = effect.item()

    return results

def path_patch_all_components1(
        model: HookedTransformer,
        clean_tokens: Tensor,
        corrupted_tokens: Tensor,
        answer_tokens: Tensor,
        clean_cache: dict,
        quiet: bool = False
) -> Dict[str, Tensor]:
    """
    Path patch ALL circuit components:
    - Attention heads (z)
    - MLP layers
    - Attention outputs (attn_out)
    - MLP outputs (mlp_out)
    - Residual stream (resid_pre, resid_post, resid_mid)

    Args:
        model: HookedTransformer model
        clean_tokens: Clean input tokens
        corrupted_tokens: Corrupted input tokens
        answer_tokens: Answer indices
        clean_cache: Cache from clean forward pass
        quiet: Suppress progress

    Returns:
        Dictionary with patching effects for each component type
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = model.cfg.device

    # Get baselines
    clean_logits = model(clean_tokens)
    corrupted_logits = model(corrupted_tokens)

    results = {
        'attention_heads': torch.zeros((n_layers, n_heads), device=device),  # Individual heads
        'mlp_layers': torch.zeros(n_layers, device=device),  # MLP layers
        'attn_out': torch.zeros(n_layers, device=device),  # Combined attention output
        'mlp_out': torch.zeros(n_layers, device=device),  # MLP output
        'resid_pre': torch.zeros(n_layers, device=device),  # Residual before layer
        'resid_mid': torch.zeros(n_layers, device=device),  # Residual after attention
        'resid_post': torch.zeros(n_layers, device=device),  # Residual after MLP
    }

    # 1. Patch attention heads (z) - individual heads
    if not quiet:
        print("\n1. Patching attention heads...")

    for layer in tqdm(range(n_layers), desc="Attention heads", disable=quiet):
        for head in range(n_heads):
            hook_name = get_hook_name('z', layer)

            def head_hook(activation, hook, head_idx=head, name=hook_name):
                if name not in clean_cache:
                    return activation
                clean_act = clean_cache[name]
                patched = activation.clone()
                patched[:, :, head_idx] = clean_act[:, :, head_idx]
                return patched

            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, head_hook)]
            )

            effect = denoising_metric(patched_logits, clean_logits, corrupted_logits, answer_tokens)
            results['attention_heads'][layer, head] = effect.item()

    # 2. Patch MLP layers - entire MLP at each layer
    if not quiet:
        print("\n2. Patching MLP layers...")

    for layer in tqdm(range(n_layers), desc="MLP layers", disable=quiet):
        hook_name = get_hook_name('mlp_out', layer)

        def mlp_hook(activation, hook, name=hook_name):
            if name not in clean_cache:
                return activation
            return clean_cache[name].clone()

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, mlp_hook)]
        )

        effect = denoising_metric(patched_logits, clean_logits, corrupted_logits, answer_tokens)
        results['mlp_layers'][layer] = effect.item()
        results['mlp_out'][layer] = effect.item()  # Same as mlp_layers

    # 3. Patch combined attention output (attn_out)
    if not quiet:
        print("\n3. Patching attention outputs...")

    for layer in tqdm(range(n_layers), desc="Attention outputs", disable=quiet):
        hook_name = get_hook_name('attn_out', layer)

        def attn_out_hook(activation, hook, name=hook_name):
            if name not in clean_cache:
                return activation
            return clean_cache[name].clone()

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, attn_out_hook)]
        )

        effect = denoising_metric(patched_logits, clean_logits, corrupted_logits, answer_tokens)
        results['attn_out'][layer] = effect.item()

    # 4. Patch residual stream
    if not quiet:
        print("\n4. Patching residual stream...")

    for layer in tqdm(range(n_layers), desc="Residual stream", disable=quiet):
        # resid_pre: before attention
        hook_name_pre = get_hook_name('resid_pre', layer)

        def resid_pre_hook(activation, hook, name=hook_name_pre):
            if name not in clean_cache:
                return activation
            return clean_cache[name].clone()

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name_pre, resid_pre_hook)]
        )

        effect = denoising_metric(patched_logits, clean_logits, corrupted_logits, answer_tokens)
        results['resid_pre'][layer] = effect.item()

        # resid_mid: after attention, before MLP
        hook_name_mid = get_hook_name('resid_mid', layer)

        # Check if hook exists in cache (some models may not have resid_mid)
        if hook_name_mid in clean_cache:
            def resid_mid_hook(activation, hook, name=hook_name_mid):
                if name not in clean_cache:
                    return activation
                return clean_cache[name].clone()

            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name_mid, resid_mid_hook)]
            )

            effect = denoising_metric(patched_logits, clean_logits, corrupted_logits, answer_tokens)
            results['resid_mid'][layer] = effect.item()

        # resid_post: after MLP
        hook_name_post = get_hook_name('resid_post', layer)

        def resid_post_hook(activation, hook, name=hook_name_post):
            if name not in clean_cache:
                return activation
            return clean_cache[name].clone()

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name_post, resid_post_hook)]
        )

        effect = denoising_metric(patched_logits, clean_logits, corrupted_logits, answer_tokens)
        results['resid_post'][layer] = effect.item()

    return results
def path_patch_all_components1(
    model: HookedTransformer,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    answer_tokens: Tensor,
    clean_cache: dict,
    quiet: bool = False
) -> Dict[str, Tensor]:
    """
    Path patch ALL circuit components:
    - Attention heads (z)
    - MLP layers
    - Attention outputs (attn_out)
    - MLP outputs (mlp_out)
    - Residual stream (resid_pre, resid_post, resid_mid)

    Args:
        model: HookedTransformer model
        clean_tokens: Clean input tokens
        corrupted_tokens: Corrupted input tokens
        answer_tokens: Answer indices
        clean_cache: Cache from clean forward pass
        quiet: Suppress progress

    Returns:
        Dictionary with patching effects for each component type
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = model.cfg.device

    # Get baselines
    clean_logits = model(clean_tokens)
    corrupted_logits = model(corrupted_tokens)

    results = {
        'attention_heads': torch.zeros((n_layers, n_heads), device=device),  # Individual heads
        'mlp_layers': torch.zeros(n_layers, device=device),                   # MLP layers
        'attn_out': torch.zeros(n_layers, device=device),                     # Combined attention output
        'mlp_out': torch.zeros(n_layers, device=device),                      # MLP output
        'resid_pre': torch.zeros(n_layers, device=device),                    # Residual before layer
        'resid_mid': torch.zeros(n_layers, device=device),                    # Residual after attention
        'resid_post': torch.zeros(n_layers, device=device),                   # Residual after MLP
    }

    # 1. Patch attention heads (z) - individual heads
    if not quiet:
        print("\n1. Patching attention heads...")

    for layer in tqdm(range(n_layers), desc="Attention heads", disable=quiet):
        for head in range(n_heads):
            hook_name = get_hook_name('z', layer)

            def head_hook(activation, hook, head_idx=head):
                if hook_name not in clean_cache:
                    return activation
                clean_act = clean_cache[hook_name]
                patched = activation.clone()
                patched[:, :, head_idx] = clean_act[:, :, head_idx]
                return patched

            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, head_hook)]
            )

            effect = denoising_metric(patched_logits, clean_logits, corrupted_logits, answer_tokens)
            results['attention_heads'][layer, head] = effect.item()

    # 2. Patch MLP layers - entire MLP at each layer
    if not quiet:
        print("\n2. Patching MLP layers...")

    for layer in tqdm(range(n_layers), desc="MLP layers", disable=quiet):
        # Try mlp_out first (post-activation), fallback to mlp_in
        hook_name = get_hook_name('mlp_out', layer)

        def mlp_hook(activation, hook):
            if hook_name not in clean_cache:
                return activation
            return clean_cache[hook_name].clone()

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, mlp_hook)]
        )

        effect = denoising_metric(patched_logits, clean_logits, corrupted_logits, answer_tokens)
        results['mlp_layers'][layer] = effect.item()
        results['mlp_out'][layer] = effect.item()  # Same as mlp_layers

    # 3. Patch combined attention output (attn_out)
    if not quiet:
        print("\n3. Patching attention outputs...")

    for layer in tqdm(range(n_layers), desc="Attention outputs", disable=quiet):
        hook_name = get_hook_name('attn_out', layer)

        def attn_out_hook(activation, hook):
            if hook_name not in clean_cache:
                return activation
            return clean_cache[hook_name].clone()

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, attn_out_hook)]
        )

        effect = denoising_metric(patched_logits, clean_logits, corrupted_logits, answer_tokens)
        results['attn_out'][layer] = effect.item()

    # 4. Patch residual stream
    if not quiet:
        print("\n4. Patching residual stream...")

    for layer in tqdm(range(n_layers), desc="Residual stream", disable=quiet):
        # resid_pre: before attention
        hook_name = get_hook_name('resid_pre', layer)

        def resid_hook(activation, hook):
            if hook_name not in clean_cache:
                return activation
            return clean_cache[hook_name].clone()

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, resid_hook)]
        )

        effect = denoising_metric(patched_logits, clean_logits, corrupted_logits, answer_tokens)
        results['resid_pre'][layer] = effect.item()


        # resid_post: after MLP
        hook_name = get_hook_name('resid_post', layer)

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, resid_hook)]
        )

        effect = denoising_metric(patched_logits, clean_logits, corrupted_logits, answer_tokens)
        results['resid_post'][layer] = effect.item()

    return results

def discover_complete_circuit(
    model: HookedTransformer,
    dataloader: DataLoader,
    top_k_per_component: int = 10,
    component_types: List[str] = None,
    test_faithfulness: bool = True,
    ablation_type: str = 'mean',
    quiet: bool = False
) -> dict:
    """
    Discover complete circuit including ALL components with faithfulness testing

    Computes TWO types of faithfulness:
    1. Simple:
       - Necessity: (clean - circuit_ablated) / (clean - corrupted_input)
       - Sufficiency: (circuit_only - fully_ablated) / (clean - fully_ablated)
    2. Paper-style recovery:
       - (circuit_only - fully_ablated) / (clean - fully_ablated)
       (This matches sufficiency when using circuit_only as circuit_perf.)

    Args:
        model: HookedTransformer model
        dataloader: DataLoader with (clean, corrupted, labels)
        top_k_per_component: Number of top components to keep per type
        component_types: Which components to include (default: all)
        test_faithfulness: Whether to test necessity and sufficiency
        ablation_type: 'zero' or 'mean' ablation for faithfulness tests
        quiet: Suppress progress

    Returns:
        Complete circuit with all component types and faithfulness metrics
    """
    if component_types is None:
        component_types = [
            'attention_heads',  # Individual attention heads
            'mlp_layers',       # MLP layers
            'attn_out',         # Combined attention outputs
            'mlp_out',          # MLP outputs
            'resid_post',       # Residual stream
        ]

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    print("=" * 60)
    print("COMPLETE CIRCUIT DISCOVERY")
    print("=" * 60)
    print(f"Components: {', '.join(component_types)}")
    print(f"Top-k per component: {top_k_per_component}")
    print(f"Test faithfulness: {test_faithfulness}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    clean_list, corrupted_list, labels_list = [], [], []
    i = 0
    for clean, corrupted, labels in dataloader:
        clean_list.extend(clean)
        corrupted_list.extend(corrupted)
        labels_list.append(labels)
        i += 1
        if i >= 1:  # Limit batches for speed; adjust as needed
            break

    clean_tokens = tokenize_prompts(model, clean_list)
    corrupted_tokens = tokenize_prompts(model, corrupted_list)
    answer_tokens = torch.cat(labels_list, dim=0).to(model.cfg.device)

    print(f"Clean shape: {clean_tokens.shape}")
    print(f"Corrupted shape: {corrupted_tokens.shape}")
    print(f"Sample clean tokens: {clean_tokens[0]}, {clean_list[0]}")
    print(f"Sample corrupted tokens: {corrupted_tokens[0]}, {corrupted_list[0]}")

    # Get clean cache
    print("\nRunning clean forward pass...")
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    clean_perf = compute_logit_diff(clean_logits, answer_tokens).mean().item()

    corrupted_logits = model(corrupted_tokens)
    corrupted_perf = compute_logit_diff(corrupted_logits, answer_tokens).mean().item()

    print(f"Clean performance: {clean_perf:.4f}")
    print(f"Corrupted performance: {corrupted_perf:.4f}")

    if abs(clean_perf - corrupted_perf) < 0.01:
        print("\n WARNING: Clean and corrupted have very similar performance!")
        print("   This suggests the model doesn't distinguish between them.")
        print("   Path patching will likely give zero effects.")

    # Patch all components
    print("\nPatching all components...")
    effects = path_patch_all_components(
        model, clean_tokens, corrupted_tokens, answer_tokens, clean_cache, quiet
    )

    # Display results for each component type
    print("\n" + "=" * 60)
    print("COMPONENT EFFECTS")
    print("=" * 60)

    circuit_components = []

    for component_type in component_types:
        if component_type not in effects:
            continue

        component_effects = effects[component_type]

        print(f"\n{component_type.upper().replace('_', ' ')}:")
        print(f"  Shape: {component_effects.shape}")
        print(f"  Max effect: {component_effects.max().item():.4f}")
        print(f"  Mean effect: {component_effects.mean().item():.4f}")

        # Get top-k for this component
        if component_type == 'attention_heads':
            # Flatten and get top-k heads
            flat = component_effects.flatten()
            top_k = min(top_k_per_component, flat.numel())
            top_indices = torch.topk(flat, k=top_k).indices

            print(f"\n  Top {top_k} heads:")
            for i, idx in enumerate(top_indices):
                layer = idx.item() // n_heads
                head = idx.item() % n_heads
                effect = component_effects[layer, head].item()
                print(f"    {i + 1}. L{layer}H{head}: {effect:.4f}")
                circuit_components.append({
                    'type': 'attention_head',
                    'layer': layer,
                    'head': head,
                    'effect': effect
                })
        else:
            # Get top-k layers for this component
            top_k = min(top_k_per_component, component_effects.numel())
            top_indices = torch.topk(component_effects, k=top_k).indices

            print(f"\n  Top {top_k} layers:")
            for i, idx in enumerate(top_indices):
                layer = idx.item()
                effect = component_effects[layer].item()
                print(f"    {i + 1}. Layer {layer}: {effect:.4f}")
                circuit_components.append({
                    'type': component_type,
                    'layer': layer,
                    'effect': effect
                })

    # Sort all components by effect
    circuit_components.sort(key=lambda x: abs(x['effect']), reverse=True)

    print("\n" + "=" * 60)
    print("COMPLETE CIRCUIT (Top components across all types)")
    print("=" * 60)

    top_overall = min(20, len(circuit_components))
    for i, comp in enumerate(circuit_components[:top_overall]):
        if comp['type'] == 'attention_head':
            print(f"  {i + 1}. L{comp['layer']}H{comp['head']} ({comp['type']}): {comp['effect']:.4f}")
        else:
            print(f"  {i + 1}. L{comp['layer']} ({comp['type']}): {comp['effect']:.4f}")

    # Faithfulness testing
    faithfulness_results = {}
    necessity_scores = []
    sufficiency_scores = []

    if test_faithfulness and len(circuit_components) > 0:
        from trace.circuits.faithfulness import (
            test_circuit_necessity,
            test_circuit_sufficiency,
            compute_faithfulness,
            create_all_components_list
        )

        print("\n" + "=" * 60)
        print("FAITHFULNESS TESTING")
        print("=" * 60)

        # Test necessity
        print("\nTesting necessity (ablating circuit from clean)...")
        circuit_ablated_perf, necessity_scores = test_circuit_necessity(
            model, clean_tokens, answer_tokens,
            circuit_components, ablation_type, quiet
        )
        print(f"  Performance with circuit ablated: {circuit_ablated_perf:.4f}")

        # Test sufficiency
        print("\nTesting sufficiency (keeping only circuit)...")
        all_components = create_all_components_list(model)
        circuit_only_perf, sufficiency_scores = test_circuit_sufficiency(
            model, clean_tokens, answer_tokens,
            circuit_components, all_components, ablation_type, quiet
        )

        # Fully ablated baseline is the first entry in sufficiency_scores
        fully_ablated_perf = sufficiency_scores[0] if len(sufficiency_scores) > 0 else None

        print(f"  Fully ablated performance:        {fully_ablated_perf:.4f}" if fully_ablated_perf is not None else
              "  Fully ablated performance:        N/A")
        print(f"  Performance with only circuit:    {circuit_only_perf:.4f}")

        # Compute faithfulness with consistent baselines
        # - Necessity uses corrupted INPUT as task-contrast baseline
        # - Sufficiency / paper recovery uses fully ablated baseline
        faithfulness_results = compute_faithfulness(
            clean_performance=clean_perf,
            corrupted_input_performance=corrupted_perf,
            circuit_ablated_performance=circuit_ablated_perf,
            circuit_only_performance=circuit_only_perf,
            fully_ablated_performance=fully_ablated_perf,
        )

        print("\n" + "=" * 60)
        print("FAITHFULNESS METRICS")
        print("=" * 60)

        # Method 1: Simple faithfulness
        print("\nMethod 1: Simple (Necessity & Sufficiency)")
        print(f"  Necessity:   {faithfulness_results['necessity_faithfulness']:.4f}")
        if faithfulness_results.get('sufficiency_faithfulness') is not None:
            print(f"  Sufficiency: {faithfulness_results['sufficiency_faithfulness']:.4f}")
        print(f"  Overall:     {faithfulness_results['overall_faithfulness']:.4f}")

        # Method 2: Paper / recovery faithfulness
        print("\nMethod 2: Paper / Recovery Formula")
        if faithfulness_results.get('paper_faithfulness') is not None and fully_ablated_perf is not None:
            print(f"  Paper Faithfulness: {faithfulness_results['paper_faithfulness']:.4f}")
            print(f"  Formula: (circuit_perf - ablated_perf) / (baseline_perf - ablated_perf)")
            print(f"         = ({circuit_only_perf:.4f} - {fully_ablated_perf:.4f}) / ({clean_perf:.4f} - {fully_ablated_perf:.4f})")
            print(f"         = {faithfulness_results['paper_faithfulness']:.4f}")
        else:
            print("  Paper Faithfulness: N/A (insufficient data)")

        print("=" * 60)

    return {
        'effects': effects,
        'circuit_components': circuit_components,
        'clean_performance': clean_perf,
        'corrupted_performance': corrupted_perf,
        'component_types': component_types,
        'faithfulness': faithfulness_results,
        'necessity_scores': necessity_scores,
        'sufficiency_scores': sufficiency_scores,
    }

def discover_complete_circuit1(
    model: HookedTransformer,
    dataloader: DataLoader,
    top_k_per_component: int = 10,
    component_types: List[str] = None,
    test_faithfulness: bool = True,
    ablation_type: str = 'mean',
    quiet: bool = False
) -> dict:
    """
    Discover complete circuit including ALL components with faithfulness testing

    Computes TWO types of faithfulness:
    1. Simple: Necessity and sufficiency
    2. Paper: (circuit_perf - ablated_perf) / (baseline_perf - ablated_perf)

    Args:
        model: HookedTransformer model
        dataloader: DataLoader with (clean, corrupted, labels)
        top_k_per_component: Number of top components to keep per type
        component_types: Which components to include (default: all)
        test_faithfulness: Whether to test necessity and sufficiency
        ablation_type: 'zero' or 'mean' ablation for faithfulness tests
        quiet: Suppress progress

    Returns:
        Complete circuit with all component types and BOTH faithfulness metrics
    """
    if component_types is None:
        component_types = [
            'attention_heads',  # Individual attention heads
            'mlp_layers',       # MLP layers
            'attn_out',         # Combined attention outputs
            'mlp_out',          # MLP outputs
            'resid_post',       # Residual stream
        ]

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    print("="*60)
    print("COMPLETE CIRCUIT DISCOVERY")
    print("="*60)
    print(f"Components: {', '.join(component_types)}")
    print(f"Top-k per component: {top_k_per_component}")
    print(f"Test faithfulness: {test_faithfulness}")
    print("="*60)

    # Load data
    print("\nLoading data...")
    clean_list, corrupted_list, labels_list = [], [], []
    i = 0
    for clean, corrupted, labels in dataloader:
        clean_list.extend(clean)
        corrupted_list.extend(corrupted)
        labels_list.append(labels)
        i += 1
        if i >= 1:  # Limit to 10 batches for speed; adjust as
            break

    clean_tokens = tokenize_prompts(model, clean_list)
    corrupted_tokens = tokenize_prompts(model, corrupted_list)
    answer_tokens = torch.cat(labels_list, dim=0).to(model.cfg.device)

    print(f"Clean shape: {clean_tokens.shape}")
    print(f"Corrupted shape: {corrupted_tokens.shape}")
    print(f"Sample clean tokens: {clean_tokens[0]}, {clean_list[0]}")
    print(f"Sample corrupted tokens: {corrupted_tokens[0], {corrupted_list[0]}}")

    # Get clean cache
    print("\nRunning clean forward pass...")
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    clean_perf = compute_logit_diff(clean_logits, answer_tokens).mean().item()

    corrupted_logits = model(corrupted_tokens)
    corrupted_perf = compute_logit_diff(corrupted_logits, answer_tokens).mean().item()

    print(f"Clean performance: {clean_perf:.4f}")
    print(f"Corrupted performance: {corrupted_perf:.4f}")

    if abs(clean_perf - corrupted_perf) < 0.01:
        print("\n WARNING: Clean and corrupted have very similar performance!")
        print("   This suggests the model doesn't distinguish between them.")
        print("   Path patching will likely give zero effects.")

    # Patch all components
    print("\nPatching all components...")
    effects = path_patch_all_components(
        model, clean_tokens, corrupted_tokens, answer_tokens, clean_cache, quiet
    )

    # Display results for each component type
    print("\n" + "="*60)
    print("COMPONENT EFFECTS")
    print("="*60)

    circuit_components = []

    for component_type in component_types:
        if component_type not in effects:
            continue

        component_effects = effects[component_type]

        print(f"\n{component_type.upper().replace('_', ' ')}:")
        print(f"  Shape: {component_effects.shape}")
        print(f"  Max effect: {component_effects.max().item():.4f}")
        print(f"  Mean effect: {component_effects.mean().item():.4f}")

        # Get top-k for this component
        if component_type == 'attention_heads':
            # Flatten and get top-k heads
            flat = component_effects.flatten()
            top_k = min(top_k_per_component, flat.numel())
            top_indices = torch.topk(flat, k=top_k).indices

            print(f"\n  Top {top_k} heads:")
            for i, idx in enumerate(top_indices):
                layer = idx.item() // n_heads
                head = idx.item() % n_heads
                effect = component_effects[layer, head].item()
                print(f"    {i+1}. L{layer}H{head}: {effect:.4f}")
                circuit_components.append({
                    'type': 'attention_head',
                    'layer': layer,
                    'head': head,
                    'effect': effect
                })
        else:
            # Get top-k layers for this component
            top_k = min(top_k_per_component, component_effects.numel())
            top_indices = torch.topk(component_effects, k=top_k).indices

            print(f"\n  Top {top_k} layers:")
            for i, idx in enumerate(top_indices):
                layer = idx.item()
                effect = component_effects[layer].item()
                print(f"    {i+1}. Layer {layer}: {effect:.4f}")
                circuit_components.append({
                    'type': component_type,
                    'layer': layer,
                    'effect': effect
                })

    # Sort all components by effect
    circuit_components.sort(key=lambda x: abs(x['effect']), reverse=True)

    print("\n" + "="*60)
    print("COMPLETE CIRCUIT (Top components across all types)")
    print("="*60)

    top_overall = min(20, len(circuit_components))
    for i, comp in enumerate(circuit_components[:top_overall]):
        if comp['type'] == 'attention_head':
            print(f"  {i+1}. L{comp['layer']}H{comp['head']} ({comp['type']}): {comp['effect']:.4f}")
        else:
            print(f"  {i+1}. L{comp['layer']} ({comp['type']}): {comp['effect']:.4f}")

    # Faithfulness testing
    faithfulness_results = {}
    necessity_scores = []
    sufficiency_scores = []

    if test_faithfulness and len(circuit_components) > 0:
        from trace.circuits.faithfulness import (
            test_circuit_necessity,
            test_circuit_sufficiency,
            compute_faithfulness,
            create_all_components_list
        )

        print("\n" + "="*60)
        print("FAITHFULNESS TESTING")
        print("="*60)

        # Test necessity
        print("\nTesting necessity (ablating circuit from clean)...")
        circuit_ablated_perf, necessity_scores = test_circuit_necessity(
            model, clean_tokens, answer_tokens,
            circuit_components, ablation_type, quiet
        )
        print(f"  Performance with circuit ablated: {circuit_ablated_perf:.4f}")

        # Test sufficiency
        print("\nTesting sufficiency (keeping only circuit)...")
        all_components = create_all_components_list(model)
        circuit_only_perf, sufficiency_scores = test_circuit_sufficiency(
            model, clean_tokens, answer_tokens,
            circuit_components, all_components, ablation_type, quiet
        )
        print(f"  Performance with only circuit: {circuit_only_perf:.4f}")

        # Compute BOTH faithfulness metrics
        faithfulness_results = compute_faithfulness(
            clean_performance=clean_perf,
            corrupted_performance=corrupted_perf,
            circuit_ablated_performance=circuit_ablated_perf,
            circuit_only_performance=circuit_only_perf,
            # Paper formula uses these (defaults to clean/corrupted if not provided)
            baseline_performance=clean_perf,
            ablated_performance=corrupted_perf,
            circuit_performance=circuit_only_perf
        )

        print("\n" + "="*60)
        print("FAITHFULNESS METRICS")
        print("="*60)

        # Method 1: Simple faithfulness
        print("\nMethod 1: Simple (Necessity & Sufficiency)")
        print(f"  Necessity:   {faithfulness_results['necessity_faithfulness']:.4f}")
        if faithfulness_results['sufficiency_faithfulness'] is not None:
            print(f"  Sufficiency: {faithfulness_results['sufficiency_faithfulness']:.4f}")
        print(f"  Overall:     {faithfulness_results['overall_faithfulness']:.4f}")

        # Method 2: Paper faithfulness
        print("\nMethod 2: Paper Formula")
        if faithfulness_results['paper_faithfulness'] is not None:
            print(f"  Paper Faithfulness: {faithfulness_results['paper_faithfulness']:.4f}")
            print(f"  Formula: (circuit_perf - ablated_perf) / (baseline_perf - ablated_perf)")
            print(f"         = ({circuit_only_perf:.4f} - {corrupted_perf:.4f}) / ({clean_perf:.4f} - {corrupted_perf:.4f})")
            print(f"         = {faithfulness_results['paper_faithfulness']:.4f}")
        else:
            print(f"  Paper Faithfulness: N/A (insufficient data)")

        print("="*60)

    return {
        'effects': effects,
        'circuit_components': circuit_components,
        'clean_performance': clean_perf,
        'corrupted_performance': corrupted_perf,
        'component_types': component_types,
        'faithfulness': faithfulness_results,
        'necessity_scores': necessity_scores,
        'sufficiency_scores': sufficiency_scores,
    }