

from torch import Tensor
from typing import List, Tuple, Dict, Optional
from transformer_lens import HookedTransformer
from tqdm import tqdm
from trace.circuits.utils import compute_logit_diff, get_hook_name, compute_cnp_metric


def create_component_ablation_hook(
        components_to_ablate: List[Dict],
        ablation_type: str = 'zero'
):
    """Create hook that ablates multiple component types"""
    # Organize by component type and layer
    ablation_map = {}
    for comp in components_to_ablate:
        comp_type = comp['type']
        layer = comp['layer']

        if comp_type not in ablation_map:
            ablation_map[comp_type] = {}

        if layer not in ablation_map[comp_type]:
            ablation_map[comp_type][layer] = []

        if comp_type == 'attention_head':
            ablation_map[comp_type][layer].append(comp['head'])

    def ablation_hook(activation, hook):
        """Hook that ablates specified components"""
        hook_name = hook.name

        for comp_type, layers in ablation_map.items():
            layer = hook.layer()

            if layer not in layers:
                continue

            if comp_type == 'attention_head' and 'attn.hook_z' in hook_name:
                heads_to_ablate = layers[layer]
                if ablation_type == 'zero':
                    activation[:, :, heads_to_ablate, :] = 0
                elif ablation_type == 'mean':
                    mean_act = activation.mean(dim=0, keepdim=True)
                    activation[:, :, heads_to_ablate, :] = mean_act[:, :, heads_to_ablate, :]

            elif comp_type == 'mlp_layers' and 'hook_mlp_out' in hook_name:
                if ablation_type == 'zero':
                    activation[:] = 0
                elif ablation_type == 'mean':
                    activation[:] = activation.mean(dim=0, keepdim=True)

            elif comp_type == 'attn_out' and 'hook_attn_out' in hook_name:
                if ablation_type == 'zero':
                    activation[:] = 0
                elif ablation_type == 'mean':
                    activation[:] = activation.mean(dim=0, keepdim=True)

            elif comp_type == 'mlp_out' and 'hook_mlp_out' in hook_name:
                if ablation_type == 'zero':
                    activation[:] = 0
                elif ablation_type == 'mean':
                    activation[:] = activation.mean(dim=0, keepdim=True)

            elif comp_type == 'resid_post' and 'hook_resid_post' in hook_name:
                if ablation_type == 'zero':
                    activation[:] = 0
                elif ablation_type == 'mean':
                    activation[:] = activation.mean(dim=0, keepdim=True)

        return activation

    return ablation_hook


def test_circuit_necessity(
        model: HookedTransformer,
        tokens: Tensor,
        answer_tokens: Tensor,
        circuit_components: List[Dict],
        ablation_type: str = 'mean',
        input_length: Tensor = None,
        use_cnp: bool = False,
        quiet: bool = False
) -> Tuple[float, List[float]]:
    """
    Test necessity: How much does performance drop when circuit is ablated?
    Now supports variable-length sequences
    """
    # Get clean baseline
    clean_logits = model(tokens)

    if use_cnp:
        if input_length is None:
            raise ValueError("input_length required for CNP metric")
        clean_score = compute_cnp_metric(clean_logits, answer_tokens, input_length, mean=True).item()
    else:
        clean_score = compute_logit_diff(clean_logits, answer_tokens, input_length, mean=True).item()

    scores = [clean_score]
    ablated_components = []

    iterator = tqdm(circuit_components, desc="Necessity test", disable=quiet)

    for comp in iterator:
        ablated_components.append(comp)

        # Get all unique layers that need hooks
        hook_names = set()
        for c in ablated_components:
            if c['type'] == 'attention_head':
                hook_names.add(get_hook_name('z', c['layer']))
            elif c['type'] == 'mlp_layers' or c['type'] == 'mlp_out':
                hook_names.add(get_hook_name('mlp_out', c['layer']))
            elif c['type'] == 'attn_out':
                hook_names.add(get_hook_name('attn_out', c['layer']))
            elif c['type'] == 'resid_post':
                hook_names.add(get_hook_name('resid_post', c['layer']))

        hook_filter = lambda name: any(h in name for h in hook_names)
        hook_fn = create_component_ablation_hook(ablated_components, ablation_type)

        ablated_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_filter, hook_fn)]
        )

        if use_cnp:
            score = compute_cnp_metric(ablated_logits, answer_tokens, input_length, mean=True).item()
        else:
            score = compute_logit_diff(ablated_logits, answer_tokens, input_length, mean=True).item()
        scores.append(score)

    return scores[-1], scores


def test_circuit_sufficiency(
        model: HookedTransformer,
        tokens: Tensor,
        answer_tokens: Tensor,
        circuit_components: List[Dict],
        all_components: List[Dict],
        ablation_type: str = 'mean',
        input_length: Tensor = None,
        use_cnp: bool = False,
        quiet: bool = False
) -> Tuple[float, List[float]]:
    """
    Test sufficiency: How much does performance recover with only circuit?
    Now supports variable-length sequences
    """
    # Start with all components ablated
    ablated_components = all_components.copy()

    # Get fully ablated baseline
    hook_names = set()
    for c in ablated_components:
        if c['type'] == 'attention_head':
            hook_names.add(get_hook_name('z', c['layer']))
        elif c['type'] in ['mlp_layers', 'mlp_out']:
            hook_names.add(get_hook_name('mlp_out', c['layer']))
        elif c['type'] == 'attn_out':
            hook_names.add(get_hook_name('attn_out', c['layer']))

    hook_filter = lambda name: any(h in name for h in hook_names)
    hook_fn = create_component_ablation_hook(ablated_components, ablation_type)

    fully_ablated_logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_filter, hook_fn)]
    )

    if use_cnp:
        if input_length is None:
            raise ValueError("input_length required for CNP metric")
        fully_ablated_score = compute_cnp_metric(fully_ablated_logits, answer_tokens, input_length, mean=True).item()
    else:
        fully_ablated_score = compute_logit_diff(fully_ablated_logits, answer_tokens, input_length, mean=True).item()

    scores = [fully_ablated_score]

    iterator = tqdm(reversed(circuit_components), desc="Sufficiency test", disable=quiet)

    for comp in iterator:
        # Remove this component from ablation list
        ablated_components = [c for c in ablated_components if not (
                c['type'] == comp['type'] and
                c['layer'] == comp['layer'] and
                c.get('head') == comp.get('head')
        )]

        if len(ablated_components) > 0:
            hook_names = set()
            for c in ablated_components:
                if c['type'] == 'attention_head':
                    hook_names.add(get_hook_name('z', c['layer']))
                elif c['type'] in ['mlp_layers', 'mlp_out']:
                    hook_names.add(get_hook_name('mlp_out', c['layer']))
                elif c['type'] == 'attn_out':
                    hook_names.add(get_hook_name('attn_out', c['layer']))

            hook_filter = lambda name: any(h in name for h in hook_names)
            hook_fn = create_component_ablation_hook(ablated_components, ablation_type)

            restored_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_filter, hook_fn)]
            )
        else:
            restored_logits = model(tokens)

        if use_cnp:
            score = compute_cnp_metric(restored_logits, answer_tokens, input_length, mean=True).item()
        else:
            score = compute_logit_diff(restored_logits, answer_tokens, input_length, mean=True).item()
        scores.append(score)

    return scores[-1], scores


def compute_faithfulness(
        clean_performance: float,
        corrupted_input_performance: float,
        circuit_ablated_performance: float,
        circuit_only_performance: Optional[float] = None,
        fully_ablated_performance: Optional[float] = None,
        eps: float = 1e-6,
        clamp: bool = False,
) -> Dict[str, Optional[float]]:
    """
    Compute faithfulness metrics
    """
    results: Dict[str, Optional[float]] = {
        "necessity_faithfulness": None,
        "sufficiency_faithfulness": None,
        "overall_faithfulness": None,
        "paper_faithfulness": None,
    }

    # Necessity
    denom_necessity = clean_performance - corrupted_input_performance
    if abs(denom_necessity) > eps:
        necessity = (clean_performance - circuit_ablated_performance) / denom_necessity
        if clamp:
            necessity = max(0.0, min(1.0, necessity))
        results["necessity_faithfulness"] = necessity
    else:
        results["necessity_faithfulness"] = 0.0

    # Sufficiency
    if circuit_only_performance is not None and fully_ablated_performance is not None:
        denom_sufficiency = clean_performance - fully_ablated_performance
        if abs(denom_sufficiency) > eps:
            sufficiency = (circuit_only_performance - fully_ablated_performance) / denom_sufficiency
            if clamp:
                sufficiency = max(0.0, min(1.0, sufficiency))
            results["sufficiency_faithfulness"] = sufficiency
            results["paper_faithfulness"] = sufficiency
        else:
            results["sufficiency_faithfulness"] = 0.0
            results["paper_faithfulness"] = 0.0

    necessity_val = results["necessity_faithfulness"]
    sufficiency_val = results["sufficiency_faithfulness"]
    if necessity_val is not None and sufficiency_val is not None:
        results["overall_faithfulness"] = 0.5 * (necessity_val + sufficiency_val)
    else:
        results["overall_faithfulness"] = necessity_val

    return results


def create_all_components_list(model: HookedTransformer) -> List[Dict]:
    """Create list of all components in model"""
    all_components = []

    for layer in range(model.cfg.n_layers):
        # Add all attention heads
        for head in range(model.cfg.n_heads):
            all_components.append({
                'type': 'attention_head',
                'layer': layer,
                'head': head
            })

        # Add MLP
        all_components.append({
            'type': 'mlp_layers',
            'layer': layer
        })

    return all_components