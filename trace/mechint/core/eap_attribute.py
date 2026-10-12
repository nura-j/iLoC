from typing import Callable, List, Optional, Literal, Tuple
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
import torch
from torch import Tensor
from typing import Callable, Optional
from functools import partial
from einops import einsum
from tqdm import tqdm
from .eap_utils import tokenize_plus, make_hooks_and_matrices, compute_mean_activations
from .eap_evaluate import evaluate_graph, evaluate_baseline
from .eap_graph import Graph


def get_scores_exact(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],
                     intervention: Literal['patching', 'zero', 'mean', 'mean-positional'] = 'patching',
                     intervention_dataloader: Optional[DataLoader] = None, quiet=False):
    """Gets scores via exact patching, by repeatedly calling evaluate graph.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        intervention (Literal[patching, zero, mean,mean, optional): the intervention to use. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader], optional): the dataloader over which to take the mean. Defaults to None.
        quiet (bool, optional): _description_. Defaults to False.
    """

    graph.in_graph |= graph.real_edge_mask  # All edges that are real are now in the graph
    baseline = evaluate_baseline(model, dataloader, metric).mean().item()
    edges = graph.edges.values() if quiet else tqdm(graph.edges.values())
    for edge in edges:
        edge.in_graph = False
        intervened_performance = evaluate_graph(model, graph, dataloader, metric, intervention=intervention,
                                                intervention_dataloader=intervention_dataloader,
                                                quiet=True, skip_clean=True).mean().item()
        edge.score = intervened_performance - baseline
        edge.in_graph = True

    # This is just to make the return type the same as all of the others; we've actually already updated the score matrix
    return graph.scores


def get_scores_eap(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],
                   intervention: Literal['patching', 'zero', 'mean', 'mean-positional'] = 'patching',
                   intervention_dataloader: Optional[DataLoader] = None, quiet=False,
                   track_last_token_only: bool = False):
    """Gets edge attribution scores using EAP.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=next(model.parameters()).device, dtype=model.cfg.dtype)

    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        # print('Processing the clean', clean)
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        # print('Processing the corrupted', corrupted)
        corrupted_tokens, corrupted_attention_mask, corrupted_input_lengths, n_pos_corrupted = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph,
                                                                                                           batch_size,
                                                                                                           n_pos,
                                                                                                           scores,
                                                                                                           input_lengths=input_lengths,
                                                                                                           track_last_token_only = track_last_token_only)

        with torch.inference_mode():
            if intervention == 'patching':
                # We intervene by subtracting out clean and adding in corrupted activations
                with model.hooks(fwd_hooks_corrupted):
                    # print(f'corrupted tokens shape: {corrupted_tokens.shape}')
                    if isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                      'model_type') and model.cfg.model_type == 'decoder_only':
                        # For decoder-only models, we need to do a full forward pass to get the correct causal attention mask
                        _ = model(tgt=corrupted_tokens, attention_mask=corrupted_attention_mask)
                    elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                        'model_type') and model.cfg.model_type == 'encoder_decoder':
                        # For encoder-decoder models, we need to pass in the encoder tokens to get the correct cross-attention mask
                        # Here, we just pass in the corrupted tokens as both encoder and decoder inputs
                        _ = model(enc=corrupted_tokens, tgt=corrupted_tokens, attention_mask=corrupted_attention_mask,
                                  enc_attention_mask=corrupted_attention_mask)
                    elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                        'model_type') and model.cfg.model_type == 'encoder_only':
                        _ = model(src=corrupted_tokens, attention_mask=corrupted_attention_mask)
                    else:
                        _ = model(corrupted_tokens, attention_mask=corrupted_attention_mask)
            elif 'mean' in intervention:
                # In the case of zero or mean ablation, we skip the adding in corrupted activations
                # but in mean ablations, we need to add the mean in
                activation_difference += means

            # For some metrics (e.g. accuracy or KL), we need the clean logits
            # print(f'clean tokens shape: {clean_tokens.shape}')
            if isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                              'model_type') and model.cfg.model_type == 'decoder_only':
                # For decoder-only models, we need to do a full forward pass to get the correct causal attention mask
                clean_logits = model(tgt=clean_tokens, attention_mask=attention_mask)
            elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                'model_type') and model.cfg.model_type == 'encoder_decoder':
                # For encoder-decoder models, we need to pass in the encoder tokens to get the correct cross-attention mask
                # Here, we just pass in the clean tokens as both encoder and decoder inputs
                clean_logits = model(enc=clean_tokens, tgt=clean_tokens, attention_mask=attention_mask,
                                     enc_attention_mask=attention_mask)
            elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                'model_type') and model.cfg.model_type == 'encoder_only':
                clean_logits = model(src=clean_tokens, attention_mask=attention_mask)
            else:
                clean_logits = model(clean_tokens, attention_mask=attention_mask)
        # if isinstance(model, HookedTransformer):
        #     pass
        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            # print(f'clean tokens shape (2): {clean_tokens.shape}')
            if isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                              'model_type') and model.cfg.model_type == 'decoder_only':
                # For decoder-only models, we need to do a full forward pass to get the correct causal attention mask
                logits = model(tgt=clean_tokens, attention_mask=attention_mask)
            elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                'model_type') and model.cfg.model_type == 'encoder_decoder':
                # For encoder-decoder models, we need to pass in the encoder tokens to get the correct cross-attention mask
                # Here, we just pass in the clean tokens as both encoder and decoder inputs
                logits = model(enc=clean_tokens, tgt=clean_tokens, attention_mask=attention_mask
                               , enc_attention_mask=attention_mask)
            elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                'model_type') and model.cfg.model_type == 'encoder_only':
                logits = model(src=clean_tokens, attention_mask=attention_mask)
            else:
                logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            # print(f'metric value: {metric_value}')
            metric_value.backward()

    scores /= total_items
    # print(f'scores device: {scores.device}, dtype: {scores.dtype}')
    print(f'Sample scores, max, min: {scores.flatten().max()}, {scores.flatten().min()}')
    return scores


def get_scores_eap_ig_vary_len(
    model,
    graph,
    dataloader,
    metric: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    steps: int = 30,
    quiet: bool = False,
):
    """
    EAP-IG variant that supports clean/corrupted pairs with different token lengths.

    Semantics (single-token metric):
      - Metric is evaluated at clean last valid token position (len_clean - 1).
      - Activation "baseline" for corrupted is taken from corrupted last valid token position (len_corr - 1).
      - Activation differences are placed ONLY at the clean last position (anchor), so einsum over positions remains valid.

    Returns:
      scores: [graph.n_forward, graph.n_backward]
    """

    device = model.cfg.device
    dtype = model.cfg.dtype
    d_model = model.cfg.d_model

    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=dtype)

    def forward_model(tokens: Tensor, attn_mask: Tensor) -> Tensor:
        """Match your existing model-type branching."""
        if isinstance(model, torch.nn.Module) and hasattr(model.cfg, "model_type"):
            if model.cfg.model_type == "decoder_only":
                return model(tgt=tokens, attention_mask=attn_mask)
            if model.cfg.model_type == "encoder_decoder":
                return model(enc=tokens, tgt=tokens, attention_mask=attn_mask, enc_attention_mask=attn_mask)
            if model.cfg.model_type == "encoder_only":
                return model(src=tokens, attention_mask=attn_mask)
        return model(tokens, attention_mask=attn_mask)

    def make_store_hooks(store: Tensor):
        """
        Store activations into:
          store[:, :, fwd_index, :] for all forward nodes.

        Shapes:
          - hook_embed: [B, T, d_model] -> fwd index 0
          - attn.hook_result: [B, T, n_heads, d_model] -> fwd slice for all heads at that layer
          - hook_mlp_out: [B, T, d_model] -> fwd index for mlp node
        """

        def store_input(activations, hook):
            store[:, :, graph.forward_index(graph.nodes["input"], attn_slice=False), :] = activations.detach()

        def store_attn(layer: int, activations, hook):
            # activations: [B, T, n_heads, d_model]
            node = graph.nodes[f"a{layer}.h0"]
            fwd_slice = graph.forward_index(node, attn_slice=True)  # slice over all heads
            store[:, :, fwd_slice, :] = activations.detach()

        def store_mlp(layer: int, activations, hook):
            node = graph.nodes[f"m{layer}"]
            fwd_idx = graph.forward_index(node, attn_slice=False)
            store[:, :, fwd_idx, :] = activations.detach()

        hooks = []
        hooks.append((graph.nodes["input"].out_hook, store_input))
        for layer in range(graph.cfg["n_layers"]):
            # Attention results
            hooks.append((graph.nodes[f"a{layer}.h0"].out_hook, partial(store_attn, layer)))
            # MLP outputs
            hooks.append((graph.nodes[f"m{layer}"].out_hook, partial(store_mlp, layer)))
        return hooks

    def make_bwd_hooks(activation_difference: Tensor, len_clean: Tensor):
        """
        Backward hooks accumulate scores using:
          einsum(activation_difference[:, :, :prev_index], grads) -> [forward, backward]
        with gradients masked to clean last-token positions.
        """

        def gradient_hook(prev_index, bwd_index, gradients, hook):
            grads = gradients.detach()

            # Ensure grads has shape [B, T, backward, d_model]
            # - For MLP in hook: typically [B, T, d_model] -> unsqueeze to [B, T, 1, d_model]
            # - For Q/K/V hooks: typically [B, T, n_heads, d_model] -> already OK
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)

            B, T = grads.shape[0], grads.shape[1]

            # Mask grads to clean last token position per example
            lengths = len_clean.to(device=grads.device).long()
            last_idx = (lengths - 1).clamp(min=0, max=T - 1)

            mask = torch.zeros((B, T), device=grads.device, dtype=grads.dtype)
            mask[torch.arange(B, device=grads.device), last_idx] = 1.0
            while mask.ndim < grads.ndim:
                mask = mask.unsqueeze(-1)
            grads = grads * mask

            s = einsum(
                activation_difference[:, :, :prev_index, :],
                grads,
                "batch pos forward hidden, batch pos backward hidden -> forward backward",
            )

            # If backward is singleton (MLP/logits), squeeze that dim
            s = s.squeeze(1)
            scores[:prev_index, bwd_index] += s

        bwd_hooks = []

        for layer in range(graph.cfg["n_layers"]):
            # Use a{layer}.h0 only as a handle for layer hook strings and slicing indices
            attn_node = graph.nodes[f"a{layer}.h0"]
            prev_index_attn = graph.prev_index(attn_node)

            for i, letter in enumerate("qkv"):
                bwd_idx = graph.backward_index(attn_node, qkv=letter, attn_slice=True)  # slice all heads
                bwd_hooks.append((attn_node.qkv_inputs[i], partial(gradient_hook, prev_index_attn, bwd_idx)))

            mlp_node = graph.nodes[f"m{layer}"]
            prev_index_mlp = graph.prev_index(mlp_node)
            bwd_idx_mlp = graph.backward_index(mlp_node, attn_slice=False)
            bwd_hooks.append((mlp_node.in_hook, partial(gradient_hook, prev_index_mlp, bwd_idx_mlp)))

        # logits in_hook is resid_post of last block; gradients shape [B, T, d_model]
        logit_node = graph.nodes["logits"]
        prev_index_logits = graph.prev_index(logit_node)
        bwd_idx_logits = graph.backward_index(logit_node, attn_slice=False)
        bwd_hooks.append((logit_node.in_hook, partial(gradient_hook, prev_index_logits, bwd_idx_logits)))
        return bwd_hooks

    total_items = 0
    iterator = dataloader if quiet else tqdm(dataloader)
    print(f"Using {steps} IG steps (vary-len, last-token anchored).")

    for clean, corrupted, label in iterator:
        # Tokenize separately (variable lengths are fine)
        clean_tokens, clean_mask, len_clean, n_pos_clean = tokenize_plus(model, clean)
        corr_tokens, corr_mask, len_corr, n_pos_corr = tokenize_plus(model, corrupted)

        B = clean_tokens.shape[0]
        T = max(n_pos_clean, n_pos_corr)

        # Pad tokens and masks to T
        if n_pos_clean < T:
            pad = T - n_pos_clean
            clean_tokens = torch.nn.functional.pad(clean_tokens, (0, pad), value=0)
            clean_mask = torch.nn.functional.pad(clean_mask, (0, pad), value=0)
        if n_pos_corr < T:
            pad = T - n_pos_corr
            corr_tokens = torch.nn.functional.pad(corr_tokens, (0, pad), value=0)
            corr_mask = torch.nn.functional.pad(corr_mask, (0, pad), value=0)

        total_items += B

        # Store activations for clean and corrupted
        acts_clean = torch.zeros((B, T, graph.n_forward, d_model), device=device, dtype=dtype)
        acts_corr = torch.zeros((B, T, graph.n_forward, d_model), device=device, dtype=dtype)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=make_store_hooks(acts_corr)):
                _ = forward_model(corr_tokens, corr_mask)

            with model.hooks(fwd_hooks=make_store_hooks(acts_clean)):
                clean_logits = forward_model(clean_tokens, clean_mask)

        # Anchor positions
        p_clean = (len_clean.to(device).long() - 1).clamp(min=0, max=T - 1)  # [B]
        p_corr = (len_corr.to(device).long() - 1).clamp(min=0, max=T - 1)    # [B]
        b_idx = torch.arange(B, device=device)

        # Build activation_difference: zeros everywhere except at clean last token positions
        activation_difference = torch.zeros_like(acts_clean)
        diff_last = acts_corr[b_idx, p_corr, :, :] - acts_clean[b_idx, p_clean, :, :]  # [B, n_forward, d_model]
        activation_difference[b_idx, p_clean, :, :] = diff_last

        # Backward hooks use activation_difference + gradient masking to clean last token
        bwd_hooks = make_bwd_hooks(activation_difference, len_clean)

        # Prepare constants for input interpolation at hook_embed
        fwd_input_idx = graph.forward_index(graph.nodes["input"], attn_slice=False)
        corr_embed_last = acts_corr[b_idx, p_corr, fwd_input_idx, :].detach()  # [B, d_model]

        # IG loop (k=1..steps includes endpoint alpha=1)
        for k in range(1, steps + 1):
            alpha = k / steps

            def embed_interp_hook(activations, hook):
                # activations: [B, T, d_model] (clean hook_embed output)
                out = activations.clone()
                clean_last = out[b_idx, p_clean, :]  # [B, d_model] (keeps grad path)
                out[b_idx, p_clean, :] = corr_embed_last + alpha * (clean_last - corr_embed_last)
                return out

            with model.hooks(
                fwd_hooks=[(graph.nodes["input"].out_hook, embed_interp_hook)],
                bwd_hooks=bwd_hooks,
            ):
                logits = forward_model(clean_tokens, clean_mask)
                metric_value = metric(logits, clean_logits, len_clean, label)

                if torch.isnan(metric_value).any().item():
                    raise ValueError("Metric value is NaN in get_scores_eap_ig_vary_len().")

                metric_value.backward()
                model.zero_grad(set_to_none=True)

        if torch.isnan(scores).any().item():
            raise ValueError("Scores became NaN in get_scores_eap_ig_vary_len().")

    # Normalize: dataset mean, then IG average
    if total_items > 0:
        scores /= total_items
    scores /= steps

    return scores


def get_scores_eap_ig(model: HookedTransformer, graph: Graph, dataloader: DataLoader,
                      metric: Callable[[Tensor], Tensor], steps=30, quiet=False,
                      track_last_token_only: bool = False):
    """Gets edge attribution scores using EAP with integrated gradients.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        steps (int, optional): number of IG steps. Defaults to 30.
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=next(model.parameters()).device, dtype=model.cfg.dtype)
    print(f"Using {steps} steps for IG")
    print('Tracking last token only:', track_last_token_only)
    # Fix the track last token only incase incorrectly set
    if isinstance(track_last_token_only, str):
        track_last_token_only = track_last_token_only.lower() in ['true', '1', 'yes']
    # print(f'scores device: {scores.device}, dtype: {scores.dtype}')

    total_items = 0
    total_backward_passes = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:

        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, corrupted_attention_mask, corrupted_input_lengths, n_pos_corrupted = tokenize_plus(model, corrupted)
        max_n_pos = max(n_pos, n_pos_corrupted)
        if n_pos != n_pos_corrupted: #and not track_last_token_only:
            print(f"Number of positions must match, but do not: {n_pos} (clean) != {n_pos_corrupted} (corrupted)")
            print(clean)
            print(corrupted)
            continue
        batch_size = len(clean)
        total_items += batch_size
            # raise ValueError("Number of positions must match")
        # Pad clean tokens and masks to max_n_pos if needed
        if n_pos < max_n_pos:
            pad_length = max_n_pos - n_pos
            clean_tokens = torch.nn.functional.pad(clean_tokens, (0, pad_length), value=0)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_length), value=0)

        # Pad corrupted tokens and masks to max_n_pos if needed
        if n_pos_corrupted < max_n_pos:
            pad_length = max_n_pos - n_pos_corrupted
            corrupted_tokens = torch.nn.functional.pad(corrupted_tokens, (0, pad_length), value=0)
            corrupted_attention_mask = torch.nn.functional.pad(corrupted_attention_mask, (0, pad_length), value=0)

        # Here, we get our fwd / bwd hooks and the activation difference matrix
        # The forward corrupted hooks add the corrupted activations to the activation difference matrix
        # The forward clean hooks subtract the clean activations
        # The backward hooks get the gradient, and use that, plus the activation difference, for the scores
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph,
                                                                                                           batch_size,
                                                                                                           max_n_pos,#n_pos,
                                                                                                           scores,
                                                                                                           input_lengths=input_lengths,
                                                                                                           track_last_token_only = track_last_token_only)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                if isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                  'model_type') and model.cfg.model_type == 'decoder_only':
                    # For decoder-only models, we need to do a full forward pass to get the correct causal attention mask
                    _ = model(tgt=corrupted_tokens, attention_mask=corrupted_attention_mask)
                elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                    'model_type') and model.cfg.model_type == 'encoder_decoder':
                    # For encoder-decoder models, we need to pass in the encoder tokens to get the correct cross-attention mask
                    # Here, we just pass in the corrupted tokens as both encoder and decoder inputs
                    _ = model(enc=corrupted_tokens, tgt=corrupted_tokens, attention_mask=corrupted_attention_mask,
                              enc_attention_mask=corrupted_attention_mask)
                elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                    'model_type') and model.cfg.model_type == 'encoder_only':
                    _ = model(src=corrupted_tokens, attention_mask=corrupted_attention_mask)
                else:
                    _ = model(corrupted_tokens, attention_mask=corrupted_attention_mask)

            input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                if isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                  'model_type') and model.cfg.model_type == 'decoder_only':
                    # For decoder-only models, we need to do a full forward pass to get the correct causal attention mask
                    clean_logits = model(tgt=clean_tokens, attention_mask=attention_mask)
                elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                    'model_type') and model.cfg.model_type == 'encoder_decoder':
                    # For encoder-decoder models, we need to pass in the encoder tokens to get the correct cross-attention mask
                    # Here, we just pass in the clean tokens as both encoder and decoder inputs
                    clean_logits = model(enc=clean_tokens, tgt=clean_tokens, attention_mask=attention_mask,
                                         enc_attention_mask=attention_mask)
                elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                    'model_type') and model.cfg.model_type == 'encoder_only':
                    clean_logits = model(src=clean_tokens, attention_mask=attention_mask)
                else:
                    clean_logits = model(clean_tokens, attention_mask=attention_mask)

            input_activations_clean = input_activations_corrupted - activation_difference[:, :,
                                                                    graph.forward_index(graph.nodes['input'])]

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (
                            input_activations_clean - input_activations_corrupted)
                new_input.requires_grad = True
                return new_input

            return hook_fn

        total_steps = 0
        for step in range(0, steps):
            total_steps += 1
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))],
                             bwd_hooks=bwd_hooks):
                if isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                  'model_type') and model.cfg.model_type == 'decoder_only':
                    # For decoder-only models, we need to do a full forward pass to get the correct causal attention mask
                    logits = model(tgt=clean_tokens, attention_mask=attention_mask)
                elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                    'model_type') and model.cfg.model_type == 'encoder_decoder':
                    # For encoder-decoder models, we need to pass in the encoder tokens to get the correct cross-attention mask
                    # Here, we just pass in the clean tokens as both encoder and decoder inputs
                    logits = model(enc=clean_tokens, tgt=clean_tokens, attention_mask=attention_mask,
                                   enc_attention_mask=attention_mask)
                elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                    'model_type') and model.cfg.model_type == 'encoder_only':
                    logits = model(src=clean_tokens, attention_mask=attention_mask)
                else:
                    # logits = model(input_ids=clean_tokens, attention_mask=attention_mask)
                    logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                if torch.isnan(metric_value).any().item():
                    print("Metric value is NaN")
                    print(f"Clean: {clean}")
                    print(f"Corrupted: {corrupted}")
                    print(f"Label: {label}")
                    print(f"Metric: {metric}")
                    raise ValueError("Metric value is NaN")
                metric_value.backward()

            if torch.isnan(scores).any().item():
                print("Metric value is NaN")
                print(f"Clean: {clean}")
                print(f"Corrupted: {corrupted}")
                print(f"Label: {label}")
                print(f"Metric: {metric}")
                print(f'Step: {step}')
                raise ValueError("Metric value is NaN")

    scores /= total_items
    scores /= total_steps

    return scores


def get_scores_ig_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader,
                              metric: Callable[[Tensor], Tensor],
                              intervention: Literal['patching', 'zero', 'mean', 'mean-positional'] = 'patching',
                              steps=30, intervention_dataloader: Optional[DataLoader] = None, quiet=False,
                              track_last_token_only: bool = False):
    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    scores = torch.zeros((graph.n_forward, graph.n_backward), device=next(model.parameters()).device, dtype=model.cfg.dtype)

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(model, graph, batch_size, n_pos,
                                                                                     scores)
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        if intervention == 'patching':
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                if isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                  'model_type') and model.cfg.model_type == 'decoder_only':
                    # For decoder-only models, we need to do a full forward pass to get the correct causal attention mask
                    _ = model(tgt=corrupted_tokens, attention_mask=attention_mask)
                elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                    'model_type') and model.cfg.model_type == 'encoder_decoder':
                    # For encoder-decoder models, we need to pass in the encoder tokens to get the correct cross-attention mask
                    # Here, we just pass in the corrupted tokens as both encoder and decoder inputs
                    _ = model(enc=corrupted_tokens, tgt=corrupted_tokens, attention_mask=attention_mask,
                              enc_attention_mask=attention_mask)
                elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                    'model_type') and model.cfg.model_type == 'encoder_only':
                    _ = model(src=corrupted_tokens, attention_mask=attention_mask)
                else:
                    _ = model(corrupted_tokens, attention_mask=attention_mask)

        elif 'mean' in intervention:
            activation_difference += means

        with model.hooks(fwd_hooks=fwd_hooks_clean):
            if isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                              'model_type') and model.cfg.model_type == 'decoder_only':
                # For decoder-only models, we need to do a full forward pass to get the correct causal attention mask
                clean_logits = model(tgt=clean_tokens, attention_mask=attention_mask)
            elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                'model_type') and model.cfg.model_type == 'encoder_decoder':
                # For encoder-decoder models, we need to pass in the encoder tokens to get the correct cross-attention mask
                # Here, we just pass in the clean tokens as both encoder and decoder inputs
                clean_logits = model(enc=clean_tokens, tgt=clean_tokens, attention_mask=attention_mask,
                                     enc_attention_mask=attention_mask)
            elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                'model_type') and model.cfg.model_type == 'encoder_only':
                clean_logits = model(src=clean_tokens, attention_mask=attention_mask)
            else:
                clean_logits = model(clean_tokens, attention_mask=attention_mask)
            activation_difference += activations_corrupted.clone().detach() - activations_clean.clone().detach()

        def output_interpolation_hook(k: int, clean: torch.Tensor, corrupted: torch.Tensor):
            def hook_fn(activations: torch.Tensor, hook):
                alpha = k / steps
                new_output = alpha * clean + (1 - alpha) * corrupted
                return new_output

            return hook_fn

        total_steps = 0

        nodeslist = [graph.nodes['input']]
        for layer in range(graph.cfg['n_layers']):
            nodeslist.append(graph.nodes[f'a{layer}.h0'])
            nodeslist.append(graph.nodes[f'm{layer}'])

        for node in nodeslist:
            for step in range(1, steps + 1):
                total_steps += 1

                clean_acts = activations_clean[:, :, graph.forward_index(node)]
                corrupted_acts = activations_corrupted[:, :, graph.forward_index(node)]
                fwd_hooks = [(node.out_hook, output_interpolation_hook(step, clean_acts, corrupted_acts))]

                with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                    if isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                      'model_type') and model.cfg.model_type == 'decoder_only':
                        # For decoder-only models, we need to do a full forward pass to get the correct causal attention mask
                        logits = model(tgt=clean_tokens, attention_mask=attention_mask)
                    elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                        'model_type') and model.cfg.model_type == 'encoder_decoder':
                        # For encoder-decoder models, we need to pass in the encoder tokens to get the correct cross-attention mask
                        # Here, we just pass in the clean tokens as both encoder and decoder inputs
                        logits = model(enc=clean_tokens, tgt=clean_tokens, attention_mask=attention_mask,
                                       enc_attention_mask=attention_mask)
                    elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                        'model_type') and model.cfg.model_type == 'encoder_only':
                        logits = model(src=clean_tokens, attention_mask=attention_mask)
                    else:
                        logits = model(clean_tokens, attention_mask=attention_mask)
                    metric_value = metric(logits, clean_logits, input_lengths, label)

                    metric_value.backward(retain_graph=True)

    scores /= total_items
    scores /= total_steps

    return scores


def get_scores_clean_corrupted(model: HookedTransformer, graph: Graph, dataloader: DataLoader,
                               metric: Callable[[Tensor], Tensor], quiet=False,
                               track_last_token_only: bool = False):
    """Gets scores using the clean-corrupted method: like EAP-IG, but just do it on the clean and corrupted inputs, instead of all the intermediate steps.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        quiet (bool, optional): whether to silence tqdm. Defaults to False.

    Returns:
        _type_: _description_
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=next(model.parameters()).device, dtype=model.cfg.dtype)

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph,
                                                                                                           batch_size,
                                                                                                           n_pos,
                                                                                                           scores)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                if isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                  'model_type') and model.cfg.model_type == 'decoder_only':
                    # For decoder-only models, we need to do a full forward pass to get the correct causal attention mask
                    _ = model(tgt=corrupted_tokens, attention_mask=attention_mask)
                elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                    'model_type') and model.cfg.model_type == 'encoder_decoder':
                    # For encoder-decoder models, we need to pass in the encoder tokens to get the correct cross-attention mask
                    # Here, we just pass in the corrupted tokens as both encoder and decoder inputs
                    _ = model(enc=corrupted_tokens, tgt=corrupted_tokens, attention_mask=attention_mask,
                              enc_attention_mask=attention_mask)
                elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                    'model_type') and model.cfg.model_type == 'encoder_only':
                    _ = model(src=corrupted_tokens, attention_mask=attention_mask)
                else:
                    _ = model(corrupted_tokens, attention_mask=attention_mask)

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

        total_steps = 2
        with model.hooks(bwd_hooks=bwd_hooks):
            if isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                              'model_type') and model.cfg.model_type == 'decoder_only':
                # For decoder-only models, we need to do a full forward pass to get the correct causal attention mask
                logits = model(tgt=clean_tokens, attention_mask=attention_mask)
            elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                'model_type') and model.cfg.model_type == 'encoder_decoder':
                # For encoder-decoder models, we need to pass in the encoder tokens to get the correct cross-attention mask
                # Here, we just pass in the clean tokens as both encoder and decoder inputs
                logits = model(enc=clean_tokens, tgt=clean_tokens, attention_mask=attention_mask,
                               enc_attention_mask=attention_mask)
            elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                'model_type') and model.cfg.model_type == 'encoder_only':
                logits = model(src=clean_tokens, attention_mask=attention_mask)
            else:
                logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()
            model.zero_grad()

            corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask)
            corrupted_metric_value = metric(corrupted_logits, clean_logits, input_lengths, label)
            corrupted_metric_value.backward()
            model.zero_grad()

    scores /= total_items
    scores /= total_steps

    return scores


def get_scores_information_flow_routes(model: HookedTransformer, graph: Graph, dataloader: DataLoader,
                                       quiet=False,
                                       track_last_token_only: bool = False) -> torch.Tensor:
    """Gets scores using Ferrando et al.'s (2024) information flow routes method.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        quiet (bool, optional): whether to silence tqdm. Defaults to False.

    Returns:
        Tensor: scores based on information flow routes
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=next(model.parameters()).device, dtype=model.cfg.dtype)

    def make_hooks(n_pos: int, input_lengths: torch.Tensor) -> List[Tuple[str, Callable]]:
        output_activations = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model),
                                         device=model.cfg.device, dtype=model.cfg.dtype)

        def output_hook(index, activations, hook):
            try:
                acts = activations.detach()
                output_activations[:, :, index] = acts
            except RuntimeError as e:
                print(hook.name, output_activations[:, :, index].size(), output_activations.size())
                raise e

        # compute the score directly, without saving the input activations
        def input_hook(prev_index, bwd_index, input_lengths, activations, hook):
            acts = activations.detach()
            try:
                if acts.ndim == 3:
                    acts = acts.unsqueeze(2)
                # acts : batch pos backward hidden
                # output acts: batch pos forward hidden
                # add forward and backwards dimensions to acts and output acts respectively
                acts = acts.unsqueeze(2)
                unsqueezed_output_activations = output_activations.unsqueeze(3)

                # acts : batch pos 1 backward hidden
                # output acts: batch pos forward 1 hidden
                proximity = torch.clamp(
                    - torch.linalg.vector_norm(unsqueezed_output_activations[:, :, :prev_index] - acts, ord=1,
                                               dim=-1) + torch.linalg.vector_norm(acts, ord=1, dim=-1), min=0)
                importance = proximity / torch.sum(proximity, dim=2, keepdim=True)
                # importance: batch pos forward backward
                # aggregate over positions via sum/mean to get importance: forward backward
                # first mask out importances for padding positions
                max_len = input_lengths.max()
                mask = torch.arange(max_len, device=input_lengths.device,
                                    dtype=input_lengths.dtype).expand(len(input_lengths),
                                                                      max_len) < input_lengths.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                # print(importance.size(), mask.size())
                importance *= mask
                importance = importance.sum(1) / input_lengths.view(-1, 1, 1)  # mean over positions
                importance = importance.sum(0)

                # importance: forward backward
                # squeezing backward dim in case it isn't real (i.e. it's an MLP)
                importance = importance.squeeze(1)
                scores[:prev_index, bwd_index] += importance

            except RuntimeError as e:
                print(hook.name, unsqueezed_output_activations[:, :, prev_index].size(), acts.size())
                raise e

        hooks = []
        node = graph.nodes['input']
        fwd_index = graph.forward_index(node)
        hooks.append((node.out_hook, partial(output_hook, fwd_index)))

        for layer in range(graph.cfg['n_layers']):
            node = graph.nodes[f'a{layer}.h0']
            fwd_index = graph.forward_index(node)
            hooks.append((node.out_hook, partial(output_hook, fwd_index)))
            prev_index = graph.prev_index(node)
            for i, letter in enumerate('qkv'):
                bwd_index = graph.backward_index(node, qkv=letter)
                hooks.append((node.qkv_inputs[i], partial(input_hook, prev_index, bwd_index, input_lengths)))

            node = graph.nodes[f'm{layer}']
            fwd_index = graph.forward_index(node)
            bwd_index = graph.backward_index(node)
            prev_index = graph.prev_index(node)
            hooks.append((node.out_hook, partial(output_hook, fwd_index)))
            hooks.append((node.in_hook, partial(input_hook, prev_index, bwd_index, input_lengths)))

        node = graph.nodes['logits']
        prev_index = graph.prev_index(node)
        bwd_index = graph.backward_index(node)
        hooks.append((node.in_hook, partial(input_hook, prev_index, bwd_index, input_lengths)))
        return hooks

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, _, _ in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)

        hooks = make_hooks(n_pos, input_lengths)
        with torch.inference_mode():
            with model.hooks(fwd_hooks=hooks):
                if isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                  'model_type') and model.cfg.model_type == 'decoder_only':
                    # For decoder-only models, we need to do a full forward pass to get the correct causal attention mask
                    _ = model(tgt=clean_tokens, attention_mask=attention_mask)
                elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                    'model_type') and model.cfg.model_type == 'encoder_decoder':
                    # For encoder-decoder models, we need to pass in the encoder tokens to get the correct cross-attention mask
                    # Here, we just pass in the clean tokens as both encoder and decoder inputs
                    _ = model(enc=clean_tokens, tgt=clean_tokens, attention_mask=attention_mask,
                              enc_attention_mask=attention_mask)
                elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                    'model_type') and model.cfg.model_type == 'encoder_only':
                    _ = model(src=clean_tokens, attention_mask=attention_mask)
                else:
                    _ = model(clean_tokens, attention_mask=attention_mask)

    scores /= total_items

    return scores


allowed_aggregations = {'sum', 'mean'}


def attribute(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],
              method: Literal[
                  'EAP', 'EAP-IG-inputs', 'clean-corrupted', 'EAP-IG-activations', 'information-flow-routes', 'exact'],
              intervention: Literal['patching', 'zero', 'mean', 'mean-positional'] = 'patching', aggregation='sum',
              ig_steps: Optional[int] = None, intervention_dataloader: Optional[DataLoader] = None, quiet=False,
              track_last_token_only: bool = False):
    print('Starting attribution...')
    assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    assert model.cfg.use_split_qkv_input, "Model must be configured to use split qkv inputs (model.cfg.use_split_qkv_input)"
    assert model.cfg.use_hook_mlp_in, "Model must be configured to use hook MLP in (model.cfg.use_hook_mlp_in)"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"

    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')

    # Scores are by default summed across the d_model dimension
    # This means that scores are a [n_src_nodes, n_dst_nodes] tensor
    if method == 'EAP':
        scores = get_scores_eap(model, graph, dataloader, metric, intervention=intervention,
                                intervention_dataloader=intervention_dataloader, quiet=quiet,
                                track_last_token_only=track_last_token_only)
    elif method == 'EAP-IG-inputs':
        if intervention != 'patching':
            raise ValueError(f"intervention must be 'patching' for EAP-IG-inputs, but got {intervention}")
        print('Using EAP-IG on inputs')
        scores = get_scores_eap_ig(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet,
                                   track_last_token_only=track_last_token_only)
    elif method == 'EAP-IG-inputs-vary-length':
        if intervention != 'patching':
            raise ValueError(f"intervention must be 'patching' for EAP-IG-inputs-vary-length, but got {intervention}")
        print('Using EAP-IG on inputs with varying lengths')
        scores = get_scores_eap_ig_vary_len(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet)
    elif method == 'clean-corrupted':
        if intervention != 'patching':
            raise ValueError(f"intervention must be 'patching' for clean-corrupted, but got {intervention}")
        scores = get_scores_clean_corrupted(model, graph, dataloader, metric, quiet=quiet)
    elif method == 'EAP-IG-activations':
        scores = get_scores_ig_activations(model, graph, dataloader, metric, steps=ig_steps, intervention=intervention,
                                           intervention_dataloader=intervention_dataloader, quiet=quiet)
    elif method == 'information-flow-routes':
        scores = get_scores_information_flow_routes(model, graph, dataloader, quiet=quiet)
    elif method == 'exact':
        scores = get_scores_exact(model, graph, dataloader, metric, intervention=intervention,
                                  intervention_dataloader=intervention_dataloader,
                                  quiet=quiet)
    elif method == 'path-patching':
        from trace.circuits import get_scores_path_patching
        scores = get_scores_path_patching(
            model, graph, dataloader, metric,
            intervention='denoising',
            quiet=quiet,
            track_last_token_only=track_last_token_only
        )

    elif method == 'path-patching-vary-length':
        from trace.circuits import get_scores_path_patching_vary_len
        scores = get_scores_path_patching_vary_len(
            model, graph, dataloader, metric,
            intervention='denoising',
            quiet=quiet
        )
    elif method == 'path-patching-standalone':
        # Import wrapper
        pass
    else:
        raise ValueError(
            f"method must be in ['EAP', 'EAP-IG-inputs', 'clean-corrupted', 'EAP-IG-activations', 'information-flow-routes', 'exact'], but got {method}")

    print(
        f'Scores computed, shape: {scores.shape}, min: {scores.min().item():.6f}, max: {scores.max().item():.6f}, mean: {scores.mean().item():.6f}')

    if aggregation == 'mean':
        scores /= model.cfg.d_model
    print(f'Attribution method: {method}, intervention: {intervention}, aggregation: {aggregation}')
    print(
        f'Scores shape: {scores.shape}, min: {scores.min().item():.6f}, max: {scores.max().item():.6f}, mean: {scores.mean().item():.6f}')
    graph.scores[:] = scores.to(graph.scores.device)
    print('Attribution complete, with scores stored in graph.scores', graph.scores.shape)
