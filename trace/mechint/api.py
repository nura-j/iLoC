"""
Convenience API functions for common workflows
"""

from typing import Dict, Any, Optional, List
from .core.analyzer import MechIntAnalyzer


def create_analyzer(
    method: str = 'EAP-IG-inputs',
    top_k: int = 50,
    ig_steps: int = 5,
    cnp_dataset_path: Optional[str] = None,
    task_family: Optional[str] = None,
    role: Optional[str] = None,
    save_owl: bool = True,
    owl_output_dir: str = "./data/experiments",
    **kwargs
) -> MechIntAnalyzer:
    """
    Create a MechInt analyzer with sensible defaults

    Args:
        method: EAP method ('EAP-IG-inputs', 'EAP-IG-logits', etc.)
        top_k: Number of top edges to keep
        ig_steps: Number of integrated gradient steps
        cnp_dataset_path: Path to counterfactual dataset
        task_family: Task family (e.g., 'SemanticRoles', 'IOI')
        role: Role/phenomenon (e.g., 'INSTRUMENT', 'NameMover')
        save_owl: Enable OWL export
        owl_output_dir: Directory for OWL exports
        **kwargs: Additional config parameters

    Returns:
        MechIntAnalyzer instance = results, graph

    Example:
        >>> analyzer = create_analyzer(
        ...     cnp_dataset_path='./data/instrument.csv',
        ...     task_family='SemanticRoles',
        ...     role='INSTRUMENT',
        ...     top_k=50
        ... )
        >>> results, graph = analyzer.analyze(model, tokenizer=tokenizer)
    """
    return MechIntAnalyzer(
        method=method,
        top_k=top_k,
        ig_steps=ig_steps,
        cnp_dataset_path=cnp_dataset_path,
        task_family=task_family,
        role=role,
        save_owl=save_owl,
        owl_output_dir=owl_output_dir,
        **kwargs
    )


def analyze_and_export(
    model,
    cnp_dataset_path: str,
    task_family: str,
    role: str,
    model_name: str,
    tokenizer=None,
    step: int = 0,
    method: str = 'EAP-IG-inputs',
    top_k: int = 50,
    ig_steps: int = 5,
    compute_validation: bool = False,
    output_dir: str = "./data/experiments",
    operation_type: Optional[str] = None,   # κ from L(C) = <sigma, kappa, phi, tau>
    logical_category: Optional[str] = None, # τ: "PREDICATE" | "ROLE" | "OPERATOR"
    experiment_id: Optional[str] = None,    # unique OWL/TTL name (auto if None)
    eap_batch_size: int = 10,
    k_sweep=None,
) -> tuple[Dict[str, Any], Optional[Any], Optional[str]]:
    """
    One-line function: analyze circuit + export to OWL.

    Args:
        model: Transformer model
        cnp_dataset_path: Path to counterfactual dataset
        task_family: Task family name
        role: Role/phenomenon name
        model_name: Model identifier
        tokenizer: Tokenizer (optional)
        step: Training step
        method: EAP method
        top_k: Number of top edges
        ig_steps: Integrated gradient steps (default 5)
        compute_validation: Compute epsilon-minimality + permutation test kappa statistics
            (default False; set True for final paper runs — adds ~1200 evaluate_graph calls)
        output_dir: Output directory
        operation_type: kappa - operation type from L(C) (e.g. "binding", "selection")
        logical_category: tau - logical category from L(C) ("PREDICATE"|"ROLE"|"OPERATOR")

    Returns:
        Tuple of (results, graph, owl_path)

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
        >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        >>>
        >>> results, graph, owl_path = analyze_and_export(
        ...     model=model,
        ...     cnp_dataset_path="./data/instrument_cnp.csv",
        ...     task_family="SemanticRoles",
        ...     role="INSTRUMENT",
        ...     model_name="pythia-70m",
        ...     tokenizer=tokenizer
        ... )
        >>> print(f"Exported to: {owl_path}")
    """
    analyzer = create_analyzer(
        method=method,
        top_k=top_k,
        ig_steps=ig_steps,
        cnp_dataset_path=cnp_dataset_path,
        task_family=task_family,
        role=role,
        save_owl=True,
        owl_output_dir=output_dir,
        compute_validation=compute_validation,
        eap_batch_size=eap_batch_size,
        k_sweep=k_sweep,
    )

    results, graph = analyzer.analyze(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        step=step,
        experiment_id=experiment_id,
        task_family=task_family,
        role=role,
    )

    # κ/τ: analyzer.analyze() doesn't accept extra metadata, so we patch the
    # OWL file after the fact; κ/τ are also stored in the results JSON
    # (export_task_json call in run_experiment.py).
    if (operation_type or logical_category) and results.get("owl_export_path"):
        try:
            from trace.mechint.storage.owl_exporter import OWLExporter as _OE
            _oe = _OE()
            _oe.patch_kappa_tau(
                results["owl_export_path"],
                operation_type=operation_type,
                logical_category=logical_category,
            )
        except Exception as e:
            print(f"[api] warning: could not patch κ/τ into TTL: {e} "
                  f"(κ/τ still stored in results JSON)")

    owl_path = results.get("owl_export_path")
    return results, graph, owl_path


def batch_analyze(
    model,
    experiments: List[Dict[str, Any]],
    tokenizer=None,
    method: str = 'EAP-IG-inputs',
    top_k: int = 50,
    output_dir: str = "./data/experiments",
) -> Dict[str, tuple]:
    """
    Batch analyze multiple circuits with automatic OWL export

    Args:
        model: Transformer model
        experiments: List of experiment dicts with keys:
            - cnp_dataset_path: Path to CNP dataset
            - task_family: Task family name
            - role: Role/phenomenon name
            - step: (optional) Training step
        tokenizer: Tokenizer
        method: EAP method
        top_k: Top-k edges
        output_dir: Output directory

    Returns:
        Dict mapping experiment_id → (results, graph, owl_path)

    Example:
        >>> experiments = [
        ...     {
        ...         'cnp_dataset_path': './data/instrument.csv',
        ...         'task_family': 'SemanticRoles',
        ...         'role': 'INSTRUMENT',
        ...         'step': 0
        ...     },
        ...     {
        ...         'cnp_dataset_path': './data/beneficiary.csv',
        ...         'task_family': 'SemanticRoles',
        ...         'role': 'BENEFICIARY',
        ...         'step': 0
        ...     }
        ... ]
        >>>
        >>> results_dict = batch_analyze(
        ...     model=model,
        ...     experiments=experiments,
        ...     tokenizer=tokenizer
        ... )
    """
    all_results = {}

    for exp_config in experiments:
        cnp_path = exp_config['cnp_dataset_path']
        task_family = exp_config['task_family']
        role = exp_config['role']
        step = exp_config.get('step', 0)
        model_name = exp_config.get('model_name', '')

        print(f"\n{'='*60}")
        print(f"Processing: {task_family}/{role}")
        print(f"{'='*60}")

        results, graph, owl_path = analyze_and_export(
            model=model,
            cnp_dataset_path=cnp_path,
            task_family=task_family,
            role=role,
            model_name=model_name,
            tokenizer=tokenizer,
            step=step,
            method=method,
            top_k=top_k,
            output_dir=output_dir,
        )

        # Generate experiment ID for key
        exp_id = f"{task_family.lower()}_{role.lower()}"
        all_results[exp_id] = (results, graph, owl_path)

    return all_results


def quick_export(
    graph: Any,  # Graph object from trace.eap.graph
    experiment_id: str,
    task_family: str,
    role: str,
    model_name: str = "",
    step: int = 0,
    output_dir: str = "./data/experiments",
    operation_type: Optional[str] = None,   # κ
    logical_category: Optional[str] = None, # τ
    **metadata
) -> str:
    """
    Quick export an existing graph to OWL (without running EAP)

    Args:
        graph: EAP Graph object
        experiment_id: Experiment identifier
        task_family: Task family name
        role: Role/phenomenon name
        model_name: Model name
        step: Training step
        output_dir: Output directory
        **metadata: Additional metadata

    Returns:
        Path to exported OWL file

    Example:
        >>> # If you already have a graph from EAP
        >>> owl_path = quick_export(
        ...     graph=my_graph,
        ...     experiment_id='semanticroles_instrument',
        ...     task_family='SemanticRoles',
        ...     role='INSTRUMENT',
        ...     model_name='pythia-70m'
        ... )
    """
    from trace.mechint.storage.owl_exporter import OWLExporter

    exporter = OWLExporter(output_dir=output_dir)

    config = {
        "method": metadata.get("method", "EAP"),
        "top_k": metadata.get("top_k", 50),
        "ig_steps": metadata.get("ig_steps", 5),
    }

    meta = {
        "model_name": model_name,
        **metadata
    }

    # Inject κ/τ into metadata so _add_provenance picks them up via its
    # metadata.get() fallback — works with both old and new exporter versions.
    if operation_type:
        meta["operation_type"]   = operation_type
    if logical_category:
        meta["logical_category"] = logical_category

    owl_path = exporter.export_circuit(
        graph=graph,
        experiment_id=experiment_id,
        step=step,
        config=config,
        metadata=meta,
        eap_results={},
        task_family=task_family,
        role_or_phenomenon=role,
        operation_type=operation_type,
        logical_category=logical_category,
    )

    return owl_path


__all__ = [
    'create_analyzer',
    'analyze_and_export',
    'batch_analyze',
    'quick_export',
]
