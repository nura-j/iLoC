# trace/mechint/core/analyzer.py

"""
MechInt Analyzer - Unified EAP + OWL Export
"""

import os
from functools import partial
from typing import Dict, Any, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .config import MechIntConfig

from .validation_statistics import (
    validate_epsilon_minimality,
    permutation_test,
    compute_validation_statistics
)

class MechIntAnalyzer:
    """
    Unified mechanistic interpretability analyzer

    Combines:
    - EAP circuit discovery
    - Automatic OWL export to knowledge graph
    - Metadata tracking for experiments
    """

    def __init__(
        self,
        config: Optional[MechIntConfig] = None,
        method: str = 'EAP-IG-inputs',
        top_k: int = 50,
        ig_steps: int = 5,
        cnp_dataset_path: Optional[str] = None,
        save_owl: bool = True,
        owl_output_dir: str = "./data/experiments",
        task_family: Optional[str] = None,
        role: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize MechInt analyzer

        Args:
            config: MechIntConfig object (if provided, other args are ignored)
            method: EAP method
            top_k: Number of top edges
            ig_steps: IG steps
            cnp_dataset_path: Path to CNP dataset
            save_owl: Enable OWL export
            owl_output_dir: OWL output directory
            task_family: Task family name
            role: Role/phenomenon name
            **kwargs: Additional config parameters
        """
        # Create config if not provided
        if config is None:
            config = MechIntConfig(
                method=method,
                top_k=top_k,
                ig_steps=ig_steps,
                cnp_dataset_path=cnp_dataset_path,
                save_owl=save_owl,
                owl_output_dir=owl_output_dir,
                task_family=task_family,
                role=role,
                **kwargs
            )

        # Store config
        self.config = config
        self.device = self.config.device

        # Store MechInt-specific settings
        self.save_owl = config.save_owl
        self.owl_output_dir = config.owl_output_dir
        self.task_family = config.task_family
        self.role = config.role

        # Storage for CNP dataset
        self.cnp_dataset = None

        # Lazy-load OWL exporter
        self._owl_exporter = None

        if self.save_owl:
            print(f"[MechInt] OWL export enabled → {self.owl_output_dir}")

    def load_cnp_dataset(self, dataset_path, tokenizer):
        """Load CNP dataset from CSV or JSON"""

        class EAPDataset(Dataset):
            def __init__(self, filepath):
                if filepath.endswith('.csv'):
                    self.df = pd.read_csv(filepath)
                elif filepath.endswith('.json'):
                    import json
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    self.df = pd.DataFrame(data)
                else:
                    raise ValueError(f"Unsupported file format: {filepath}")

                # Validate required columns
                required_cols = ['clean', 'corrupted', 'correct_idx', 'incorrect_idx']
                missing_cols = [col for col in required_cols if col not in self.df.columns]
                if missing_cols:
                    raise ValueError(f"Dataset missing required columns: {missing_cols}")

            def __len__(self):
                return len(self.df)

            def __getitem__(self, index):
                row = self.df.iloc[index]
                return row['clean'], row['corrupted'], [row['correct_idx'], row['incorrect_idx']]

            def to_dataloader(self, batch_size):
                def collate_fn(xs):
                    clean, corrupted, labels = zip(*xs)
                    return list(clean), list(corrupted), torch.tensor(labels)

                return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn)

        return EAPDataset(dataset_path)

    @property
    def owl_exporter(self):
        """Lazy-load OWL exporter"""
        if self._owl_exporter is None and self.save_owl:
            from trace.mechint.storage.owl_exporter import OWLExporter
            self._owl_exporter = OWLExporter(output_dir=self.owl_output_dir)
        return self._owl_exporter

    def analyze(
        self,
        model,
        batch_loader=None,
        tokenizer=None,
        model_name: str = "",
        step: int = 0,
        experiment_id: Optional[str] = None,
        task_family: Optional[str] = None,
        role: Optional[str] = None,
    ) -> tuple[Dict[str, Any], Optional['Graph']]:
        """
        Run EAP analysis with optional OWL export

        Args:
            model: Transformer model
            batch_loader: Data loader (optional)
            tokenizer: Tokenizer
            model_name: Model identifier
            step: Training/checkpoint step
            experiment_id: Experiment ID (auto-generated if None)
            task_family: Override config task_family
            role: Override config role


        Returns:
            Tuple of (results_dict, graph)
            - results_dict contains 'owl_export_path' if export succeeded
        """
        from .eap_graph import Graph
        from .eap_attribute import attribute
        from .eap_evaluate import evaluate_graph, evaluate_baseline
        from .eap_cnp import cnp_metric

        # Use instance settings if not provided
        if task_family is None:
            task_family = self.task_family
        if role is None:
            role = self.role

        # Generate experiment ID if needed
        if experiment_id is None:
            experiment_id = self._generate_experiment_id(
                model_name, task_family, role
            )

        print(f"\n[MechInt] Running analysis...")
        print(f"[MechInt]   Model: {model_name}")
        print(f"[MechInt]   Task: {task_family}/{role}")
        print(f"[MechInt]   Step: {step}")
        print(f"[MechInt]   Experiment: {experiment_id}")

        # Load CNP dataset
        if not self.config.cnp_dataset_path:
            print("[MechInt] ✗ No CNP dataset path provided")
            return {}, None

        self.cnp_dataset = self.load_cnp_dataset(self.config.cnp_dataset_path, tokenizer)
        if len(self.cnp_dataset) == 0:
            print("[MechInt] ✗ CNP dataset is empty")
            return {}, None

        eap_dataloader = self.cnp_dataset.to_dataloader(self.config.eap_batch_size or 10)

        # Create computational graph
        graph = Graph.from_model(model)

        # Run EAP attribution
        attribute(
            model, graph, eap_dataloader,
            partial(cnp_metric, loss=True, mean=True),
            method=self.config.method,
            ig_steps=self.config.ig_steps,
            track_last_token_only=self.config.track_last_token_only
        )

        # Apply top-k selection
        graph.apply_topn(self.config.top_k, True)

        # Evaluate performance
        baseline_performance = evaluate_baseline(
            model, eap_dataloader, partial(cnp_metric, loss=False, mean=False),
            track_last_token_only=self.config.track_last_token_only
        ).mean().item()

        circuit_performance = evaluate_graph(
            model, graph, eap_dataloader, partial(cnp_metric, loss=False, mean=False),
            track_last_token_only=self.config.track_last_token_only
        ).mean().item()

        # Ablated performance
        empty_graph = Graph.from_model(model)
        empty_graph.reset(empty=True)
        ablated_performance = evaluate_graph(
            model, empty_graph, eap_dataloader, partial(cnp_metric, loss=False, mean=False),
            skip_clean=False,
            track_last_token_only=self.config.track_last_token_only
        ).mean().item()

        # Calculate faithfulness
        if baseline_performance != ablated_performance:
            faithfulness = (circuit_performance - ablated_performance) / \
                          (baseline_performance - ablated_performance)
        else:
            faithfulness = 0.0
            print("[MechInt] Warning: baseline == ablated, cannot compute faithfulness")

        faithfulness_simple = circuit_performance / baseline_performance \
            if baseline_performance != 0 else 0.0

        # k-sweep: faithfulness at multiple edge budgets. Attribution is
        # k-independent and top-k circuits are nested, so this costs one
        # extra evaluation per budget, not a new discovery run.
        k_sweep_results = {}
        _sweep = sorted(set(self.config.k_sweep or []))
        for _k in _sweep:
            if _k == self.config.top_k:
                _perf_k = circuit_performance
            else:
                graph.apply_topn(_k, True)
                _perf_k = evaluate_graph(
                    model, graph, eap_dataloader,
                    partial(cnp_metric, loss=False, mean=False),
                    track_last_token_only=self.config.track_last_token_only
                ).mean().item()
            _faith_k = ((_perf_k - ablated_performance) /
                        (baseline_performance - ablated_performance)
                        if baseline_performance != ablated_performance else 0.0)
            k_sweep_results[str(_k)] = {
                "circuit_performance": _perf_k,
                "faithfulness": _faith_k,
            }
        if _sweep:
            graph.apply_topn(self.config.top_k, True)
            print(f"[MechInt]   k-sweep faithfulness: "
                  + ", ".join(f"k={k}: {v['faithfulness']:.3f}"
                              for k, v in k_sweep_results.items()))

        # Results dictionary
        results = {
            "method": self.config.method,
            "top_k": self.config.top_k,
            "baseline_performance": baseline_performance,
            "circuit_performance": circuit_performance,
            "ablated_performance": ablated_performance,
            "faithfulness": faithfulness,
            "faithfulness_simple": faithfulness_simple,
            **({"k_sweep": k_sweep_results} if k_sweep_results else {}),
            "num_edges_in_circuit": sum(1 for edge in graph.edges.values() if edge.in_graph),
            "num_nodes_in_circuit": sum(1 for node in graph.nodes.values() if node.in_graph),
            "performance_drop": baseline_performance - circuit_performance,
            "performance_drop_pct": (baseline_performance - circuit_performance) / baseline_performance * 100
            if baseline_performance != 0 else 0,
        }
        if self.config.compute_validation:
            print("\n" + "=" * 60)
            print("Computing Validation Statistics (κ)")
            print("=" * 60)

            validation_stats = compute_validation_statistics(
                graph=graph,
                model=model,
                dataloader=eap_dataloader,
                metric_fn=partial(cnp_metric, loss=False, mean=False),
                baseline_performance=baseline_performance,
                epsilon=self.config.epsilon,
                n_permutations=self.config.n_permutations,
                track_last_token_only=self.config.track_last_token_only,
                verbose=self.config.validation_verbose
            )
            results.update({
                "epsilon_minimal": validation_stats['epsilon_minimal'],
                "permutation_pvalue": validation_stats['permutation_pvalue'],
                "statistically_significant": validation_stats['statistically_significant'],
            })
            if self.config.log_dir:
                validation_path = os.path.join(
                    self.config.log_dir,
                    f"validation_{model_name}_step_{step}.json"
                )
                with open(validation_path, 'w') as f:
                    import json
                    json.dump({
                        'epsilon_minimal': validation_stats['epsilon_minimal'],
                        'epsilon_details': {
                            k: v for k, v in validation_stats['epsilon_minimal_details'].items()
                            if k != 'non_essential_edges'  # Skip for brevity
                        },
                        'permutation_pvalue': validation_stats['permutation_pvalue'],
                        'permutation_details': {
                            k: v for k, v in validation_stats['permutation_test_details'].items()
                            if k != 'null_distribution'  # Skip large array
                        },
                    }, f, indent=2)
                print(f"Validation details saved to: {validation_path}")

        print(f"\n{'=' * 60}")
        print(f"MechInt Analysis Results (Step {step})")
        print(f"{'=' * 60}")
        print(f"Baseline (full model):  {baseline_performance:.4f}")
        print(f"Circuit (pruned):       {circuit_performance:.4f}")
        print(f"Ablated (empty):        {ablated_performance:.4f}")
        print(f"Faithfulness:           {faithfulness:.4f}")
        print(f"Circuit edges:          {results['num_edges_in_circuit']}")
        print(f"Circuit nodes:          {results['num_nodes_in_circuit']}")
        print(f"{'=' * 60}\n")

        # Save graphs if requested
        if self.config.save_graphs and self.config.log_dir:
            self._save_graphs(graph, model_name, step)

        if graph is None:
            print("[MechInt] ✗ No graph produced, skipping OWL export")
            return results, graph

        # Export to OWL if enabled
        if self.save_owl and self.owl_exporter:
            try:
                print(f"\n[MechInt] Exporting to OWL...")

                # Collect metadata
                metadata = self._collect_metadata(model, model_name)

                # Extract config
                config_dict = self._extract_config_dict()

                # Export
                owl_path = self.owl_exporter.export_circuit(
                    graph=graph,
                    experiment_id=experiment_id,
                    step=step,
                    config=config_dict,
                    metadata=metadata,
                    eap_results=results, # should include epsilon_minimal, permutation_pvalue
                    task_family=task_family,
                    role_or_phenomenon=role,
                )

                results["owl_export_path"] = owl_path
                print(f"[MechInt] ✓ Exported to: {owl_path}")

            except Exception as e:
                print(f"[MechInt] ✗ OWL export failed: {e}")
                import traceback
                traceback.print_exc()
                results["owl_export_error"] = str(e)

        return results, graph

    def _generate_experiment_id(
        self,
        model_name: str,
        task_family: Optional[str],
        role: Optional[str]
    ) -> str:
        """Generate experiment ID from model and task"""
        parts = []

        if task_family:
            parts.append(task_family.lower())
        if role:
            parts.append(role.lower())
        if model_name:
            short_name = model_name.split("/")[-1].replace("_", "").replace("-", "")
            parts.append(short_name)

        if not parts:
            parts = ["experiment"]

        return "_".join(parts)

    def _collect_metadata(self, model, model_name: str) -> Dict[str, Any]:
        """Collect model metadata"""
        metadata = {"model_name": model_name}

        # Extract from model config
        if hasattr(model, 'cfg'):
            cfg = model.cfg
            metadata.update({
                "n_layers": getattr(cfg, 'n_layers', None),
                "d_model": getattr(cfg, 'd_model', None),
                "n_heads": getattr(cfg, 'n_heads', None),
                "model_type": getattr(cfg, 'model_type', None),
            })
        elif hasattr(model, 'config'):
            config = model.config
            metadata.update({
                "n_layers": getattr(config, 'num_hidden_layers', None),
                "d_model": getattr(config, 'hidden_size', None),
                "n_heads": getattr(config, 'num_attention_heads', None),
                "model_type": getattr(config, 'model_type', None),
            })

        return metadata

    def _extract_config_dict(self) -> Dict[str, Any]:
        """Extract configuration as dictionary"""
        return {
            "method": self.config.method,
            "ig_steps": self.config.ig_steps,
            "top_k": self.config.top_k,
        }

    def _save_graphs(self, graph, model_name: str, step: int):
        """Save EAP graphs"""
        graph_dir = os.path.join(self.config.log_dir, 'eap_graphs')
        os.makedirs(graph_dir, exist_ok=True)

        if self.config.save_json:
            json_path = os.path.join(graph_dir, f'{model_name}_graph_step_{step}.json')
            graph.to_json(json_path)

        if self.config.save_pt:
            pt_path = os.path.join(graph_dir, f'{model_name}_graph_step_{step}.pt')
            graph.to_pt(pt_path)

    def add_learned_specification(
        self,
        owl_path: str,
        spec_text: str,
        predicates: str,
        confidence: float,
        ilp_signature: Dict[str, Any],
    ) -> str:
        """
        Add ILP-learned specification to circuit

        Args:
            owl_path: Path to circuit OWL file
            spec_text: FOL clause text
            predicates: Comma-separated predicates
            confidence: ILP confidence score
            ilp_signature: Dict with n_clauses, avg_depth, etc.

        Returns:
            Updated OWL file path
        """
        if not self.owl_exporter:
            raise ValueError("OWL exporter not enabled")

        return self.owl_exporter.add_learned_specification(
            circuit_ttl_path=owl_path,
            spec_text=spec_text,
            predicates=predicates,
            confidence=confidence,
            ilp_signature=ilp_signature,
        )
