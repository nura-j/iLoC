# trace/mechint/__init__.py

"""
MechInt: Unified Mechanistic Interpretability Package

This package combines EAP (Edge Attribution Patching) circuit discovery
with OWL storage and SPARQL querying capabilities.

Usage:
    # Basic EAP analysis
    >>> from trace.mechint import MechIntAnalyzer
    >>> analyzer = MechIntAnalyzer(
    ...     method='EAP-IG-inputs',
    ...     top_k=50,
    ...     cnp_dataset_path='./data/instrument_cnp.csv',
    ...     task_family='SemanticRoles',
    ...     role='INSTRUMENT'
    ... )
    >>> results, graph = analyzer.analyze(model, tokenizer=tokenizer)

    # With OWL export enabled (default)
    >>> analyzer = MechIntAnalyzer(
    ...     method='EAP-IG-inputs',
    ...     save_owl=True,  # default
    ...     owl_output_dir='./data/experiments'
    ... )
    >>> results, graph = analyzer.analyze(model, model_name="pythia-70m")
    >>> owl_path = results['owl_export_path']

    # Query stored circuits
    >>> from trace.mechint import CircuitQuery
    >>> query = CircuitQuery('./data/experiments')
    >>> query.load_experiments()
    >>> circuits = query.find_circuits_by_task('SemanticRoles')

    # One-liner convenience function
    >>> from trace.mechint import analyze_and_export
    >>> results, graph, owl_path = analyze_and_export(
    ...     model=model,
    ...     cnp_dataset_path='./data/instrument.csv',
    ...     task_family='SemanticRoles',
    ...     role='INSTRUMENT',
    ...     model_name='pythia-70m',
    ...     tokenizer=tokenizer
    ... )
"""

# Core analysis (imports are safe, no circular dependencies)
from .core.analyzer import MechIntAnalyzer
from .core.config import MechIntConfig

# Lazy-load storage and query to avoid circular imports
def _lazy_load_storage():
    """Lazy-load OWL exporter"""
    from trace.mechint.storage.owl_exporter import OWLExporter
    return OWLExporter

def _lazy_load_query():
    """Lazy-load Circuit query engine"""
    from trace.mechint.query.sparql_interface import CircuitQueryEngine
    return CircuitQueryEngine

def _lazy_load_ontology():
    """Lazy-load ontology schema"""
    from trace.mechint.ontology.schema import (
        MECHINT, MECHINT_BASE, MECHINT_SCHEMA, MechIntSchema
    )
    return MECHINT, MECHINT_BASE, MECHINT_SCHEMA, MechIntSchema

# Create lazy-loaded classes
class OWLExporter:
    """Lazy-loaded OWL Exporter"""
    _cls = None

    def __new__(cls, *args, **kwargs):
        if cls._cls is None:
            cls._cls = _lazy_load_storage()
        return cls._cls(*args, **kwargs)

class CircuitQuery:
    """Lazy-loaded Circuit Query Engine"""
    _cls = None

    def __new__(cls, *args, **kwargs):
        if cls._cls is None:
            cls._cls = _lazy_load_query()
        return cls._cls(*args, **kwargs)

# Convenience functions
from .api import (
    analyze_and_export,
    batch_analyze,
    create_analyzer,
    quick_export
)

# Pipeline functions
from .pipeline import (
    reconstruct_graph,
    run_cfs_pipeline,
    run_ilp_arch_pipeline,
    export_distance_report,
    compute_delta_t_preservation,
    analyse_dual_pool,
    compute_subsumption_matrix,
    cfs_clustering_test,
)

# Transfer engine
from .transfer import TransferEngine, TransferReport, CandidateResult

__all__ = [
    # Core analysis
    'MechIntAnalyzer',
    'MechIntConfig',

    # Storage & Query (lazy-loaded)
    'OWLExporter',
    'CircuitQuery',

    # API
    'analyze_and_export',
    'batch_analyze',
    'create_analyzer',
    'quick_export',

    # Pipeline
    'reconstruct_graph',
    'run_cfs_pipeline',
    'run_ilp_arch_pipeline',
    'export_distance_report',
    'compute_delta_t_preservation',
    'analyse_dual_pool',
    'compute_subsumption_matrix',
    'cfs_clustering_test',

    # Transfer
    'TransferEngine',
    'TransferReport',
    'CandidateResult',
]

__version__ = '0.2.0'
