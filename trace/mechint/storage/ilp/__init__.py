"""
ILP Learning Pipeline

Modules for Inductive Logic Programming to learn semantic specifications.
"""

from .predicate_extraction import PredicateExtractor
from .ilp_arch import ArchitecturalILPLearner

# Lazy imports — ilp_sem requires problog which may not be installed.
# ilp_integration depends on ilp_sem, so it is also deferred.
try:
    from .ilp_sem import SemanticSpecificationLearner
    from .ilp_integration import ILPIntegration, process_single_circuit
    _has_problog = True
except ImportError:
    _has_problog = False

try:
    from .adapters.csv_adapter import CSVTaskAdapter, load_task_examples, prepare_ilp_data
    _has_csv_adapter = True
except ImportError:
    _has_csv_adapter = False

__all__ = [
    'PredicateExtractor',
    'SemanticSpecificationLearner',
    'ArchitecturalILPLearner',
    'ILPIntegration',
    'process_single_circuit'
]

if _has_csv_adapter:
    __all__.extend([
        'CSVTaskAdapter',
        'load_task_examples',
        'prepare_ilp_data'
    ])