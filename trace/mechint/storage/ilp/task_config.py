"""
Task Configuration for ILP Learning

Defines everything task-specific as data, not code:
- Predicate vocabulary
- Hypothesis templates
- Role mapping (if applicable)
- Background knowledge (if applicable)

"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set


@dataclass
class TaskConfig:
    """
    Complete task specification for ILP learning.

    Defines everything the pipeline needs to know about a task
    without any task-specific code in the core modules.

    Attributes:
        task_name: task identifier.

        target_predicate: The predicate ILP learns (head of clauses).

        predicate_vocabulary: Task-specific predicates and their metadata (Paper Table 2.2).
            Each entry: predicate_name -> {arity, args, [admissible_universe]}.


        hypothesis_templates: Candidate rule templates using {target} and
            {role_label} placeholders rendered at learning time.

        role_label: Maps target predicate to a role label (e.g. 'TIME').
            None for tasks without semantic roles.

        background_knowledge: Extra Prolog rules appended to fact database.

        example_key_mapping: Maps generic keys to task-specific CSV/dict keys.
            Generic keys: 'entity', 'entity_idx', 'label', 'correct',
            'prompt', 'value', 'reference', 'distractor'.

        admissible_universes: Predicate -> list of admissible values for CWA.
    """

    task_name: str
    target_predicate: str
    predicate_vocabulary: Dict[str, Dict]
    hypothesis_templates: List[str]
    role_label: Optional[str] = None
    background_knowledge: List[str] = field(default_factory=list)
    example_key_mapping: Dict[str, str] = field(default_factory=dict)
    admissible_universes: Dict[str, List[str]] = field(default_factory=dict)
    # Gap 8: formal mode declarations constraining the hypothesis language (§3.6, Appendix D)
    # Format follows ILASP / Metagol convention:
    #   modeh(1, target_pred(+type, -type))  — head modes
    #   modeb(*, body_pred(+type, -type))    — body modes
    # "+" = input (bound), "-" = output (generated), "#" = constant
    mode_declarations: List[str] = field(default_factory=list)

    # Label predicates: predicates used only to construct the pos/neg example split.
    # They are TRUE BY CONSTRUCTION for every positive example and FALSE for every
    # negative, so any ILP learner finds them immediately as trivial separators —
    # learning nothing about the circuit.
    #
    # These are EXCLUDED from:
    #   1. The ILP fact database passed to _generate_hypotheses
    #   2. Any hypothesis whose body consists solely of label predicates
    #
    # For semantic roles: {'role'}  (role(E,'LOCATION') just echoes the dataset label)
    # For IOI / GT:       set()     (no label predicate in their fact programs)
    # For any future task: declare whichever predicate encodes the class membership label
    label_predicates: Set[str] = field(default_factory=set)

    def get_mode_declarations(self) -> List[str]:
        """
        Gap 8: Return mode declarations constraining the ILP hypothesis language.

        Mode declarations specify which variable bindings are allowed in clause
        heads and bodies, bounding the search space (§3.6, Appendix D).

        Returns self.mode_declarations if explicitly set; otherwise auto-generates
        them from predicate_vocabulary and target_predicate.

        Format (ILASP/Metagol):
            modeh(1, pred(+circuit, -entity))    — head: 1 solution
            modeb(*, role(+entity, #role_label)) — body: any number, constant role
        """
        if self.mode_declarations:
            return self.mode_declarations
        # Auto-generate from vocabulary
        return _auto_mode_declarations(self.target_predicate, self.predicate_vocabulary)

    def get_admissible_universe(self, predicate: str) -> List[str]:
        """
        Get Closed World Assumption - CWA admissible universe for a predicate.

        Falls back to the 'admissible_universe' key inside
        predicate_vocabulary if not in self.admissible_universes.
        """
        if predicate in self.admissible_universes:
            return self.admissible_universes[predicate]
        vocab_entry = self.predicate_vocabulary.get(predicate, {})
        return vocab_entry.get('admissible_universe', [])

    def render_hypothesis(self, template: str) -> str:
        """
        Render a hypothesis template with task-specific values.

        Placeholders:
            {target}     -> self.target_predicate
            {role_label} -> self.role_label (or TARGET_PREDICATE in upper)
        """
        return template.format(
            target=self.target_predicate,
            role_label=self.role_label or self.target_predicate.upper()
        )

    def get_hypotheses(self) -> List[str]:
        """Return all hypothesis templates rendered with task values."""
        return [self.render_hypothesis(t) for t in self.hypothesis_templates]

    def has_predicate(self, name: str) -> bool:
        """Check whether a predicate is in this task's vocabulary."""
        return name in self.predicate_vocabulary

    def get_example_key(self, generic_key: str, *fallback_keys: str) -> str:
        """
        Resolve a generic key to the task's actual key name.

        Args:
            generic_key: Canonical key (e.g. 'entity')
            *fallback_keys: Additional keys to try in order

        Returns:
            The first key that exists in example_key_mapping,
            or the generic_key itself as final fallback.
        """
        if generic_key in self.example_key_mapping:
            return self.example_key_mapping[generic_key]
        for fb in fallback_keys:
            if fb in self.example_key_mapping:
                return self.example_key_mapping[fb]
        return generic_key


# ============================================================
# Built-in Semantic Role Configs
# ============================================================

_SEMANTIC_ROLE_UNIVERSE = [
    'TIME', 'LOCATION', 'INSTRUMENT', 'PATIENT', 'AGENT',
    'MANNER', 'PURPOSE', 'PATH'
]

_SEMANTIC_ROLE_VOCAB = {
    'entity': {
        'arity': 3,
        'args': ['ID', 'Text', 'Position'],
    },
    'role': {
        'arity': 2,
        'args': ['ID', 'RoleType'],
        'admissible_universe': _SEMANTIC_ROLE_UNIVERSE,
    },
    'token_at': {
        'arity': 2,
        'args': ['Position', 'Token'],
    },
    'activates': {
        'arity': 3,
        'args': ['Component', 'Entity', 'Score'],
    },
}

_SEMANTIC_ROLE_KEY_MAPPING = {
    'entity':     'target_entity',
    'entity_idx': 'token_idx',
    'label':      'label',
    'correct':    'correct',
    'prompt':     'prompt',   # _build_contrastive_examples sets both 'prompt' and 'input'
}


def _auto_mode_declarations(
        target_predicate: str,
        predicate_vocabulary: Dict[str, Dict],
) -> List[str]:
    """
    Gap 8: Auto-generate mode declarations from predicate vocabulary.

    For each predicate in the vocabulary, generate a modeb declaration
    that marks all args as input/output based on standard ILP conventions:
      - First arg (typically circuit/context ID): input (+)
      - Second arg (typically entity): output (-)
      - Further args: constant (#) if from a closed universe, else output (-)

    The target predicate generates a modeh declaration.
    """
    decls: List[str] = []

    # Head mode — one solution per call
    vocab = predicate_vocabulary.get(target_predicate, {})
    n_args = vocab.get("arity", 2)
    args_str = _mode_args(vocab.get("args", []), n_args)
    decls.append(f"modeh(1, {target_predicate}({args_str})).")

    # Body modes — each auxiliary predicate can appear any number of times
    for pred_name, pred_info in predicate_vocabulary.items():
        if pred_name == target_predicate:
            continue
        n = pred_info.get("arity", len(pred_info.get("args", [])))
        args_str = _mode_args(pred_info.get("args", []), n, pred_info.get("admissible_universe"))
        decls.append(f"modeb(*, {pred_name}({args_str})).")

    return decls


def _mode_args(arg_names: List[str], n_args: int,
               universe: Optional[List[str]] = None) -> str:
    """
    Build mode arg string from argument name list.

    Convention:
      - "ID" / "Circuit" / "C" → input (+)
      - Names ending in "Type" or "Role" → constant (#)  if universe is closed
      - Everything else → output (-)
    """
    result = []
    for i, name in enumerate(arg_names[:n_args]):
        name_upper = name.upper()
        if i == 0 and name_upper in ("ID", "CIRCUIT", "C", "CONTEXT"):
            result.append(f"+{name.lower()}")
        elif "TYPE" in name_upper or "ROLE" in name_upper or "LABEL" in name_upper:
            result.append(f"#{name.lower()}")
        elif i == 0:
            result.append(f"+{name.lower()}")
        else:
            result.append(f"-{name.lower()}")
    # Pad if arg_names shorter than n_args
    for i in range(len(arg_names), n_args):
        result.append(f"-arg{i}")
    return ", ".join(result)


def _sem_role_config(task_name: str, target_pred: str, role: str) -> TaskConfig:
    """Factory for semantic-role task configs.

    Templates describe genuine circuit-behaviour predicates only.
    'role' is declared as a label_predicate so it is stripped from the
    ILP fact database before learning — it encodes the dataset split,
    not anything the circuit does.  Any future task with a class-membership
    predicate should declare it in label_predicates the same way.
    """
    _sem_mode_decls = [
        f"modeh(1, {target_pred}(+circuit, -entity)).",
        # role is NOT in modeb — it's a label predicate, not a search predicate
        "modeb(*, entity(-entity, -text, -position)).",         # entity is output
        "modeb(*, activates(-component, +entity, -score)).",    # H existential, entity bound
        "modeb(*, at_layer(+component, -layer)).",              # H bound from activates
        "modeb(*, token_at(+position, -token)).",               # position bound from entity
    ]
    return TaskConfig(
        task_name=task_name,
        target_predicate=target_pred,
        predicate_vocabulary=_SEMANTIC_ROLE_VOCAB.copy(),
        hypothesis_templates=[
            # Circuit-behaviour templates: ask what the circuit activates on,
            # where, and whether the token appears at that position.
            # These generalise across all semantic roles — role label is not
            # part of the hypothesis, only of the example construction.
            "{target}(C, E) :- entity(E, _, _), activates(H, E, A).",
            "{target}(C, E) :- entity(E, _, Pos), activates(H, E, A), at_layer(H, L).",
            "{target}(C, E) :- entity(E, Text, Pos), token_at(Pos, Text), activates(H, E, A).",
            "{target}(C, E) :- entity(E, Text, Pos), token_at(Pos, Text).",
        ],
        role_label=role,
        example_key_mapping=_SEMANTIC_ROLE_KEY_MAPPING.copy(),
        admissible_universes={
            'role': _SEMANTIC_ROLE_UNIVERSE,
        },
        mode_declarations=_sem_mode_decls,
        label_predicates={'role'},   # excluded from ILP fact DB and hypothesis search
    )


TIME_CONFIG       = _sem_role_config('time_entity',    'time_entity',    'TIME')
LOCATION_CONFIG   = _sem_role_config('location_entity','location_entity','LOCATION')
INSTRUMENT_CONFIG = _sem_role_config('instrument',     'instrument',     'INSTRUMENT')
PATH_CONFIG       = _sem_role_config('path_entity',    'path_entity',    'PATH')
MANNER_CONFIG     = _sem_role_config('manner_entity',  'manner_entity',  'MANNER')
AGENT_CONFIG      = _sem_role_config('agent_entity',   'agent_entity',   'AGENT')
PATIENT_CONFIG    = _sem_role_config('patient_entity', 'patient_entity', 'PATIENT')
PURPOSE_CONFIG    = _sem_role_config('purpose_entity', 'purpose_entity', 'PURPOSE')

# Convenience: all semantic-role configs in one dict for batch processing
SEMANTIC_ROLE_CONFIGS: Dict[str, 'TaskConfig'] = {
    'time':       TIME_CONFIG,
    'location':   LOCATION_CONFIG,
    'instrument': INSTRUMENT_CONFIG,
    'path':       PATH_CONFIG,
    'manner':     MANNER_CONFIG,
    'agent':      AGENT_CONFIG,
    'patient':    PATIENT_CONFIG,
    'purpose':    PURPOSE_CONFIG,
}


# ============================================================
# IOI Config factory
# ============================================================

_IOI_VOCAB = {
    'is_name': {
        'arity': 2,
        'args': ['Token', 'Position'],
    },
    'is_repeated': {
        'arity': 1,
        'args': ['Token'],
    },
    'not_repeated': {
        'arity': 1,
        'args': ['Token'],
    },
    'earlier_than': {
        'arity': 2,
        'args': ['Token1', 'Token2'],
    },
    'activates': {
        'arity': 3,
        'args': ['Component', 'Token', 'Score'],
    },
}

_IOI_TEMPLATES = [
    "{target}(C, N) :- is_name(N, _), not_repeated(N).",
    "{target}(C, N) :- is_name(N, _), not_repeated(N), earlier_than(N, _).",
    "{target}(C, N) :- is_name(N, _), not_repeated(N), activates(H, N, A).",
    "{target}(C, N) :- is_name(N, _), not_repeated(N), activates(H, N, A), at_layer(H, L).",
]

_IOI_KEY_MAPPING = {
    # Keys match the normalised dict produced by _build_contrastive_examples
    'entity':         'target_entity',
    'entity_idx':     'token_idx',
    'distractor':     'distractor',
    'distractor_idx': 'distractor_idx',
    'correct':        'correct',
    'prompt':         'prompt',
    'label':          'label',
}

_IOI_MODE_DECLS = [
    "modeh(1, indirect_object(+circuit, -token)).",
    "modeb(*, is_name(+token, -position)).",   # token is bound from head output
    "modeb(*, not_repeated(+token)).",          # token is bound
    "modeb(*, is_repeated(+token)).",           # token is bound
    "modeb(*, earlier_than(+token, -token)).",  # first token bound, second output
    "modeb(*, activates(-component, +token, -score)).",  # H is existential component
    "modeb(*, at_layer(+component, -layer)).",  # H bound from activates output
]


def _ioi_config(
        task_name: str = 'indirect_object',
        target_predicate: str = 'indirect_object',
) -> TaskConfig:
    """Factory for IOI-family task configs."""
    return TaskConfig(
        task_name=task_name,
        target_predicate=target_predicate,
        predicate_vocabulary=_IOI_VOCAB.copy(),
        hypothesis_templates=list(_IOI_TEMPLATES),
        role_label=None,
        example_key_mapping=_IOI_KEY_MAPPING.copy(),
        mode_declarations=list(_IOI_MODE_DECLS),
    )


IOI_CONFIG = _ioi_config()


# ============================================================
# Greater-Than Config factory
# ============================================================

_GT_VOCAB = {
    'is_number': {
        'arity': 3,
        'args': ['Token', 'Position', 'Value'],
    },
    'reference_num': {
        'arity': 2,
        'args': ['Token', 'Value'],
    },
    'candidate_num': {
        'arity': 2,
        'args': ['Token', 'Value'],
    },
    'activates': {
        'arity': 3,
        'args': ['Component', 'Token', 'Score'],
    },
}

_GT_TEMPLATES = [
    "{target}(C, N) :- candidate_num(N, V), reference_num(_, Ref), V > Ref.",
    "{target}(C, N) :- candidate_num(N, V), reference_num(_, Ref), V > Ref, activates(H, N, A).",
    "{target}(C, N) :- candidate_num(N, V), reference_num(_, Ref), V > Ref, activates(H, N, A), at_layer(H, L).",
]

_GT_KEY_MAPPING = {
    'entity':      'target_number',
    'entity_idx':  'reference_idx',
    'reference':   'reference_number',
    'correct':     'correct',
    'prompt':      'prompt',
    'value':       'target_number',
}

_GT_MODE_DECLS = [
    "modeh(1, greater_than_prediction(+circuit, -token)).",
    "modeb(*, candidate_num(-token, -value)).",          # token and value are outputs
    "modeb(*, reference_num(-token, -value)).",          # token and value are outputs
    "modeb(*, is_number(-token, -position, -value)).",   # all outputs
    "modeb(*, activates(-component, +token, -score)).",  # H existential, token bound
    "modeb(*, at_layer(+component, -layer)).",           # H bound from activates
]


def _gt_config(
        task_name: str = 'greater_than_prediction',
        target_predicate: str = 'greater_than_prediction',
) -> TaskConfig:
    """Factory for Greater-Than-family task configs."""
    return TaskConfig(
        task_name=task_name,
        target_predicate=target_predicate,
        predicate_vocabulary=_GT_VOCAB.copy(),
        hypothesis_templates=list(_GT_TEMPLATES),
        role_label=None,
        example_key_mapping=_GT_KEY_MAPPING.copy(),
        mode_declarations=list(_GT_MODE_DECLS),
    )


GT_CONFIG = _gt_config()


# ============================================================
# Task Registry
# ============================================================

TASK_REGISTRY: Dict[str, TaskConfig] = {
    'time':         TIME_CONFIG,
    'location':     LOCATION_CONFIG,
    'instrument':   INSTRUMENT_CONFIG,
    'path':         PATH_CONFIG,
    'manner':       MANNER_CONFIG,
    'agent':        AGENT_CONFIG,
    'patient':      PATIENT_CONFIG,
    'purpose':      PURPOSE_CONFIG,
    'ioi':          IOI_CONFIG,
    'greater_than': GT_CONFIG,
}


def get_task_config(task_name: str) -> TaskConfig:
    """
    Look up a built-in task config by name.

    Args:
        task_name: Registry key (e.g. 'time', 'ioi', 'greater_than')

    Returns:
        TaskConfig instance

    Raises:
        ValueError: If task_name is not in the registry.
    """
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task: '{task_name}'. "
            f"Available: {sorted(TASK_REGISTRY.keys())}. "
            f"Or create a custom TaskConfig instance."
        )
    return TASK_REGISTRY[task_name]


def register_task(name: str, config: TaskConfig) -> None:
    """
    Register a custom task config at runtime.

    Args:
        name: Registry key
        config: TaskConfig instance
    """
    TASK_REGISTRY[name] = config
