"""
Predicate Extraction Module (Generalized)

Converts OWL circuits + task examples → FOL predicate facts for ILP.
All task-specific behaviour is driven by TaskConfig — no hardcoded
role labels, predicate maps, or field names.

Fact emission uses a dispatch table keyed by the 'fact_type' metadata
in each predicate's vocabulary entry.  When no explicit fact_type is
given, the predicate name itself is used as the dispatch key.

Implements draft §2.2.

"""

from rdflib import Graph, Namespace, URIRef, Literal
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

from .task_config import TaskConfig



def _safe_int(val, default: int = 0) -> int:
    """Convert val to int safely; return default if val is None, NaN, or unconvertible."""
    if val is None:
        return default
    try:
        f = float(val)
        import math
        if math.isnan(f):
            return default
        return int(f)
    except (TypeError, ValueError):
        return default

def _safe_bool(val, default: bool = False) -> bool:
    """Convert val to bool safely, handling string 'True'/'False', NaN, None."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ('true', '1', 'yes', 'y')
    try:
        import math
        f = float(val)
        if math.isnan(f):
            return default
        return bool(f)
    except (TypeError, ValueError):
        return default


class PredicateExtractor:
    """
    Extract FOL predicates from OWL circuits for ILP learning.

    Converts:
    - OWL circuit (RDF graph) → Structural facts (components, edges)
    - Task examples (dicts)   → Task facts driven by TaskConfig
    - Schema constraints      → Background knowledge
    - TaskConfig CWA universes → Bounded CWA negations

    Output: Complete Prolog/ProbLog program F(C,x)
    """

    def __init__(self, task_config: TaskConfig, schema=None):
        """
        Args:
            task_config: TaskConfig specifying predicates and key mappings
            schema: MECHINT_SCHEMA (optional, for admissible universes)
        """
        self.task_config = task_config
        self.schema = schema
        self.namespace = Namespace("http://purl.org/mechint#")

        self.entity_counter = 0
        self.entity_map = {}

    # ================================================================
    # Structural fact extraction (task-independent)
    # ================================================================

    def extract_circuit_facts(
            self,
            circuit_graph: Graph,
            circuit_uri: URIRef
    ) -> List[str]:
        """
        Extract structural facts from OWL circuit.

        Returns Prolog facts for circuit structure: components, edges,
        layers, types, connectivity.  These are task-independent.
        """
        facts = []
        circuit_id = self._simplify_uri(circuit_uri)
        facts.append(f"circuit({circuit_id}).")

        # Components
        comp_q = f"""
        PREFIX mechint: <{self.namespace}>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?comp ?type ?layer ?head WHERE {{
            <{circuit_uri}> mechint:hasComponent ?comp .
            ?comp rdf:type ?type .
            OPTIONAL {{ ?comp mechint:inLayer ?layer }}
            OPTIONAL {{ ?comp mechint:headIndex ?head }}
        }}
        """
        for row in circuit_graph.query(comp_q):
            cid = self._simplify_uri(row.comp)
            ctype = str(row.type).split('#')[-1]
            if 'AttentionHead' in ctype:
                facts.append(f"head({cid}).")
                facts.append(f"component_type({cid}, attention).")
            elif 'MLP' in ctype:
                facts.append(f"mlp({cid}).")
                facts.append(f"component_type({cid}, mlp).")
            if row.layer is not None:
                facts.append(f"at_layer({cid}, {int(row.layer)}).")
            if row.head is not None:
                facts.append(f"head_index({cid}, {int(row.head)}).")

        # Edges
        edge_q = f"""
        PREFIX mechint: <{self.namespace}>
        SELECT ?edge ?source ?target ?edgeType WHERE {{
            <{circuit_uri}> mechint:hasMemberEdge ?edge .
            ?edge mechint:hasSource ?source ;
                  mechint:hasTarget ?target .
            OPTIONAL {{ ?edge mechint:edgeType ?edgeType }}
        }}
        """
        ec = 0
        for row in circuit_graph.query(edge_q):
            ec += 1
            sid = self._simplify_uri(row.source)
            tid = self._simplify_uri(row.target)
            et = str(row.edgeType) if row.edgeType else 'unknown'
            facts.append(f"edge(e{ec}, {sid}, {tid}, '{et}').")
            facts.append(f"connects({sid}, {tid}).")

        facts.append("")
        facts.append("% Background knowledge")
        facts.append("compatible(C1, C2) :- edge(_, C1, C2, _).")
        facts.append("compatible(C1, C2) :- edge(_, C2, C1, _).")

        return facts

    # ================================================================
    # Task fact extraction (config-driven)
    # ================================================================

    def extract_task_facts(
            self,
            examples: List[Dict],
            circuit_uri: URIRef,
            tokenizer_mapping: Optional[Dict] = None
    ) -> List[str]:
        """
        Extract task-specific facts from examples.

        Iterates over each declared predicate in the task config and
        dispatches to the appropriate emitter based on the predicate's
        'fact_type' metadata.
        """
        facts = []
        circuit_id = self._simplify_uri(circuit_uri)
        cfg = self.task_config

        self.entity_counter = 0
        self.entity_map = {}

        vocab_names = list(cfg.predicate_vocabulary.keys())

        for example in examples:
            self.entity_counter += 1
            entity_id = f"e{self.entity_counter}"

            entity_text = self._get_field(example, cfg, 'entity', 'target_entity', '')
            entity_pos = self._get_field(example, cfg, 'entity_idx', 'token_idx', 0)
            is_correct = _safe_bool(self._get_field(
                example, cfg, 'correct',
                'is_correct', 'model_correct', 'both_correct',
                'label_correct', 'correct_prediction',
                default=False))

            # Deduplicate entities
            entity_pos = _safe_int(entity_pos, default=0)
            entity_key = (str(entity_text), entity_pos)
            if entity_key in self.entity_map:
                entity_id = self.entity_map[entity_key]
            else:
                self.entity_map[entity_key] = entity_id

            # Emit facts for each declared predicate
            ctx = _EmitContext(
                entity_id=entity_id,
                entity_text=entity_text,
                entity_pos=entity_pos,
                example=example,
                cfg=cfg,
                tokenizer_mapping=tokenizer_mapping,
                extractor=self
            )
            for pred_name in vocab_names:
                pred_facts = _emit_for_predicate(pred_name, ctx)
                facts.extend(pred_facts)

            # Positive / negative annotation
            target_pred = cfg.target_predicate
            if is_correct:
                facts.append(f"positive({target_pred}({circuit_id}, {entity_id})).")
            else:
                facts.append(f"negative({target_pred}({circuit_id}, {entity_id})).")

        # Diagnostic: warn if no positive examples were produced
        pos_count = sum(1 for f in facts if f.startswith('positive('))
        neg_count = sum(1 for f in facts if f.startswith('negative('))
        if pos_count == 0 and examples:
            # Sample the first example to help diagnose the key mapping
            sample = examples[0] if examples else {}
            sample_keys = list(sample.keys())[:10]
            print(f"  [PredicateExtractor] WARNING: 0 positive examples from "
                  f"{len(examples)} rows.")
            print(f"  Correct-column tried: "
                  f"'{cfg.example_key_mapping.get('correct', 'correct')}' + "
                  f"fallbacks (is_correct, model_correct, both_correct, ...)")
            print(f"  Actual CSV columns seen: {sample_keys}")
            print(f"  Sample 'correct'-like values: "
                  f"{ {k: sample[k] for k in sample_keys if 'correct' in k.lower()} }")
            print(f"  Negative examples produced: {neg_count}")

        # Background knowledge from config
        if cfg.background_knowledge:
            facts.append("")
            facts.append("% Background knowledge (from task config)")
            facts.extend(cfg.background_knowledge)

        return facts

    # ================================================================
    # Bounded Closed-World Assumption (config-driven)
    # ================================================================

    def apply_bounded_cwa(
            self,
            facts: List[str],
            predicate_info: Optional[Dict] = None
    ) -> List[str]:
        """
        Apply bounded CWA for predicates with admissible universes.

        For each predicate P with universe U (from task config):
        If P(entity, val) not in facts AND val in U → add not_P(entity, val)
        """
        cwa_facts = []
        cfg = self.task_config

        for pred_name in cfg.predicate_vocabulary:
            universe = cfg.get_admissible_universe(pred_name)
            if not universe:
                continue

            existing = set()
            entity_ids = set()
            prefix = f"{pred_name}("

            for fact in facts:
                if fact.startswith(prefix):
                    inner = fact[len(prefix):-2]
                    parts = [p.strip().strip("'\"") for p in inner.split(',')]
                    if len(parts) >= 2:
                        entity_ids.add(parts[0])
                        existing.add((parts[0], parts[1]))

            if not entity_ids:
                continue

            cwa_facts.append("")
            cwa_facts.append(
                f"% Closed-world assumption for {pred_name} "
                f"(bounded by admissible universe)"
            )
            for eid in sorted(entity_ids):
                for val in universe:
                    if (eid, val) not in existing:
                        cwa_facts.append(f"not_{pred_name}({eid}, '{val}').")

        return cwa_facts

    # ================================================================
    # Build complete fact database
    # ================================================================

    def build_fact_database(
            self,
            circuit_graph: Graph,
            circuit_uri: URIRef,
            examples: List[Dict],
            tokenizer_mapping: Optional[Dict] = None,
            apply_cwa: bool = True
    ) -> str:
        """
        Build complete F(C,x) fact database.

        Combines structural facts, task facts, and CWA negations.
        """
        all_facts = []
        circuit_id = self._simplify_uri(circuit_uri)

        all_facts.append(f"% Fact database for circuit: {circuit_id}")
        all_facts.append(f"% Task: {self.task_config.task_name}")
        all_facts.append(f"% Generated by PredicateExtractor")
        all_facts.append("")

        all_facts.append("% === Structural Facts ===")
        structural = self.extract_circuit_facts(circuit_graph, circuit_uri)
        all_facts.extend(structural)
        all_facts.append("")

        all_facts.append("% === Task Facts ===")
        task = self.extract_task_facts(examples, circuit_uri, tokenizer_mapping)
        all_facts.extend(task)
        all_facts.append("")

        if apply_cwa:
            all_facts.append("% === Closed-World Assumption ===")
            cwa = self.apply_bounded_cwa(structural + task)
            all_facts.extend(cwa)
            all_facts.append("")

        return "\n".join(all_facts)

    # ================================================================
    # Helpers
    # ================================================================

    def _get_field(self, example: Dict, cfg: TaskConfig,
                   generic_key: str, *fallbacks, default=None):
        """Read a field using the config's key mapping with fallbacks."""
        mapped = cfg.example_key_mapping.get(generic_key)
        if mapped and mapped in example:
            return example[mapped]
        if generic_key in example:
            return example[generic_key]
        for fb in fallbacks:
            if isinstance(fb, str) and fb in example:
                return example[fb]
        return default

    def _simplify_uri(self, uri) -> str:
        """Simplify URI to Prolog-friendly identifier."""
        uri_str = str(uri)
        if '#' in uri_str:
            local = uri_str.split('#')[-1]
        else:
            local = uri_str.split('/')[-1]
        if local.startswith('circuit_'):
            local = 'c_' + local[8:]
        elif local.startswith('head_'):
            local = 'h' + local[5:]
        elif local.startswith('mlp_'):
            local = 'm' + local[4:]
        local = local.replace('-', '_').replace('.', '_')
        return local

    def get_predicate_info(self, predicate_name: str) -> Optional[Dict]:
        """Get metadata for a predicate from task config vocabulary."""
        return self.task_config.predicate_vocabulary.get(predicate_name)

    def list_predicates(self) -> List[str]:
        """List all predicate names in this task's vocabulary."""
        return list(self.task_config.predicate_vocabulary.keys())


# ================================================================
# Fact emission dispatch (module-level, outside the class)
# ================================================================
# Dispatch table mapping fact_type → emitter function.
# Each emitter receives an _EmitContext and returns List[str].
# The dispatch key is the 'fact_type' from predicate metadata,
# falling back to the predicate name itself.


class _EmitContext:
    """Bundles all data an emitter needs."""
    __slots__ = (
        'entity_id', 'entity_text', 'entity_pos',
        'example', 'cfg', 'tokenizer_mapping', 'extractor'
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_field(self, generic_key, *fallbacks, default=None):
        return self.extractor._get_field(
            self.example, self.cfg, generic_key, *fallbacks, default=default
        )


def _emit_entity_facts(pred_name: str, ctx: _EmitContext) -> List[str]:
    pos = _safe_int(ctx.entity_pos)
    return [f'entity({ctx.entity_id}, "{ctx.entity_text}", {pos}).']


def _emit_role_facts(pred_name: str, ctx: _EmitContext) -> List[str]:
    if ctx.cfg.role_label is None:
        return []
    role = ctx.get_field('label', default=ctx.cfg.role_label)
    return [f"role({ctx.entity_id}, '{role}')."]


def _emit_token_at_facts(pred_name: str, ctx: _EmitContext) -> List[str]:
    tokens = ctx.example.get('tokens', [])
    pos = _safe_int(ctx.entity_pos)
    if 0 <= pos < len(tokens):
        return [f'token_at({pos}, "{tokens[pos]}").']
    return []


def _emit_activates_facts(pred_name: str, ctx: _EmitContext) -> List[str]:
    if not ctx.tokenizer_mapping:
        return []
    pos = _safe_int(ctx.entity_pos)
    if pos not in ctx.tokenizer_mapping:
        return []
    out = []
    for comp_id, score in ctx.tokenizer_mapping[pos]:
        cs = (ctx.extractor._simplify_uri(comp_id)
              if isinstance(comp_id, URIRef) else comp_id)
        out.append(f"activates({cs}, {ctx.entity_id}, {score:.3f}).")
    return out


def _emit_name_flag_facts(pred_name: str, ctx: _EmitContext) -> List[str]:
    pos = _safe_int(ctx.entity_pos)
    return [f"{pred_name}({ctx.entity_id}, {pos})."]


def _emit_repetition_facts(pred_name: str, ctx: _EmitContext) -> List[str]:
    """Handle is_repeated / not_repeated style predicates.

    Determines whether the entity text appears more than once in the
    prompt.  Uses actual string counting, not distractor comparison,
    to get the correct answer for IOI-style data where the subject
    (repeated name) can appear as either target or distractor.
    """
    facts = []
    meta = ctx.cfg.predicate_vocabulary.get(pred_name, {})
    polarity = meta.get('repetition_polarity')

    if polarity is None:
        name_lower = pred_name.lower()
        if 'not_' in name_lower or 'unique' in name_lower:
            polarity = 'unique'
        else:
            polarity = 'repeated'

    # Count occurrences of entity text in the prompt
    prompt = ctx.get_field('prompt', 'input', default='')
    entity_str = str(ctx.entity_text).lower()
    count = 0
    if prompt and entity_str:
        # Count word-level occurrences (strip punctuation)
        words = [w.strip('.,!?;:\'\"') for w in prompt.lower().split()]
        count = words.count(entity_str)

    is_repeated = count > 1

    if is_repeated and polarity == 'repeated':
        facts.append(f"{pred_name}({ctx.entity_id}).")
    elif not is_repeated and polarity == 'unique':
        facts.append(f"{pred_name}({ctx.entity_id}).")

    return facts


def _emit_ordering_facts(pred_name: str, ctx: _EmitContext) -> List[str]:
    """Handle positional ordering predicates (earlier_than)."""
    dist_idx = ctx.get_field('distractor_idx', default=None)
    if dist_idx is not None and _safe_int(ctx.entity_pos) < _safe_int(dist_idx):
        dist_eid = f"d{ctx.extractor.entity_counter}"
        return [f"{pred_name}({ctx.entity_id}, {dist_eid})."]
    return []


def _emit_numeric_facts(pred_name: str, ctx: _EmitContext) -> List[str]:
    val = ctx.get_field('value', 'target_number', default=0)
    return [f"{pred_name}({ctx.entity_id}, {ctx.entity_pos}, {val})."]


def _emit_numeric_ref_facts(pred_name: str, ctx: _EmitContext) -> List[str]:
    ref = ctx.get_field('reference', 'reference_number', default=0)
    return [f"{pred_name}({ctx.entity_id}, {ref})."]


def _emit_numeric_cand_facts(pred_name: str, ctx: _EmitContext) -> List[str]:
    val = ctx.get_field('value', 'target_number', default=0)
    return [f"{pred_name}({ctx.entity_id}, {val})."]


# Master dispatch table
_EMITTER_TABLE: Dict[str, callable] = {
    'entity':           _emit_entity_facts,
    'role':             _emit_role_facts,
    'token_at':         _emit_token_at_facts,
    'activates':        _emit_activates_facts,
    'is_name':          _emit_name_flag_facts,
    'is_repeated':      _emit_repetition_facts,
    'not_repeated':     _emit_repetition_facts,
    'repetition':       _emit_repetition_facts,
    'earlier_than':     _emit_ordering_facts,
    'positional_order': _emit_ordering_facts,
    'is_number':        _emit_numeric_facts,
    'reference_num':    _emit_numeric_ref_facts,
    'candidate_num':    _emit_numeric_cand_facts,
}


def _emit_for_predicate(pred_name: str, ctx: _EmitContext) -> List[str]:
    """
    Top-level dispatch: look up fact_type from predicate metadata,
    fall back to pred_name, then dispatch to the emitter table.
    """
    meta = ctx.cfg.predicate_vocabulary.get(pred_name, {})
    fact_type = meta.get('fact_type', pred_name)

    emitter = _EMITTER_TABLE.get(fact_type)
    if emitter is not None:
        return emitter(pred_name, ctx)

    # Unknown fact_type — skip silently
    return []
