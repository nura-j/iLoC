"""
ILP Semantic Specification Learner (Generalized)

Learns FOL specifications from fact databases via ILP (§2.3).
Uses ProbLog for hypothesis evaluation and learning.

All task-specific behaviour (hypothesis templates, role maps) is
driven by TaskConfig  -  no hardcoded predicate names or role labels.

"""

from typing import List, Dict, Optional, Tuple, Set
from problog.program import PrologString, PrologFile
from problog import get_evaluatable
from problog.learning.lfi import LFIProblem
import re
import math
import tempfile
import os
from collections import Counter

from .task_config import TaskConfig


class SemanticSpecificationLearner:
    """
    Learn semantic specifications via ILP.

    Uses ProbLog to learn first-order logic rules that explain
    positive examples while avoiding negative examples.

    Hypothesis generation is driven entirely by TaskConfig templates.
    """

    def __init__(
            self,
            task_config: TaskConfig,
            ilp_system: str = 'problog',
            config: Optional[Dict] = None
    ):
        """
        Initialize ILP learner.

        Args:
            task_config: TaskConfig driving hypothesis generation
            ilp_system: 'problog' (only option for now)
            config: Optional configuration overrides
        """
        self.task_config = task_config
        self.ilp_system = ilp_system
        self.config = config or self._default_config()

        if ilp_system != 'problog':
            raise ValueError(f"Only 'problog' supported, got: {ilp_system}")

    def learn_specification(
            self,
            facts: str,
            positive_examples: List[str],
            negative_examples: List[str],
            background_knowledge: Optional[str] = None
    ) -> Dict:
        """
        Learn FOL specification via ILP.

        Args:
            facts: Prolog program (fact database from Module 1)
            positive_examples: e.g. ["positive(instrument(c1, e1))."]
            negative_examples: e.g. ["negative(instrument(c1, e2))."]
            background_knowledge: Optional additional rules

        Returns:
            Dict with:
                - specification_text: Learned FOL rule
                - predicates: List of predicates used in rule body
                - confidence: Learning confidence [0,1]
                - ilp_signature: S_ILP metadata

        Process:
            1. Parse facts and examples
            2. Generate candidate hypotheses from TaskConfig templates
            3. Evaluate hypotheses (coverage + precision)
            4. Select best hypothesis
            5. Compute metadata (signature, confidence)
        """
        print("\n" + "=" * 60)
        print("ILP LEARNING")
        print("=" * 60)

        # Parse target predicate from examples
        target_predicate = self._extract_target_predicate(positive_examples)
        print(f"Target predicate: {target_predicate}")
        print(f"Positive examples: {len(positive_examples)}")
        print(f"Negative examples: {len(negative_examples)}")

        # Build complete program
        complete_program = self._build_program(
            facts,
            positive_examples,
            negative_examples,
            background_knowledge
        )

        # Generate hypotheses from task config templates
        print("\nGenerating hypotheses...")
        hypotheses = self._generate_hypotheses(
            complete_program,
            target_predicate,
            positive_examples,
            negative_examples
        )
        print(f"Generated {len(hypotheses)} candidate hypotheses")

        # Gap 5: Learn probabilistic weights λ via LFI (H = (L, λ), paper §3.2)
        print("\nLearning probabilistic weights (LFI)...")
        lfi_weights = self._learn_weights_lfi(
            hypotheses,
            facts,
            positive_examples,
            negative_examples,
        )
        if lfi_weights:
            # Strip sentinel key before display — it is not a real weight
            display_weights = {k: v for k, v in lfi_weights.items()
                               if not k.startswith('__')}
            if display_weights:
                top_lfi = max(display_weights.items(), key=lambda kv: kv[1])
                print(f"  LFI top weight: {top_lfi[0]} → λ={top_lfi[1]:.4f}")

        # Evaluate and select best (binary coverage * precision, Eq. 13 argmax)
        print("\nEvaluating hypotheses...")
        best_hypothesis = self._select_best_hypothesis(
            hypotheses,
            complete_program,
            positive_examples,
            negative_examples
        )

        if best_hypothesis is None:
            print("  Warning: No good hypothesis found, using fallback")
            best_hypothesis = self._create_fallback_hypothesis(target_predicate)

        print(f"\n  Best hypothesis:")
        print(f"  {best_hypothesis}")

        # Extract predicates used
        predicates = self._extract_predicates_from_rule(best_hypothesis)

        # Compute signature
        signature = self.compute_signature([best_hypothesis])

        # Compute confidence
        confidence = self.compute_confidence(
            [best_hypothesis],
            complete_program,
            positive_examples,
            negative_examples
        )

        print(f"\n  Learning complete:")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Predicates: {predicates}")

        # Gap 3: expose both confidence variants
        confidence_eq2 = getattr(self, '_last_confidence_eq2', confidence)
        confidence_adj = getattr(self, '_last_confidence_adj', confidence)

        # Gap 5: attach LFI weights to signature
        # Strip the sentinel before storing weights so downstream JSON stays clean
        lfi_available = bool(lfi_weights.pop("__lfi_available__", False))
        best_idx   = hypotheses.index(best_hypothesis) if best_hypothesis in hypotheses else -1
        best_lambda = lfi_weights.get(f"h{best_idx}", None) if best_idx >= 0 else None

        if not lfi_available:
            print("  Note: LFI weights are uniform fallbacks (ProbLog unavailable/failed)")
            best_lambda = None   # don't store a fake lambda

        # Embed into ilp_signature so add_ilp_semantic can store them
        signature['confidence_eq2']  = confidence_eq2
        signature['confidence_adj']  = confidence_adj
        signature['lfi_weights']     = lfi_weights if lfi_available else {}
        signature['best_lambda']     = best_lambda
        signature['lfi_available']   = lfi_available   # consumers can check this

        return {
            'specification_text': best_hypothesis,
            'predicates':         predicates,
            'confidence':         confidence,          # adjusted (backward-compat)
            'confidence_eq2':     confidence_eq2,      # Gap 3: pure Eq.2
            'confidence_adj':     confidence_adj,      # Gap 3: adjusted formula
            'best_lambda':        best_lambda,         # Gap 5: λ for H* (None if LFI failed)
            'lfi_weights':        lfi_weights if lfi_available else {},
            'lfi_available':      lfi_available,       # True = real EM weights, False = fallback
            'ilp_signature':      signature,
        }

    # ================================================================
    # Hypothesis generation (config-driven)
    # ================================================================


    # ================================================================
    # Gap 8 — Formal mode declarations (paper §3.6, Appendix B.4)
    # Three components:
    #   1. _parse_mode_declarations  — string list → structured dict
    #   2. _validate_clause_modes    — enforce +/-/# binding + modeb set
    #   3. _generate_from_modes      — generate skeletons from modes alone
    # ================================================================

    @staticmethod
    def _parse_mode_declarations(decls: List[str]) -> Dict:
        """
        Component 1 — Parse mode declaration strings into structured dict.

        Input format (ILASP/Metagol convention):
            modeh(1, location_entity(+circuit, -entity)).
            modeb(*, activates(+circuit, -entity, -score)).
            modeb(*, entity(-entity, -text, -position)).

        Output:
            {
              'head': {
                  'pred': 'location_entity',
                  'card': 1,
                  'args': [('+', 'circuit'), ('-', 'entity')]
              },
              'body': {
                  'activates': {
                      'card': '*',
                      'args': [('+', 'circuit'), ('-', 'entity'), ('-', 'score')]
                  },
                  'entity': {
                      'card': '*',
                      'args': [('-', 'entity'), ('-', 'text'), ('-', 'position')]
                  },
              }
            }

        Argument mode markers:
            +type  → input  (must be bound when predicate is called)
            -type  → output (may be unbound; predicate produces binding)
            #type  → constant (must be a ground term from a closed universe)
            _      → anonymous (wildcard, always valid)

        Robustness: silently skips malformed declarations so a bad string
        never crashes the learner.  All tasks will still get template-based
        candidates even if mode parsing fails.
        """
        parsed: Dict = {'head': None, 'body': {}}

        _arg_re = re.compile(r'([+\-#_])(\w+)|\.\.\.'  )

        def _parse_args(args_str: str) -> List[Tuple[str, str]]:
            """Parse '(+circuit, -entity, #role)' into [('+''circuit'), ...]"""
            result = []
            for part in args_str.split(','):
                part = part.strip()
                m = _arg_re.match(part)
                if m:
                    marker = m.group(1) or '_'
                    typ    = m.group(2) or 'any'
                    result.append((marker, typ))
                elif part in ('_', '...'):
                    result.append(('_', 'any'))
            return result

        _decl_re = re.compile(
            r'(modeh|modeb)\s*\(\s*([^,]+),\s*(\w+)\s*\(([^)]*)\)\s*\)\s*\.?'
        )

        for decl in decls:
            m = _decl_re.match(decl.strip())
            if not m:
                continue
            kind, card_str, pred, args_str = m.groups()
            args = _parse_args(args_str)
            card: object = 1 if card_str.strip() == '1' else '*'

            if kind == 'modeh':
                parsed['head'] = {'pred': pred, 'card': card, 'args': args}
            else:
                parsed['body'][pred] = {'card': card, 'args': args}

        return parsed

    @staticmethod
    def _validate_clause_modes(
            clause: str,
            parsed_modes: Dict,
            max_body_literals: int = 4,
    ) -> Tuple[bool, str]:
        """
        Component 2 — Validate a candidate clause against mode declarations.

        Checks three things (paper §3.6, Appendix B.4):

        (a) Every body predicate is in the declared modeb set.
            A predicate not in modeb is illegal regardless of availability.

        (b) Variable binding directions are consistent with +/-/# markers,
            tracked left-to-right through the body literal sequence:
            - Variables appearing as outputs (-) in earlier literals are
              'bound' for subsequent literals.
            - A variable used as input (+) in a literal must already be
              bound (either from the head or a prior body literal).
            - A constant (#) argument must be a ground Prolog term
              (quoted atom, number, or underscore), not a fresh variable.

        (c) Body length ≤ max_body_literals.

        Returns:
            (True, '')          if clause is valid under the mode declarations
            (False, reason_str)  if clause violates any condition

        Robustness: if parsed_modes has no 'head' (parse failed), returns
        (True, '') so the caller falls back to template filtering only.
        """
        if not parsed_modes.get('head') or not parsed_modes.get('body'):
            return True, ''   # no modes to enforce

        if ':-' not in clause:
            return True, ''   # unconditional fact — always legal

        head_str, body_str = clause.split(':-', 1)
        body_str = body_str.strip().rstrip('.')

        # Split body on commas, respecting nested parentheses
        literals = SemanticSpecificationLearner._split_body_literals(body_str)

        # (c) length bound — comparisons (V > Ref, Lmin >= 0) don't count
        _cmp_re = re.compile(r'^\s*[A-Za-z_][\w]*\s*(>|<|>=|<=|=:=|=\\=|\\+|is)\s*')
        predicate_literals = [l for l in literals if not _cmp_re.match(l)]
        if len(predicate_literals) > max_body_literals:
            return (False,
                    f'body length {len(predicate_literals)} > max {max_body_literals}')

        # Collect variables bound by the head (output positions in modeh)
        bound_vars: Set[str] = set()
        head_modes = parsed_modes['head']
        head_match = re.match(r'(\w+)\((.*)\)$', head_str.strip().rstrip('.'))
        if head_match:
            head_args_str = head_match.group(2)
            head_arg_vals = [a.strip() for a in head_args_str.split(',')]
            for (marker, _), val in zip(head_modes['args'], head_arg_vals):
                # Head input args (+) are pre-bound (the circuit id, etc.)
                # Head output args (-) become bound AFTER the head fires
                if val and val[0].isupper():
                    bound_vars.add(val)

        modeb = parsed_modes['body']

        for lit in literals:
            lit = lit.strip()
            if not lit:
                continue

            # Extract predicate name and arguments
            lit_match = re.match(r'(\w+)\((.*)\)$', lit)
            if not lit_match:
                # Comparison operator (N > 3, Lmin >= 0) — always legal
                continue

            pred_name = lit_match.group(1)
            arg_str   = lit_match.group(2)
            arg_vals  = [a.strip() for a in arg_str.split(',')]

            # Builtins are always legal
            if pred_name in ('is', 'not', 'true', 'fail', 'between'):
                continue

            # (a) predicate must be in modeb
            if pred_name not in modeb:
                return False, f'predicate \'{pred_name}\' not in modeb'

            pred_modes = modeb[pred_name]['args']

            # (b) check binding direction for each argument
            for i, (marker, _) in enumerate(pred_modes):
                if i >= len(arg_vals):
                    break
                val = arg_vals[i].strip()

                if marker == '+':
                    # Must be bound — either a ground term or a previously bound var
                    is_var = bool(val) and val[0].isupper()
                    is_ground = (not is_var) or (val == '_')
                    if is_var and val not in bound_vars:
                        return (False,
                                f'var {val} used as input (+) in {pred_name}/arg{i+1} but not yet bound')

                elif marker == '-':
                    # Output — val becomes bound after this literal
                    if val and val[0].isupper() and val != '_':
                        bound_vars.add(val)

                elif marker == '#':
                    # Must be a ground constant — quoted atom or number
                    is_var = bool(val) and val[0].isupper()
                    if is_var and val != '_':
                        return (False,
                                f'var {val} used as constant (#) in {pred_name}/arg{i+1}')

        return True, ''

    @staticmethod
    def _split_body_literals(body: str) -> List[str]:
        """
        Split a clause body on top-level commas (not inside parentheses).

        'entity(E, T, P), activates(H, E, A), at_layer(H, L)'
        → ['entity(E, T, P)', ' activates(H, E, A)', ' at_layer(H, L)']
        """
        literals = []
        depth = 0
        current: List[str] = []
        for ch in body:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                literals.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            literals.append(''.join(current))
        return literals

    def _generate_from_modes(
            self,
            parsed_modes: Dict,
            label_preds: Set[str],
            available: Set[str],
            target_predicate: str,
    ) -> List[str]:
        """
        Component 3 — Generate clause skeletons from mode declarations alone.

        This is the core Gap 8 capability: instead of only filtering
        hand-written templates, the learner can construct new candidate
        clauses from the mode declarations by:

          1. Taking the modeh head shape
          2. For each subset of modeb predicates (up to max_body_literals),
             building a body where output variables from earlier literals
             feed input positions of later literals
          3. Yielding each legal skeleton as a Prolog clause string

        The combinatorial space is bounded by max_body_literals (default 4)
        and the number of modeb predicates (typically 4–6 per task).
        With these bounds the space is ~50–200 candidates — tractable for
        our coverage-based evaluator.

        Variable naming convention:
            First head output variable:  E  (entity)
            Head input variable:         C  (circuit)
            Component variables:         H0, H1, ...
            Auxiliary variables:         V0, V1, ...

        Generalises across all tasks: no predicate names are hardcoded.
        The function reads everything it needs from parsed_modes.
        """
        if not parsed_modes.get('head') or not parsed_modes.get('body'):
            return []

        max_body = self.config.get('max_body_literals', 4)
        head_info = parsed_modes['head']
        head_pred = head_info['pred']

        if head_pred != target_predicate:
            return []   # modes are for a different predicate — skip

        # Build head literal with canonical variable names
        head_vars, head_literal = self._build_head_literal(head_info)

        # Collect usable body predicates (exclude label predicates, respect availability)
        body_preds = {
            p: info for p, info in parsed_modes['body'].items()
            if p not in label_preds
            and (p in available or p in {'activates', 'at_layer'})
        }

        if not body_preds:
            return []

        generated: List[str] = []
        # Enumerate combinations of body predicates, length 1..max_body
        from itertools import combinations, permutations
        pred_names = list(body_preds.keys())

        for length in range(1, min(max_body, len(pred_names)) + 1):
            for combo in combinations(pred_names, length):
                # Try each ordering — binding flow differs by order
                for perm in permutations(combo):
                    clause = self._build_clause_from_perm(
                        head_literal, head_vars, list(perm), body_preds
                    )
                    if clause:
                        generated.append(clause)

        # Deduplicate preserving order
        seen: Set[str] = set()
        unique = []
        for c in generated:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        return unique

    @staticmethod
    def _build_head_literal(head_info: Dict) -> Tuple[Dict[str, str], str]:
        """
        Build the head literal string and return the variable bindings it creates.

        Returns (head_vars dict, head_literal_string).
        head_vars maps arg position index to variable name.
        """
        args_out = []
        head_vars: Dict[str, str] = {}   # marker→variable for downstream use
        var_counter = [0]

        # Canonical names for first two positions
        canonical = {0: 'C', 1: 'E'}

        for i, (marker, typ) in enumerate(head_info['args']):
            if i in canonical:
                var = canonical[i]
            else:
                var = f'V{var_counter[0]}'; var_counter[0] += 1
            args_out.append(var)
            head_vars[f'arg{i}_{marker}'] = var

        pred = head_info['pred']
        literal = f"{pred}({', '.join(args_out)})"
        return head_vars, literal

    @staticmethod
    def _build_clause_from_perm(
            head_literal: str,
            head_vars: Dict[str, str],
            perm: List[str],
            body_preds: Dict[str, Dict],
    ) -> Optional[str]:
        """
        Build a Prolog clause from an ordered list of body predicate names.

        Tracks bound variables left-to-right WITH their declared types.
        Returns None if any input (+) argument cannot be satisfied by a
        bound variable of compatible type.

        Type compatibility prevents nonsense like at_layer(E, L) where E
        is an entity variable — at_layer requires a +component argument
        that must have been introduced as a component type (H prefix) by
        a prior activates(-component, ...) literal.
        """
        # Track bound variables WITH their types: var_name -> type_str
        bound: Dict[str, str] = {}
        type_map = {'C': 'circuit', 'E': 'entity', 'N': 'token',
                    'H': 'component', 'A': 'score', 'L': 'layer',
                    'P': 'position', 'T': 'text', 'V': 'value'}
        for key, var in head_vars.items():
            bound[var] = type_map.get(var[0], 'any')

        body_literals = []

        type_prefix = {
            'circuit': 'C', 'component': 'H', 'score': 'A',
            'position': 'P', 'value': 'V', 'layer': 'L',
            'token': 'E', 'entity': 'E', 'text': 'T',
        }
        type_counter: Dict[str, int] = {}

        def _fresh_var(typ: str) -> str:
            base = type_prefix.get(typ.lower(), 'X')
            n = type_counter.get(base, 0)
            type_counter[base] = n + 1
            candidate = base if n == 0 else f'{base}{n}'
            while candidate in bound:
                type_counter[base] += 1
                n = type_counter[base]
                candidate = f'{base}{n}'
            return candidate

        def _candidates_of_type(typ: str) -> List[str]:
            """Return bound vars whose type is compatible with the requested type."""
            t = typ.lower()
            # token and entity are interchangeable in circuit facts
            compatible = {'token', 'entity'} if t in ('token', 'entity') else {t}
            return [v for v, vt in bound.items() if vt.lower() in compatible]

        for pred_name in perm:
            pred_info = body_preds[pred_name]
            arg_vals: List[str] = []
            new_bindings: List[Tuple[str, str]] = []
            ok = True

            for marker, typ in pred_info['args']:
                if marker == '+':
                    candidates = _candidates_of_type(typ)
                    if not candidates:
                        ok = False
                        break
                    # Pick most recently bound compatible var (last alphabetically)
                    arg_vals.append(sorted(candidates)[-1])
                elif marker == '-':
                    var = _fresh_var(typ)
                    arg_vals.append(var)
                    new_bindings.append((var, typ))
                elif marker == '#':
                    arg_vals.append('_')
                else:
                    arg_vals.append('_')

            if not ok:
                return None

            for var, typ in new_bindings:
                bound[var] = typ
            sep = ', '
            body_literals.append(f"{pred_name}({sep.join(arg_vals)})")

        if not body_literals:
            return None

        sep = ', '
        return f"{head_literal} :- {sep.join(body_literals)}."

    @staticmethod
    def _build_head_literal(head_info: Dict) -> Tuple[Dict[str, str], str]:
        """
        Build the head literal string and return the variable bindings it creates.

        Returns (head_vars dict, head_literal_string).
        head_vars maps arg position index to variable name.
        """
        args_out = []
        head_vars: Dict[str, str] = {}   # marker→variable for downstream use
        var_counter = [0]

        # Canonical names for first two positions
        canonical = {0: 'C', 1: 'E'}

        for i, (marker, typ) in enumerate(head_info['args']):
            if i in canonical:
                var = canonical[i]
            else:
                var = f'V{var_counter[0]}'; var_counter[0] += 1
            args_out.append(var)
            head_vars[f'arg{i}_{marker}'] = var

        pred = head_info['pred']
        literal = f"{pred}({', '.join(args_out)})"
        return head_vars, literal

    # @staticmethod
    def _generate_hypotheses(
            self,
            program: str,
            target_predicate: str,
            positive_examples: List[str],
            negative_examples: List[str]
    ) -> List[str]:
        """
        Generate candidate hypotheses — Gap 8 complete implementation.

        Three-stage pipeline (paper §3.6, Appendix B.4):

        Stage 1 — Template filtering (existing, unchanged)
            Render templates from TaskConfig, strip label predicates,
            filter by fact-database availability.

        Stage 2 — Mode validation (Gap 8, Component 2)
            Every candidate (template or generated) is checked against
            the formal mode declarations:
              (a) body predicates must be in declared modeb set
              (b) variable binding direction (+/-/#) must be consistent
              (c) body length must be ≤ max_body_literals

        Stage 3 — Mode-driven generation (Gap 8, Component 3)
            If templates pass mode validation, they form the candidate set.
            If NO templates survive, generate clause skeletons from the
            mode declarations alone (combinatorial search bounded by
            max_body_literals).  This closes the gap: the hypothesis
            language is now formally defined by the mode declarations, not
            only by what templates happen to be written by hand.

        Fallback (unchanged)
            If all else fails, emit a universal entity clause so the
            learner always returns at least one hypothesis.
        """
        label_preds: Set[str] = getattr(self.task_config, 'label_predicates', set())
        max_body    = self.config.get('max_body_literals', 4)

        # ── Shared setup ──────────────────────────────────────────────────
        clean_program = self._strip_label_predicates(program, label_preds)
        available     = self._extract_available_predicates(clean_program)
        all_hypotheses = self.task_config.get_hypotheses()

        # Gap 8 Component 1: parse mode declarations once
        raw_modes   = self.task_config.get_mode_declarations()
        parsed_modes = self._parse_mode_declarations(raw_modes)
        has_modes    = bool(parsed_modes.get('head') and parsed_modes.get('body'))

        # ── Stage 1 + 2: filter templates ────────────────────────────────
        builtins = {'_', 'true', 'fail', 'not', 'is', 'between'}
        valid: List[str] = []

        for h in all_hypotheses:
            if ':-' not in h:
                continue

            body = h.split(':-')[1]
            used_preds = set(re.findall(r'([a-z_]\w*)\(', body))

            # Reject tautological hypotheses (body = only label predicates)
            non_label = used_preds - label_preds
            if not non_label:
                continue

            # Availability filter (Stage 1): all non-label preds must be in
            # the fact DB or be activation-related grace predicates
            missing = non_label - available - builtins - label_preds
            activation_only_missing = not (missing - {'activates', 'at_layer'})
            if missing and not activation_only_missing:
                continue

            # Mode validation filter (Stage 2 — Gap 8 Component 2)
            if has_modes:
                mode_ok, reason = self._validate_clause_modes(
                    h, parsed_modes, max_body
                )
                if not mode_ok:
                    continue   # violates mode constraint — silently drop

            valid.append(h)

        # ── Stage 3: mode-driven generation if templates gave nothing ─────
        if not valid and has_modes:
            generated = self._generate_from_modes(
                parsed_modes, label_preds, available, target_predicate
            )
            # Validate generated candidates through Component 2 as well
            for h in generated:
                mode_ok, _ = self._validate_clause_modes(h, parsed_modes, max_body)
                if mode_ok:
                    valid.append(h)

        # ── Fallback: first non-tautological template ─────────────────────
        if not valid and all_hypotheses:
            for h in all_hypotheses:
                if ':-' in h:
                    body = h.split(':-')[1]
                    used = set(re.findall(r'([a-z_]\w*)\(', body))
                    if used - label_preds:
                        valid = [h]
                        break
            if not valid:
                valid = [all_hypotheses[0]]

        # ── Hard fallback: universal entity clause ────────────────────────
        if not valid:
            valid = [f"{target_predicate}(C, E) :- entity(E, _, _)."]

        return valid

    @staticmethod
    def _strip_label_predicates(program: str, label_preds: Set[str]) -> str:
        """
        Remove fact lines whose functor is a label predicate.

        Operates line-by-line: removes any line that starts with
        '<label_pred>(' so label facts are invisible to ILP search.
        Generalisable: works for any set of predicate names across any task.

        Examples removed for label_preds={'role'}:
            role(e1, 'LOCATION').     ← removed
            role(e2, 'TIME').          ← removed
            entity(e1, 'park', 3).    ← kept
        """
        if not label_preds:
            return program
        cleaned = []
        for line in program.splitlines():
            stripped = line.strip()
            functor = stripped.split('(')[0] if '(' in stripped else ''
            if functor in label_preds:
                continue
            cleaned.append(line)
        return '\n'.join(cleaned)

    # ================================================================
    # Hypothesis evaluation and selection (task-independent)
    # ================================================================

    def _select_best_hypothesis(
            self,
            hypotheses: List[str],
            program: str,
            positive_examples: List[str],
            negative_examples: List[str]
    ) -> Optional[str]:
        """
        Select best hypothesis based on coverage, precision, simplicity.

        Scoring:
            score = coverage * precision * simplicity_bonus
            simplicity_bonus = 1 / (1 + num_literals * 0.1)
        """
        best_hypothesis = None
        best_score = -1

        for hypothesis in hypotheses:
            coverage, precision = self._evaluate_hypothesis(
                hypothesis, program, positive_examples, negative_examples
            )

            num_literals = hypothesis.count(',') + 1
            simplicity = 1.0 / (1 + num_literals * 0.1)

            score = coverage * precision * simplicity

            if score > best_score:
                best_score = score
                best_hypothesis = hypothesis

        return best_hypothesis

    def _evaluate_hypothesis(
            self,
            hypothesis: str,
            program: str,
            positive_examples: List[str],
            negative_examples: List[str]
    ) -> Tuple[float, float]:
        """
        Evaluate hypothesis coverage and precision.

        Label predicates are stripped from the program before evaluation
        for the same reason they are stripped in _generate_hypotheses and
        compute_confidence: they trivially satisfy any clause that uses them,
        inflating coverage to 1.0 regardless of whether the hypothesis
        captures genuine circuit behaviour.

        Returns:
            (coverage, precision) tuple
        """
        label_preds: Set[str] = getattr(self.task_config, 'label_predicates', set())
        clean_program = self._strip_label_predicates(program, label_preds)
        test_program = clean_program + "\n" + hypothesis

        try:
            PrologString(test_program)
            use_problog = True
        except Exception:
            use_problog = False

        def _covered(pred_call: str) -> bool:
            if use_problog:
                return self._entails(test_program, pred_call)
            return self._string_match_entails(clean_program, [hypothesis], pred_call)

        explained_positives = sum(
            1 for ex in positive_examples
            if _covered(self._extract_predicate_call(ex))
        )
        explained_negatives = sum(
            1 for ex in negative_examples
            if _covered(self._extract_predicate_call(ex))
        )

        coverage = (explained_positives / len(positive_examples)
                    if positive_examples else 0.0)
        total = explained_positives + explained_negatives
        precision = explained_positives / total if total > 0 else 0.0

        return coverage, precision

    def _entails(self, program_text: str, query: str) -> bool:
        """Check if program entails query via ProbLog evaluation.

        Args:
            program_text: Complete Prolog program as string
            query: Query to check (e.g. "time_entity(c1, e1)")
        """
        try:
            full = program_text + f"\nquery({query})."
            result = get_evaluatable().create_from(PrologString(full))
            results = result.evaluate()
            return len(results) > 0 and any(
                prob > 0.5 for prob in results.values()
            )
        except Exception:
            return False

    # ================================================================
    # Signature and confidence (task-independent)
    # ================================================================

    def _clause_depth(self, body: str) -> int:
        """Return max parenthesis nesting depth in a clause body (= max term depth)."""
        max_depth = current = 0
        for ch in body:
            if ch == '(':
                current += 1
                if current > max_depth:
                    max_depth = current
            elif ch == ')':
                current -= 1
        return max_depth if max_depth > 0 else 1  # floor at 1 for any non-empty body

    def compute_signature(self, clauses: List[str]) -> Dict:
        """
        Compute S_ILP = <n_clauses, d_bar, P_H, l_bar, complexity> (§2.3).
        """
        n_clauses = len(clauses)
        depths = []
        lengths = []
        all_predicates = set()

        for clause in clauses:
            if ':-' in clause:
                _, body = clause.split(':-', 1)
                body = body.strip().rstrip('.')

                literals = [lit.strip() for lit in body.split(',')]
                length = len(literals)
                lengths.append(length)

                depth = self._clause_depth(body)
                depths.append(depth)

                for literal in literals:
                    pred_name = literal.split('(')[0].strip()
                    if pred_name:
                        all_predicates.add(pred_name)
            else:
                lengths.append(0)
                depths.append(0)

        avg_depth = sum(depths) / len(depths) if depths else 0
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        complexity = (
            n_clauses * avg_depth * avg_length / 10
            if avg_length > 0 else 0
        )

        return {
            'n_clauses': n_clauses,
            'avg_depth': avg_depth,
            'predicates': sorted(all_predicates),
            'avg_length': avg_length,
            'complexity': complexity,
        }

    def compute_confidence(
            self,
            clauses: List[str],
            facts: str,
            positive_examples: List[str],
            negative_examples: List[str]
    ) -> float:
        """
        Compute learning confidence in [0,1] (§2.4).

        confidence = (coverage + precision) / 2 * simplicity_penalty
        simplicity_penalty = 1 / (1 + log(complexity + 1))

        ProbLog path: full logical entailment via _entails().
        Fallback path: string-matching coverage when ProbLog cannot parse
          the fact database (e.g. too large, unsupported syntax).
          Generalises across tasks — no predicate names are hardcoded.
        """
        label_preds: Set[str] = getattr(self.task_config, 'label_predicates', set())
        clean_facts = self._strip_label_predicates(facts, label_preds)
        program = clean_facts + "\n" + "\n".join(clauses)

        try:
            PrologString(program)
            use_problog = True
        except Exception:
            use_problog = False

        def _covered(pred_call: str) -> bool:
            if use_problog:
                return self._entails(program, pred_call)
            return self._string_match_entails(clean_facts, clauses, pred_call)

        explained_pos = sum(1 for ex in positive_examples
                            if _covered(self._extract_predicate_call(ex)))
        explained_neg = sum(1 for ex in negative_examples
                            if _covered(self._extract_predicate_call(ex)))

        # Gap 3: pure Eq.2 = tp / |Ep|
        confidence_eq2 = (explained_pos / len(positive_examples)
                          if positive_examples else 0.0)
        total = explained_pos + explained_neg
        precision = explained_pos / total if total > 0 else 0.0

        signature = self.compute_signature(clauses)
        complexity = signature['complexity']
        simplicity_penalty = 1.0 / (1.0 + math.log(complexity + 1))

        confidence_adj = ((confidence_eq2 + precision) / 2) * simplicity_penalty
        confidence_adj = max(0.0, min(1.0, confidence_adj))

        self._last_confidence_eq2 = confidence_eq2
        self._last_confidence_adj = confidence_adj

        if not use_problog:
            print("  Note: ProbLog parse failed — using string-match coverage fallback")

        return confidence_adj

    def _string_match_entails(
            self,
            facts: str,
            clauses: List[str],
            query: str,
    ) -> bool:
        """
        Fallback coverage check when ProbLog cannot parse the program.

        Checks whether a query is satisfied by shallow variable unification
        against clause heads, then verifying each body literal appears as a
        ground fact in the fact text.  Generalises across tasks — no
        predicate names are hardcoded.
        """
        query_stripped = query.strip().rstrip('.')

        # Ground fact — fastest path
        if query_stripped + '.' in facts:
            return True

        for clause in clauses:
            if ':-' not in clause:
                head = clause.strip().rstrip('.')
                if head == query_stripped:
                    return True
                continue

            head_str, body_str = clause.split(':-', 1)
            bindings = self._match_head(head_str.strip().rstrip('.'), query_stripped)
            if bindings is None:
                continue

            literals = [l.strip() for l in body_str.strip().rstrip('.').split(',')
                        if l.strip()]
            if all(self._apply_bindings(lit, bindings) + '.' in facts
                   for lit in literals):
                return True

        return False

    @staticmethod
    def _match_head(head: str, query: str) -> Optional[Dict[str, str]]:
        """Shallow unification of clause head against query. Returns bindings or None."""
        def _parse(t):
            m = re.match(r'(\w+)\((.+)\)$', t.strip())
            return (m.group(1), [a.strip() for a in m.group(2).split(',')]) if m else (None, [])

        h_func, h_args = _parse(head)
        q_func, q_args = _parse(query)
        if h_func != q_func or len(h_args) != len(q_args):
            return None

        bindings: Dict[str, str] = {}
        for h_arg, q_arg in zip(h_args, q_args):
            if h_arg[0].isupper() or h_arg == '_':
                if h_arg in bindings and bindings[h_arg] != q_arg:
                    return None
                if h_arg != '_':
                    bindings[h_arg] = q_arg
            elif h_arg != q_arg:
                return None
        return bindings

    @staticmethod
    def _apply_bindings(literal: str, bindings: Dict[str, str]) -> str:
        """Substitute variable bindings into a literal string."""
        result = literal
        for var, val in sorted(bindings.items(), key=lambda x: -len(x[0])):
            result = re.sub(rf'\b{re.escape(var)}\b', val, result)
        return result

    # ================================================================
    # Helpers
    # ================================================================

    def _learn_weights_lfi(
            self,
            hypotheses: List[str],
            facts: str,
            positive_examples: List[str],
            negative_examples: List[str],
    ) -> Dict[str, float]:
        """
        Gap 5: Learn probabilistic weights λ for candidate hypotheses via ProbLog LFI.

        Each hypothesis H_i in the candidate set is annotated as:
            t(h_i)::head(C,E) :- body_i(C,E).

        LFI runs Expectation-Maximisation to learn λ_i = P(H_i | observations).
        The learned λ vector is H = (L, λ) from paper §3.2 / Eqs. 13-15.

        Returns
        -------
        Dict mapping hypothesis index str -> learned weight float
        E.g. {"h0": 0.97, "h1": 0.12, "h2": 0.03}
        """
        if not hypotheses:
            return {}

        # -- Build annotated source program --------------------------------
        annotated_clauses = []
        for i, hyp in enumerate(hypotheses):
            if ":-" in hyp:
                head, body = hyp.split(":-", 1)
                head = head.strip().rstrip(".")
                body = body.strip().rstrip(".")
                # t(h_i) annotation marks this clause weight as learnable
                annotated_clauses.append(
                    f"t(h{i})::{head} :- {body}."
                )
            else:
                # Fact hypothesis  -  just annotate the fact
                fact = hyp.strip().rstrip(".")
                annotated_clauses.append(f"t(h{i})::{fact}.")

        # Background facts (no t(_) annotation  -  these are fixed knowledge)
        source_program = facts + "\n\n" + "\n".join(annotated_clauses)

        # -- Convert examples to LFI format: list[list[tuple(Term, bool)]] -
        from problog.logic import Term as ProbTerm
        lfi_examples: List[List[Tuple]] = []

        def _parse_term(ex_str: str):
            """Extract Term from 'positive(pred(c1,e1)).' string."""
            inner = self._extract_predicate_call(ex_str)
            # Parse predicate name and args
            m = re.match(r'(\w+)\((.*)\)$', inner.strip())
            if not m:
                return None
            pred = m.group(1)
            args_str = m.group(2)
            # Simple arg parser: split on top-level commas
            args = []
            depth = 0
            current = ""
            for ch in args_str:
                if ch == '(': depth += 1
                elif ch == ')': depth -= 1
                if ch == ',' and depth == 0:
                    args.append(current.strip())
                    current = ""
                else:
                    current += ch
            if current.strip():
                args.append(current.strip())
            problog_args = [ProbTerm(a.strip("'\"")) if not a.strip().lstrip('-').isdigit()
                           else ProbTerm(a) for a in args]
            return ProbTerm(pred, *problog_args)

        for pos_ex in positive_examples:
            t = _parse_term(pos_ex)
            if t is not None:
                lfi_examples.append([(t, True)])

        for neg_ex in negative_examples:
            t = _parse_term(neg_ex)
            if t is not None:
                lfi_examples.append([(t, False)])

        if not lfi_examples:
            return {f"h{i}": 0.5 for i in range(len(hypotheses))}

        # -- Run LFI -------------------------------------------------------
        weights: Dict[str, float] = {}
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.pl', delete=False
            ) as tmp:
                tmp.write(source_program)
                tmp_path = tmp.name

            lfi_prob = LFIProblem(
                PrologFile(tmp_path),
                lfi_examples,
                max_iter   = 200,
                min_improv = 1e-6,
                verbose    = 0,
                leakprob   = 1e-6,   # avoid zero-probability inconsistency
            )
            lfi_prob.run()
            model_str = lfi_prob.get_model()

            # Extract weights from learned model: "0.97::head(...) :- ..."
            for i in range(len(hypotheses)):
                m = re.search(
                    rf'([\d.]+(?:e[+-]?\d+)?)::{re.escape(hypotheses[i].split(":-")[0].strip().rstrip("."))}',
                    model_str
                )
                weights[f"h{i}"] = float(m.group(1)) if m else 0.0

        except Exception as e:
            print(f"  LFI warning (falling back to uniform weights): {e}")
            # Fall back to uniform weights — does not break binary hypothesis selection,
            # but callers should check lfi_available=False before trusting lambda values.
            weights = {f"h{i}": 0.5 for i in range(len(hypotheses))}
            weights["__lfi_available__"] = False   # sentinel: LFI did not converge
        else:
            weights["__lfi_available__"] = True    # sentinel: real EM weights
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return weights

    def _build_program(
            self,
            facts: str,
            positive_examples: List[str],
            negative_examples: List[str],
            background_knowledge: Optional[str]
    ) -> str:
        """Build complete Prolog program."""
        parts = [facts]
        if background_knowledge:
            parts.append(background_knowledge)
        parts.append("\n% Examples")
        parts.extend(positive_examples)
        parts.extend(negative_examples)
        return "\n".join(parts)

    def _extract_target_predicate(self, positive_examples: List[str]) -> str:
        """Extract target predicate from positive(pred(...))."""
        if not positive_examples:
            return self.task_config.target_predicate

        match = re.search(
            r'(?:positive|negative)\((\w+)\(', positive_examples[0]
        )
        return match.group(1) if match else self.task_config.target_predicate

    def _extract_predicate_call(self, example: str) -> str:
        """Extract inner predicate call from positive(pred(...)). """
        match = re.search(r'(?:positive|negative)\((.*)\)', example)
        if match:
            return match.group(1).rstrip('.')
        return example

    def _extract_available_predicates(self, program: str) -> Set[str]:
        """Extract all predicate names from program text."""
        return set(re.findall(r'(\w+)\(', program))

    def _extract_predicates_from_rule(self, rule: str) -> List[str]:
        """Extract body predicates from a rule."""
        predicates = set()
        if ':-' in rule:
            _, body = rule.split(':-')
            matches = re.findall(r'([a-z_]\w*)\(', body)
            predicates.update(matches)
        return sorted(predicates)

    def _create_fallback_hypothesis(self, target_predicate: str) -> str:
        """
        Fallback hypothesis from task config (first template).
        No hardcoded role maps.
        """
        hypotheses = self.task_config.get_hypotheses()
        if hypotheses:
            return hypotheses[0]
        return f"{target_predicate}(C, E) :- entity(E, _, _)."

    def _default_config(self) -> Dict:
        """Default ILP configuration."""
        return {
            'max_depth':         3,
            'max_clauses':       5,
            'max_body_literals': 4,   # Gap 8: clause length bound (paper §3.6)
            'min_coverage':      0.5,
            'min_precision':     0.5,
            'timeout':           300,
        }


# ============================================================
# Convenience Function
# ============================================================

def learn_from_facts(
        facts_file: str,
        positive_examples: List[str],
        negative_examples: List[str],
        task_config: Optional[TaskConfig] = None
) -> Dict:
    """
    Convenience function to learn from a facts file.

    Args:
        facts_file: Path to .pl file with facts
        positive_examples: Positive examples
        negative_examples: Negative examples
        task_config: TaskConfig (required for general use;
            falls back to time config if not provided)

    Returns:
        Learned specification dict
    """
    if task_config is None:
        from .task_config import TIME_CONFIG
        task_config = TIME_CONFIG

    with open(facts_file, 'r') as f:
        facts = f.read()

    learner = SemanticSpecificationLearner(task_config)
    return learner.learn_specification(
        facts=facts,
        positive_examples=positive_examples,
        negative_examples=negative_examples,
    )
