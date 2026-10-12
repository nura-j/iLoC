"""
θ-Subsumption — Paper §3.6 / Appendix F

Clause γ₁ θ-subsumes γ₂ (γ₁ ⪯_θ γ₂) if there exists a substitution θ
such that every literal in γ₁θ appears in γ₂.

Mechanistic interpretations (Appendix F.1):
    - γ₁ ≺_θ γ₂ (strict): γ₁ refines γ₂ — stronger structural commitments
    - γ₁ ≡_θ γ₂ (mutual): equivalent — characterise the same circuits
    - incomparable: neither generalises the other

Tractable in our setting because:
    - Bounded clause length (l̄ ≤ 5)
    - Datalog-range predicates (all variables in positive literals)

Example from paper (Appendix F.3):
    γ_loc: arch_location(C) ← has_motif(C, attn_chain_3), component_ratio(C, attn, R), R > 0.76.
    γ_time: arch_time(C) ← component_ratio(C, attn, R), R > 0.76.

    γ_time ⪯_θ γ_loc  (time subsumes location — time is MORE GENERAL)
    γ_loc ⋠_θ γ_time   (location does NOT subsume time)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Literal:
    """A single literal: predicate(arg1, arg2, ...)."""
    predicate: str
    args: Tuple[str, ...]

    def __hash__(self):
        return hash((self.predicate, self.args))

    def __eq__(self, other):
        return self.predicate == other.predicate and self.args == other.args

    def __repr__(self):
        return f"{self.predicate}({', '.join(self.args)})"

    def apply(self, subst: Dict[str, str]) -> "Literal":
        """Apply a substitution θ to this literal."""
        new_args = tuple(subst.get(a, a) for a in self.args)
        return Literal(self.predicate, new_args)

    def is_ground(self) -> bool:
        """True if all arguments are constants (no variables)."""
        return all(not _is_variable(a) for a in self.args)


@dataclass
class Clause:
    """A Horn clause: head :- body_literal_1, body_literal_2, ..."""
    head: Literal
    body: List[Literal]

    def __repr__(self):
        if not self.body:
            return f"{self.head}."
        body_str = ", ".join(str(lit) for lit in self.body)
        return f"{self.head} :- {body_str}."

    def body_predicates(self) -> Set[str]:
        """Set of predicate names in the body."""
        return {lit.predicate for lit in self.body}

    def variables(self) -> Set[str]:
        """All variables in the clause."""
        vs = set()
        for lit in [self.head] + self.body:
            for a in lit.args:
                if _is_variable(a):
                    vs.add(a)
        return vs


def _is_variable(token: str) -> bool:
    """Variables start with uppercase or underscore in Prolog."""
    return bool(token) and (token[0].isupper() or token[0] == "_")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_clause(text: str) -> Clause:
    """
    Parse a Prolog-style clause string into a Clause object.

    Handles:
        arch_location(C) :- has_motif(C, attn_chain_3), component_ratio(C, attn, R), R > 0.76.
    """
    text = text.strip().rstrip(".")

    if ":-" in text:
        head_str, body_str = text.split(":-", 1)
    else:
        head_str = text
        body_str = ""

    head = _parse_literal(head_str.strip())
    body = _parse_body(body_str.strip()) if body_str.strip() else []

    return Clause(head=head, body=body)


def _parse_literal(text: str) -> Literal:
    """Parse 'pred(arg1, arg2)' into a Literal."""
    text = text.strip()
    match = re.match(r"(\w+)\((.+)\)$", text)
    if match:
        pred = match.group(1)
        args_str = match.group(2)
        args = _split_args(args_str)
        return Literal(pred, tuple(args))
    # Comparison operators: R > 0.76 → comparison(R, >, 0.76)
    comp_match = re.match(r"(\w+)\s*(>=|=<|>|<|==|=)\s*(.+)$", text)
    if comp_match:
        return Literal("comparison", (comp_match.group(1), comp_match.group(2), comp_match.group(3).strip()))
    # Fallback: treat as nullary predicate
    return Literal(text, ())


def _split_args(args_str: str) -> List[str]:
    """Split 'C, attn, R' respecting nested parens."""
    args = []
    depth = 0
    current = ""
    for ch in args_str:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        args.append(current.strip())
    return args


def _parse_body(body_str: str) -> List[Literal]:
    """Parse comma-separated body literals, respecting parens."""
    parts = []
    depth = 0
    current = ""
    for ch in body_str:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return [_parse_literal(p) for p in parts if p]


# ---------------------------------------------------------------------------
# θ-Subsumption
# ---------------------------------------------------------------------------

def theta_subsumes(gamma1: Clause, gamma2: Clause) -> bool:
    """
    Check if γ₁ ⪯_θ γ₂: does γ₁ subsume γ₂?

    γ₁ subsumes γ₂ if there exists a substitution θ such that
    every literal in γ₁θ appears in γ₂ (treating clauses as sets of literals).

    γ₁ is at least as general as γ₂.
    """
    # Every body literal of γ₁ must be matchable into γ₂'s body
    return _find_substitution(gamma1.body, gamma2.body, {}) is not None


def _find_substitution(
    remaining: List[Literal],
    target: List[Literal],
    subst: Dict[str, str],
) -> Optional[Dict[str, str]]:
    """
    Recursive backtracking search for a substitution θ mapping
    all literals in `remaining` into `target`.
    """
    if not remaining:
        return subst  # all literals matched

    lit = remaining[0]
    rest = remaining[1:]

    # Apply current substitution to this literal
    lit_subst = lit.apply(subst)

    # Try to match against each literal in the target
    for t_lit in target:
        new_subst = _unify_literal(lit_subst, t_lit, dict(subst))
        if new_subst is not None:
            result = _find_substitution(rest, target, new_subst)
            if result is not None:
                return result

    return None  # no match found


def _unify_literal(
    source: Literal,
    target: Literal,
    subst: Dict[str, str],
) -> Optional[Dict[str, str]]:
    """
    Try to unify source literal with target literal under substitution.
    Returns extended substitution if successful, None otherwise.
    """
    if source.predicate != target.predicate:
        return None
    if len(source.args) != len(target.args):
        return None

    new_subst = dict(subst)
    for s_arg, t_arg in zip(source.args, target.args):
        if _is_variable(s_arg):
            # Variable: check if already bound
            if s_arg in new_subst:
                if new_subst[s_arg] != t_arg:
                    return None  # conflicting binding
            else:
                new_subst[s_arg] = t_arg
        else:
            # Constant: must match exactly
            if s_arg != t_arg:
                return None

    return new_subst


# ---------------------------------------------------------------------------
# High-level comparison
# ---------------------------------------------------------------------------

def compare_clauses(text1: str, text2: str) -> str:
    """
    Compare two clause strings and return their relationship.

    Returns one of:
        "equivalent"    — mutual subsumption (γ₁ ≡_θ γ₂)
        "generalises"   — γ₁ ⪯_θ γ₂ (γ₁ is more general)
        "specialises"   — γ₂ ⪯_θ γ₁ (γ₂ is more general, γ₁ refines)
        "incomparable"  — neither subsumes the other
    """
    c1 = parse_clause(text1)
    c2 = parse_clause(text2)

    c1_subsumes_c2 = theta_subsumes(c1, c2)
    c2_subsumes_c1 = theta_subsumes(c2, c1)

    if c1_subsumes_c2 and c2_subsumes_c1:
        return "equivalent"
    elif c1_subsumes_c2:
        return "generalises"
    elif c2_subsumes_c1:
        return "specialises"
    else:
        return "incomparable"


def subsumption_matrix(clauses: Dict[str, str], relaxed: bool = False) -> Dict[str, Dict[str, str]]:
    """
    Compute pairwise subsumption relationships.

    Args:
        clauses: {name: clause_text}
        relaxed: if True, use relaxed subsumption (numeric constants as wildcards)

    Returns:
        {name1: {name2: relationship}}
    """
    names = sorted(clauses.keys())
    cmp = compare_clauses_relaxed if relaxed else compare_clauses
    matrix = {}
    for n1 in names:
        matrix[n1] = {}
        for n2 in names:
            if n1 == n2:
                matrix[n1][n2] = "self"
            else:
                matrix[n1][n2] = cmp(clauses[n1], clauses[n2])
    return matrix


# ---------------------------------------------------------------------------
# Relaxed θ-Subsumption (numeric constants treated as wildcards)
# ---------------------------------------------------------------------------

def _is_numeric(token: str) -> bool:
    """Check if token is a numeric constant (int or float)."""
    try:
        float(token)
        return True
    except (ValueError, TypeError):
        return False


def _normalise_clause(clause: Clause) -> Clause:
    """
    Replace all numeric constants with a fresh variable _Num_i.
    This makes subsumption ignore threshold differences:
    R > 0.63 and R > 0.38 both become R > _Num_0.
    """
    counter = [0]
    seen: Dict[str, str] = {}

    def _norm_literal(lit: Literal) -> Literal:
        new_args = []
        for a in lit.args:
            if _is_numeric(a):
                if a not in seen:
                    seen[a] = f"_Num{counter[0]}"
                    counter[0] += 1
                new_args.append(seen[a])
            else:
                new_args.append(a)
        return Literal(lit.predicate, tuple(new_args))

    return Clause(
        head=_norm_literal(clause.head),
        body=[_norm_literal(lit) for lit in clause.body],
    )


def theta_subsumes_relaxed(gamma1: Clause, gamma2: Clause) -> bool:
    """
    Relaxed θ-subsumption: numeric thresholds are treated as variables.

    arch_ioi(C) :- component_ratio(C, attention, R), R > 0.63.
    arch_loc(C) :- ..., component_ratio(C, attention, R), R > 0.38, ...

    Under strict subsumption these are incomparable (0.63 ≠ 0.38).
    Under relaxed subsumption, IOI subsumes Location because every
    predicate structure in IOI's body appears in Location's body
    (the threshold value is treated as a wildcard).
    """
    g1_norm = _normalise_clause(gamma1)
    g2_norm = _normalise_clause(gamma2)
    return _find_substitution(g1_norm.body, g2_norm.body, {}) is not None


def compare_clauses_relaxed(text1: str, text2: str) -> str:
    """Like compare_clauses but with relaxed numeric matching."""
    c1 = parse_clause(text1)
    c2 = parse_clause(text2)

    c1_sub_c2 = theta_subsumes_relaxed(c1, c2)
    c2_sub_c1 = theta_subsumes_relaxed(c2, c1)

    if c1_sub_c2 and c2_sub_c1:
        return "equivalent"
    elif c1_sub_c2:
        return "generalises"
    elif c2_sub_c1:
        return "specialises"
    else:
        return "incomparable"


# ---------------------------------------------------------------------------
# Predicate-structure comparison (softest level)
# ---------------------------------------------------------------------------

def predicate_structure_overlap(clauses: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """
    Compute pairwise predicate-set Jaccard over clause bodies.

    Ignores arguments entirely — only compares which predicate
    names appear. This is the softest comparison: two clauses
    using the same predicates with different constants score 1.0.

    Returns {name1: {name2: jaccard}}.
    """
    import re
    pred_sets: Dict[str, Set[str]] = {}
    for name, clause in clauses.items():
        body = clause.split(":-")[1] if ":-" in clause else ""
        preds = set(re.findall(r"(\w+)\(", body))
        # Also capture comparison operators as a predicate
        if re.search(r"[A-Z]\w*\s*(?:>=|=<|>|<)", body):
            preds.add("comparison")
        pred_sets[name] = preds

    names = sorted(pred_sets.keys())
    matrix: Dict[str, Dict[str, float]] = {}
    for n1 in names:
        matrix[n1] = {}
        for n2 in names:
            if n1 == n2:
                matrix[n1][n2] = 1.0
            else:
                s1, s2 = pred_sets[n1], pred_sets[n2]
                union = s1 | s2
                matrix[n1][n2] = round(len(s1 & s2) / len(union), 3) if union else 0.0
    return matrix
