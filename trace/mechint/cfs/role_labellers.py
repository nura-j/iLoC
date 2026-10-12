"""
role_labellers.py
=================
Task-aware token role labellers for CFS attention profiles.

Two levels of labelling:
  1. Linguistic (shared): subject, object, verb, function, entity, other
     — for cross-task comparison (RQ2)
  2. Task-specific: role_filler, scaffold, repeated_name, first_number, etc.
     — for within-task characterisation (RQ1)

Each labeller is a callable: (text, token_strs, example_meta) → List[str]
where example_meta is the dict from the contrastive pair CSV row.

Registry maps task_family → (linguistic_roles, task_roles, task_labeller).
"""

import re
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
# Labeller takes (text, token_strings, optional_metadata) → role list
TaskLabeller = Callable[[str, List[str], Optional[Dict]], List[str]]


# ---------------------------------------------------------------------------
# Linguistic roles — two granularities
# ---------------------------------------------------------------------------
# Coarse: 6 roles (used for cross-task comparison)
LINGUISTIC_ROLES = ("subject", "object", "verb", "function", "entity", "other")

# Fine: 12 roles (used for detailed linguistic analysis)
LINGUISTIC_ROLES_FINE = (
    "subject_agent",      # nsubj of active verb
    "subject_experiencer",# nsubj of stative/psych verb
    "object_patient",     # dobj of transitive verb
    "object_theme",       # pobj / indirect object
    "verb_action",        # dynamic verb (put, gave, ran)
    "verb_stative",       # stative verb (is, was, lasted)
    "function_prep",      # prepositions (in, at, from, to)
    "function_det",       # determiners (the, a, an)
    "function_aux",       # auxiliaries (was, were, did)
    "function_conj",      # conjunctions (and, or, but)
    "entity",             # named entities
    "other",              # adjectives, adverbs, modifiers
)

FUNCTION_WORDS = {
    "the", "a", "an", "to", "from", "in", "at", "on", "of", "for",
    "with", "by", "and", "or", "but", "that", "this", "is", "was",
    "were", "are", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "not", "no", "nor", "into", "through", "near", "inside", "about",
    "than", "then", "so", "if", "as", "up", "out", "off", "over",
}


def linguistic_labeller(text: str, token_strs: List[str],
                        meta: Optional[Dict] = None) -> List[str]:
    """
    Enhanced linguistic labeller: subject, object, verb, function, entity, other.

    Uses spaCy if available, with function word detection added.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)

        word_roles = []
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                word_roles.append("subject")
            elif token.dep_ in ("dobj", "pobj", "iobj", "dative"):
                word_roles.append("object")
            elif token.pos_ == "VERB":
                word_roles.append("verb")
            elif token.ent_type_:
                word_roles.append("entity")
            elif token.text.lower() in FUNCTION_WORDS or token.pos_ in ("DET", "ADP", "CCONJ", "SCONJ", "AUX", "PART"):
                word_roles.append("function")
            else:
                word_roles.append("other")

        # Align to subword tokens
        aligned = _align_to_tokens(text, word_roles, token_strs, nlp)
        if aligned:
            return aligned
    except (ImportError, OSError):
        pass

    # Fallback: function word detection + capitalisation heuristic
    return _heuristic_linguistic(token_strs)


# Stative verbs for fine-grained verb classification
_STATIVE_VERBS = {
    "is", "was", "were", "are", "been", "being", "has", "had", "have",
    "lasted", "remained", "stayed", "seemed", "appeared", "existed",
    "knew", "believed", "wanted", "liked", "loved", "hated",
}


def linguistic_labeller_fine(text: str, token_strs: List[str],
                             meta: Optional[Dict] = None) -> List[str]:
    """
    Fine-grained linguistic labeller: 12 roles.

    Distinguishes subject_agent from subject_experiencer,
    object_patient from object_theme, verb_action from verb_stative,
    and function subtypes (prep, det, aux, conj).
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)

        word_roles = []
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                # Distinguish agent (active transitive) from experiencer (stative/psych)
                head_lemma = token.head.lemma_.lower() if token.head else ""
                if head_lemma in _STATIVE_VERBS or token.head.pos_ == "ADJ":
                    word_roles.append("subject_experiencer")
                else:
                    word_roles.append("subject_agent")
            elif token.dep_ == "dobj":
                word_roles.append("object_patient")
            elif token.dep_ in ("pobj", "iobj", "dative"):
                word_roles.append("object_theme")
            elif token.pos_ == "VERB":
                if token.lemma_.lower() in _STATIVE_VERBS:
                    word_roles.append("verb_stative")
                else:
                    word_roles.append("verb_action")
            elif token.ent_type_:
                word_roles.append("entity")
            elif token.pos_ == "ADP":
                word_roles.append("function_prep")
            elif token.pos_ == "DET":
                word_roles.append("function_det")
            elif token.pos_ == "AUX":
                word_roles.append("function_aux")
            elif token.pos_ in ("CCONJ", "SCONJ"):
                word_roles.append("function_conj")
            elif token.text.lower() in FUNCTION_WORDS:
                word_roles.append("function_prep")  # fallback for unlisted function words
            else:
                word_roles.append("other")

        aligned = _align_to_tokens(text, word_roles, token_strs, nlp)
        if aligned:
            return aligned
    except (ImportError, OSError):
        pass

    return _heuristic_linguistic(token_strs)


def _heuristic_linguistic(token_strs: List[str]) -> List[str]:
    """Fallback linguistic labeller without spaCy."""
    roles = []
    for tok in token_strs:
        clean = tok.replace("Ġ", "").replace("▁", "").strip().lower()
        if clean in FUNCTION_WORDS:
            roles.append("function")
        elif clean and clean[0].isupper():
            roles.append("entity")
        else:
            roles.append("other")
    return roles


def _align_to_tokens(text, word_roles, token_strs, nlp):
    """Align word-level roles to subword tokens."""
    try:
        doc = nlp(text)
        char_roles = ["other"] * len(text)
        for token, role in zip(doc, word_roles):
            for i in range(token.idx, token.idx + len(token.text)):
                if i < len(text):
                    char_roles[i] = role

        aligned = []
        pos = 0
        for tok_str in token_strs:
            clean = tok_str.replace("Ġ", " ").replace("▁", " ").lstrip()
            idx = text.find(clean, pos) if clean else pos
            if idx == -1:
                idx = pos
            role = char_roles[min(idx, len(char_roles) - 1)]
            aligned.append(role)
            pos = idx + max(len(clean), 1)
        return aligned
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Task-specific labellers
# ---------------------------------------------------------------------------

# -- Semantic Role Binding --
SEMROLE_TASK_ROLES = (
    "role_filler",       # the entity being bound (target word)
    "scaffold_prep",     # the preposition in the scaffold (in, at, from, through)
    "scaffold_det",      # the determiner in the scaffold (the, a)
    "predicate_verb",    # the main verb of the frame
    "agent",             # the subject/agent noun
    "theme",             # the direct object/theme noun
    "modifier",          # adjectives, adverbs modifying agent/theme
    "other",
)

SCAFFOLD_PREPS = {
    "in", "at", "near", "inside", "to", "for", "from", "with",
    "using", "about", "of", "through", "into", "on",
}

SCAFFOLD_PHRASES = {
    "in the", "at the", "near the", "inside the", "to the", "for the",
    "from the", "with a", "with the", "using a", "about the", "of the",
    "through the", "into the", "at", "on",
}


def semrole_labeller(text: str, token_strs: List[str],
                     meta: Optional[Dict] = None) -> List[str]:
    """
    Fine-grained semantic role labeller.

    Distinguishes scaffold prepositions from scaffold determiners,
    marks the predicate verb, agent, theme, and modifiers.
    """
    roles = ["other"] * len(token_strs)
    text_lower = text.lower()

    # Mark scaffold phrases — split into prep and det components
    for phrase in SCAFFOLD_PHRASES:
        idx = text_lower.rfind(phrase)
        if idx != -1:
            parts = phrase.split()
            pos = idx
            for part in parts:
                p_idx = text_lower.find(part, pos)
                if p_idx != -1:
                    if part in SCAFFOLD_PREPS:
                        _mark_span(text, p_idx, p_idx + len(part),
                                   "scaffold_prep", token_strs, roles)
                    elif part in ("the", "a", "an"):
                        _mark_span(text, p_idx, p_idx + len(part),
                                   "scaffold_det", token_strs, roles)
                    pos = p_idx + len(part)

    # Mark target word (role filler) from metadata
    if meta:
        target = meta.get("target_word", "")
        if target:
            idx = text.rfind(target)
            if idx != -1:
                _mark_span(text, idx, idx + len(target), "role_filler",
                           token_strs, roles)

    # Use spaCy for verb/agent/theme if available
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        for token in doc:
            if token.pos_ == "VERB" and roles[0] != "scaffold_prep":
                _mark_span(text, token.idx, token.idx + len(token.text),
                           "predicate_verb", token_strs, roles)
            elif token.dep_ in ("nsubj", "nsubjpass"):
                _mark_span(text, token.idx, token.idx + len(token.text),
                           "agent", token_strs, roles)
            elif token.dep_ == "dobj":
                _mark_span(text, token.idx, token.idx + len(token.text),
                           "theme", token_strs, roles)
            elif token.pos_ in ("ADJ", "ADV") and token.dep_ in ("amod", "advmod"):
                _mark_span(text, token.idx, token.idx + len(token.text),
                           "modifier", token_strs, roles)
    except (ImportError, OSError):
        pass

    return roles


# -- IOI --
IOI_TASK_ROLES = (
    "repeated_name",       # the name appearing twice (target for copying)
    "first_occurrence",    # first mention of the repeated name
    "distractor_name",     # the other name (foil)
    "io_position",         # the "to" before the indirect object slot
    "context_verb",        # the main verb (gave, sent, etc.)
    "context_setting",     # setting clause ("were working at the house")
    "other",
)


def ioi_labeller(text: str, token_strs: List[str],
                 meta: Optional[Dict] = None) -> List[str]:
    """
    IOI task-specific labeller.

    Identifies repeated names, first occurrences, and the indirect object
    position from the contrastive pair structure.
    """
    roles = ["other"] * len(token_strs)

    if meta:
        target = meta.get("target_word", "").strip()
        incorrect = meta.get("incorrect_word", "").strip()

        if target:
            # The target word is the indirect object (what the model predicts)
            # In IOI, the target appears earlier in the sentence too (repeated)
            occurrences = [m.start() for m in re.finditer(re.escape(target), text)]
            if len(occurrences) >= 2:
                _mark_span(text, occurrences[0], occurrences[0] + len(target),
                           "first_occurrence", token_strs, roles)
                _mark_span(text, occurrences[1], occurrences[1] + len(target),
                           "repeated_name", token_strs, roles)
            elif len(occurrences) == 1:
                _mark_span(text, occurrences[0], occurrences[0] + len(target),
                           "repeated_name", token_strs, roles)

        if incorrect:
            idx = text.find(incorrect)
            if idx != -1:
                _mark_span(text, idx, idx + len(incorrect),
                           "distractor_name", token_strs, roles)

    # Mark "to" before indirect object position
    to_match = re.search(r'\bto\s*$', text)
    if to_match:
        _mark_span(text, to_match.start(), to_match.end(),
                   "io_position", token_strs, roles)

    # Mark main verb (gave, sent, decided)
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                _mark_span(text, token.idx, token.idx + len(token.text),
                           "context_verb", token_strs, roles)
    except (ImportError, OSError):
        pass

    return roles


# -- Greater-Than --
GT_TASK_ROLES = (
    "first_number",         # the four-digit year (1352)
    "second_number",        # the two-digit continuation (13)
    "comparison_from",      # "from" keyword
    "comparison_to",        # "to" keyword
    "year_marker",          # "the year" (structural frame)
    "subject_noun",         # "The war", "The plan" etc.
    "predicate_verb",       # "lasted"
    "other",
)


def gt_labeller(text: str, token_strs: List[str],
                meta: Optional[Dict] = None) -> List[str]:
    """
    Fine-grained Greater-Than labeller.

    Distinguishes from/to keywords, year markers, subject noun, predicate verb.
    Pattern: "The X lasted from the year YYYY to the year YY"
    """
    roles = ["other"] * len(token_strs)

    # First number (four-digit year)
    year_matches = list(re.finditer(r'\b1[0-9]{3}\b', text))
    if year_matches:
        m = year_matches[0]
        _mark_span(text, m.start(), m.end(), "first_number", token_strs, roles)

    # Second number (two-digit continuation at end)
    end_match = re.search(r'the year (\d{2})$', text)
    if end_match:
        _mark_span(text, end_match.start(1), end_match.end(1),
                   "second_number", token_strs, roles)

    # "from" and "to" as distinct comparison markers
    from_match = re.search(r'\bfrom\b', text)
    if from_match:
        _mark_span(text, from_match.start(), from_match.end(),
                   "comparison_from", token_strs, roles)
    to_match = re.search(r'\bto\b', text)
    if to_match:
        _mark_span(text, to_match.start(), to_match.end(),
                   "comparison_to", token_strs, roles)

    # "the year" phrases as structural frame markers
    for ym in re.finditer(r'\bthe year\b', text):
        _mark_span(text, ym.start(), ym.end(),
                   "year_marker", token_strs, roles)

    # Subject noun and predicate verb via spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                _mark_span(text, token.idx, token.idx + len(token.text),
                           "subject_noun", token_strs, roles)
            elif token.pos_ == "VERB" and token.dep_ == "ROOT":
                _mark_span(text, token.idx, token.idx + len(token.text),
                           "predicate_verb", token_strs, roles)
    except (ImportError, OSError):
        pass

    return roles


# ---------------------------------------------------------------------------
# Helper: mark a character span onto token roles
# ---------------------------------------------------------------------------

def _mark_span(text: str, char_start: int, char_end: int, role: str,
               token_strs: List[str], roles: List[str]):
    """Mark tokens overlapping with [char_start, char_end) as the given role."""
    pos = 0
    for i, tok_str in enumerate(token_strs):
        clean = tok_str.replace("Ġ", " ").replace("▁", " ").lstrip()
        idx = text.find(clean, pos) if clean else pos
        if idx == -1:
            idx = pos
        tok_end = idx + len(clean)

        # Check overlap
        if idx < char_end and tok_end > char_start:
            roles[i] = role

        pos = idx + max(len(clean), 1)


# ---------------------------------------------------------------------------
# Registry: task_family → (linguistic_roles, task_roles, task_labeller)
# ---------------------------------------------------------------------------

LABELLER_REGISTRY: Dict[str, Tuple[tuple, tuple, TaskLabeller]] = {
    "SemanticRoles": (LINGUISTIC_ROLES, SEMROLE_TASK_ROLES, semrole_labeller),
    "IOI":           (LINGUISTIC_ROLES, IOI_TASK_ROLES, ioi_labeller),
    "GreaterThan":   (LINGUISTIC_ROLES, GT_TASK_ROLES, gt_labeller),
}

# Fine-grained registry (optional, for detailed analysis)
LABELLER_REGISTRY_FINE: Dict[str, Tuple[tuple, tuple, TaskLabeller]] = {
    "SemanticRoles": (LINGUISTIC_ROLES_FINE, SEMROLE_TASK_ROLES, semrole_labeller),
    "IOI":           (LINGUISTIC_ROLES_FINE, IOI_TASK_ROLES, ioi_labeller),
    "GreaterThan":   (LINGUISTIC_ROLES_FINE, GT_TASK_ROLES, gt_labeller),
}


def get_labellers(task_family: str, fine: bool = False):
    """
    Get labellers for a task family.

    Args:
        task_family: registered task family name
        fine: if True, use fine-grained linguistic roles (12 categories)

    Returns (linguistic_roles, task_roles, linguistic_labeller, task_labeller).
    Falls back to linguistic-only if task family not registered.
    """
    registry = LABELLER_REGISTRY_FINE if fine else LABELLER_REGISTRY
    ling_lab = linguistic_labeller_fine if fine else linguistic_labeller
    default_roles = LINGUISTIC_ROLES_FINE if fine else LINGUISTIC_ROLES

    if task_family in registry:
        ling_roles, task_roles, task_lab = registry[task_family]
        return ling_roles, task_roles, ling_lab, task_lab
    return default_roles, None, ling_lab, None
