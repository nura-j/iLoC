"""
CSV Data Adapter (Generalized)

Converts CSV task data to ILP example format, driven by TaskConfig.
Handles multiple CSV formats:
  - Semantic roles: clean, target_word, corrupted, incorrect_word, correct_idx, incorrect_idx
  - Greater-Than:   prompt, reference_number, reference_idx, target_number, incorrect_number, correct
  - Any custom CSV: column mapping specified via TaskConfig.example_key_mapping

"""

import random
from typing import List, Dict, Optional, Tuple

from ..task_config import TaskConfig


class CSVTaskAdapter:
    """
    Adapt CSV task data to ILP example format.

    Uses TaskConfig.example_key_mapping to determine which CSV columns
    map to which ILP fields.  Falls back to auto-detection when a
    mapping is not provided.

    Supports two loading modes:
    1. load(csv_path, config) — config-driven, general
    2. load_from_csv(csv_path, role_label) — legacy API (backward compat)
    """

    def __init__(self, tokenizer=None):
        """
        Args:
            tokenizer: Tokenizer used by the model (optional — used
                for accurate span detection, not required for loading)
        """
        self.tokenizer = tokenizer

    # ================================================================
    # Primary API (config-driven)
    # ================================================================

    def load(
            self,
            csv_path: str,
            config: TaskConfig,
            max_examples: Optional[int] = None,
            train_ratio: float = 0.8
    ) -> Dict[str, List[Dict]]:
        """
        Load examples from CSV, driven by TaskConfig.

        Detects CSV format from the columns present and the config's
        key mapping, then produces standardised example dicts that
        the PredicateExtractor can consume.

        Args:
            csv_path: Path to CSV file
            config: TaskConfig for the current task
            max_examples: Optional cap on rows
            train_ratio: Train/test split ratio

        Returns:
            {'train': [...], 'test': [...], 'all': [...]}
        """
        import pandas as pd

        df = pd.read_csv(csv_path)
        if max_examples:
            df = df.head(max_examples)

        # Detect format and dispatch
        columns = set(df.columns)
        examples = []

        if _is_semantic_role_csv(columns):
            examples = self._load_semantic_role(df, config)
        elif _is_greater_than_csv(columns):
            examples = self._load_greater_than(df, config)
        else:
            # Generic: use config key mapping to find columns
            examples = self._load_generic(df, config)

        # Split
        train, test = self._split(examples, train_ratio)

        return {
            'all': examples,
            'train': train,
            'test': test,
            'num_train': len(train),
            'num_test': len(test),
        }

    # ================================================================
    # Format-specific loaders
    # ================================================================

    def _load_semantic_role(
            self, df, config: TaskConfig
    ) -> List[Dict]:
        """
        Load from semantic-role CSV.

        Expected columns: clean, target_word, corrupted, incorrect_word,
                          correct_idx, incorrect_idx
        """
        examples = []

        for _, row in df.iterrows():
            # Positive example
            examples.append(self._make_example(
                prompt=str(row['clean']),
                target_entity=str(row['target_word']),
                token_idx=int(row['correct_idx']),
                label=config.role_label,
                correct=True,
            ))

            # Negative example
            examples.append(self._make_example(
                prompt=str(row['corrupted']),
                target_entity=str(row['incorrect_word']),
                token_idx=int(row['incorrect_idx']),
                label=config.role_label,
                correct=False,
            ))

        return examples

    def _load_greater_than(
            self, df, config: TaskConfig
    ) -> List[Dict]:
        """
        Load from greater-than CSV.

        Expected columns: prompt, reference_number, reference_idx,
                          target_number, incorrect_number
        Produces pos/neg pairs per row.
        """
        examples = []

        for _, row in df.iterrows():
            prompt = str(row['prompt'])
            ref_num = row['reference_number']
            ref_idx = int(row['reference_idx'])

            # Positive: target > reference
            target_num = row['target_number']
            examples.append({
                'input': prompt,
                'prompt': prompt,
                'target_entity': str(target_num),
                'target_number': target_num,
                'reference_number': ref_num,
                'reference_idx': ref_idx,
                'token_idx': ref_idx,
                'tokens': self._tokenize(prompt),
                'entity_span': [ref_idx, ref_idx],
                'correct': True,
            })

            # Negative: incorrect <= reference
            inc_num = row['incorrect_number']
            examples.append({
                'input': prompt,
                'prompt': prompt,
                'target_entity': str(inc_num),
                'target_number': inc_num,
                'reference_number': ref_num,
                'reference_idx': ref_idx,
                'token_idx': ref_idx,
                'tokens': self._tokenize(prompt),
                'entity_span': [ref_idx, ref_idx],
                'correct': False,
            })

        return examples

    def _load_generic(
            self, df, config: TaskConfig
    ) -> List[Dict]:
        """
        Fallback: load from any CSV using config.example_key_mapping.

        Reads each row into a dict using the mapping.  If 'correct'
        column is not present, all examples are treated as positive.
        """
        km = config.example_key_mapping
        examples = []

        for _, row in df.iterrows():
            ex = {}
            # Map config keys to row values
            for generic_key, csv_col in km.items():
                if csv_col in row.index:
                    ex[generic_key] = row[csv_col]

            # Ensure minimum required fields
            if 'correct' not in ex:
                ex['correct'] = True
            if 'input' not in ex and 'prompt' not in ex:
                # Use first string column as prompt
                for col in row.index:
                    if isinstance(row[col], str) and len(str(row[col])) > 10:
                        ex['input'] = str(row[col])
                        ex['prompt'] = str(row[col])
                        break

            # Tokenize if prompt available
            prompt = ex.get('input', ex.get('prompt', ''))
            if prompt and 'tokens' not in ex:
                ex['tokens'] = self._tokenize(prompt)

            examples.append(ex)

        return examples

    # ================================================================
    # Legacy API (backward compatible)
    # ================================================================

    def load_from_csv(
            self,
            csv_path: str,
            role_label: str,
            max_examples: Optional[int] = None
    ) -> List[Dict]:
        """
        Legacy API: load from semantic-role CSV.

        Maintained for backward compatibility with existing code.
        Internally creates a minimal TaskConfig.
        """
        import pandas as pd

        df = pd.read_csv(csv_path)
        if max_examples:
            df = df.head(max_examples)

        examples = []
        for _, row in df.iterrows():
            examples.append(self._make_example(
                prompt=str(row['clean']),
                target_entity=str(row['target_word']),
                token_idx=int(row['correct_idx']),
                label=role_label,
                correct=True,
            ))
            examples.append(self._make_example(
                prompt=str(row['corrupted']),
                target_entity=str(row['incorrect_word']),
                token_idx=int(row['incorrect_idx']),
                label=role_label,
                correct=False,
            ))
        return examples

    # ================================================================
    # Helpers
    # ================================================================

    def _make_example(
            self, prompt, target_entity, token_idx,
            label=None, correct=True
    ) -> Dict:
        """Build a standardised example dict."""
        tokens = self._tokenize(prompt)
        span = self._find_entity_span(tokens, target_entity, token_idx)
        ex = {
            'input': prompt,
            'prompt': prompt,
            'target_entity': target_entity,
            'tokens': tokens,
            'entity_span': span,
            'correct': correct,
            'token_idx': token_idx,
        }
        if label is not None:
            ex['label'] = label
        return ex

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text, falling back to whitespace split."""
        if self.tokenizer is not None:
            try:
                return self.tokenizer.tokenize(text)
            except Exception:
                pass
        return text.split()

    def _find_entity_span(
            self, tokens: List[str], target_word: str, hint_idx: int
    ) -> List[int]:
        """Find token span for target entity."""
        target_lower = str(target_word).lower()

        # Try hint index first
        if 0 <= hint_idx < len(tokens):
            if target_lower in tokens[hint_idx].lower():
                return [hint_idx, hint_idx]
            for offset in [-1, 1, -2, 2]:
                ci = hint_idx + offset
                if 0 <= ci < len(tokens) and target_lower in tokens[ci].lower():
                    return [ci, ci]

        # Linear search
        for i, tok in enumerate(tokens):
            if target_lower in tok.lower():
                return [i, i]

        # Multi-token span (if tokenizer available)
        if self.tokenizer is not None:
            try:
                target_toks = self.tokenizer.tokenize(target_word)
                if len(target_toks) > 1:
                    for i in range(len(tokens) - len(target_toks) + 1):
                        if tokens[i:i + len(target_toks)] == target_toks:
                            return [i, i + len(target_toks) - 1]
            except Exception:
                pass

        return [hint_idx, hint_idx]

    def _split(
            self, examples: List[Dict], train_ratio: float
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split examples into train/test."""
        shuffled = examples.copy()
        random.shuffle(shuffled)
        idx = int(len(shuffled) * train_ratio)
        return shuffled[:idx], shuffled[idx:]

    def create_activation_mapping(
            self, circuit_graph, circuit_uri,
            examples: List[Dict],
            activation_threshold: float = 0.5
    ) -> Dict[int, List]:
        """Placeholder for activation mapping (future work)."""
        return {}

    # Backward-compatible aliases
    split_train_test = _split


# ================================================================
# Format detection
# ================================================================

def _is_semantic_role_csv(columns: set) -> bool:
    """Check if columns match semantic-role CSV format."""
    required = {'clean', 'target_word', 'corrupted', 'incorrect_word',
                'correct_idx', 'incorrect_idx'}
    return required.issubset(columns)


def _is_greater_than_csv(columns: set) -> bool:
    """Check if columns match greater-than CSV format."""
    required = {'prompt', 'reference_number', 'reference_idx',
                'target_number', 'incorrect_number'}
    return required.issubset(columns)


# ================================================================
# Convenience Functions
# ================================================================

def load_task_examples(
        csv_path: str,
        config: TaskConfig,
        tokenizer=None,
        max_examples: Optional[int] = None
) -> List[Dict]:
    """
    Convenience: load all examples from CSV (no split).

    Args:
        csv_path: Path to CSV file
        config: TaskConfig for the task
        tokenizer: Optional tokenizer
        max_examples: Optional limit

    Returns:
        List of example dicts
    """
    adapter = CSVTaskAdapter(tokenizer)
    result = adapter.load(csv_path, config, max_examples, train_ratio=1.0)
    return result['all']


def prepare_ilp_data(
        csv_path: str,
        config: TaskConfig,
        tokenizer=None,
        train_ratio: float = 0.8,
        max_examples: Optional[int] = None
) -> Dict:
    """
    Convenience: complete data preparation pipeline.

    Args:
        csv_path: Path to CSV file
        config: TaskConfig for the task
        tokenizer: Optional tokenizer
        train_ratio: Train/test split ratio
        max_examples: Optional limit

    Returns:
        Dict with 'train', 'test', 'all' examples
    """
    adapter = CSVTaskAdapter(tokenizer)
    return adapter.load(csv_path, config, max_examples, train_ratio)