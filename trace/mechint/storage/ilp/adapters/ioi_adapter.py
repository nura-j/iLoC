"""
IOI Data Adapter

Generates Indirect Object Identification (IOI) examples from
Wang et al. (2022) templates.

This adapter exists because IOI data is generated from templates,
not loaded from a CSV file. The CSVTaskAdapter handles all CSV-based
tasks (Semantic Roles, Greater-Than).

"""

import random
from typing import List, Dict, Optional, Tuple


class IOIAdapter:
    """
    Generate IOI examples from Wang et al. (2022) templates.

    IOI task: given "When {IO} and {S} went to {place}, {S} gave a
    {object} to ___", the model should predict {IO} (the indirect
    object — the name that appears exactly once, earlier in the prompt).

    Key ILP insight:
        indirect_object(C, N) :- is_name(N, _), not_repeated(N).
    """

    # Wang et al. 2022 templates (simplified)
    TEMPLATES = [
        "When {IO} and {S} went to the {place}, {S} gave a {object} to",
        "{IO} and {S} went to the {place}. {S} handed the {object} to",
        "After {IO} and {S} arrived at the {place}, {S} gave a {object} to",
        "When {IO} and {S} met at the {place}, {S} passed a {object} to",
        "{IO} and {S} were at the {place}. {S} gave the {object} to",
    ]

    NAMES = [
        "Mary", "John", "Emma", "James", "Sarah",
        "Tom", "Alice", "Bob", "Lisa", "David",
        "Kate", "Mark", "Anna", "Paul", "Jane",
    ]

    PLACES = [
        "store", "park", "school", "office", "library",
        "restaurant", "market", "hospital", "station", "museum",
    ]

    OBJECTS = [
        "gift", "book", "key", "letter", "bag",
        "toy", "note", "ring", "ticket", "card",
    ]

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

    def generate(
            self,
            n: int = 100,
            tokenizer=None
    ) -> List[Dict]:
        """
        Generate n IOI examples (balanced positive/negative).

        Each positive example: IO is the indirect object (correct).
        Each negative example: S (subject) is presented as answer (wrong).

        Args:
            n: Total number of examples (n//2 pairs -> n examples)
            tokenizer: Optional tokenizer for accurate index finding

        Returns:
            List of dicts with keys:
                prompt, target, distractor, target_idx,
                distractor_idx, correct
        """
        examples = []
        num_pairs = max(1, n // 2)

        for _ in range(num_pairs):
            # Pick distinct names
            io_name, s_name = random.sample(self.NAMES, 2)
            place = random.choice(self.PLACES)
            obj = random.choice(self.OBJECTS)
            template = random.choice(self.TEMPLATES)

            # Fill template
            prompt = template.format(
                IO=io_name, S=s_name, place=place, object=obj
            )

            # Find token indices
            if tokenizer is not None:
                io_idx, s_idx = self._find_name_indices(
                    prompt, io_name, s_name, tokenizer
                )
            else:
                io_idx, s_idx = self._find_name_indices_simple(
                    prompt, io_name, s_name
                )

            # Positive: target = IO (appears once, earlier)
            examples.append({
                'prompt': prompt,
                'target': io_name,
                'distractor': s_name,
                'target_idx': io_idx,
                'distractor_idx': s_idx,
                'correct': True,
            })

            # Negative: target = S (appears twice — wrong answer)
            examples.append({
                'prompt': prompt,
                'target': s_name,
                'distractor': io_name,
                'target_idx': s_idx,
                'distractor_idx': io_idx,
                'correct': False,
            })

        return examples

    def generate_train_test(
            self,
            n_train: int = 100,
            n_test: int = 40,
            tokenizer=None,
            seed: Optional[int] = None
    ) -> Dict[str, List[Dict]]:
        """
        Generate separate train and test sets.

        Args:
            n_train: Number of training examples
            n_test: Number of test examples
            tokenizer: Optional tokenizer
            seed: Optional seed (overrides constructor seed)

        Returns:
            {'train': [...], 'test': [...]}
        """
        if seed is not None:
            random.seed(seed)

        train = self.generate(n_train, tokenizer)
        test = self.generate(n_test, tokenizer)

        return {'train': train, 'test': test}

    # ================================================================
    # Index finding
    # ================================================================

    def _find_name_indices(
            self,
            prompt: str,
            io_name: str,
            s_name: str,
            tokenizer
    ) -> Tuple[int, int]:
        """
        Find token indices for IO and S names using a real tokenizer.

        Returns the index of the FIRST occurrence of each name.
        """
        tokens = tokenizer.tokenize(prompt)

        io_idx = self._find_token_index(tokens, io_name, fallback=0)
        s_idx = self._find_token_index(tokens, s_name, fallback=2)

        return io_idx, s_idx

    def _find_token_index(
            self,
            tokens: List[str],
            name: str,
            fallback: int = 0
    ) -> int:
        """Find first token index matching a name (case-insensitive)."""
        name_lower = name.lower()
        for i, tok in enumerate(tokens):
            # Handle BPE prefixes like 'Ġ' (GPT-2) or '▁' (sentencepiece)
            clean = tok.lower().lstrip('Ġ▁ ')
            if name_lower == clean or name_lower in clean:
                return i
        return fallback

    def _find_name_indices_simple(
            self,
            prompt: str,
            io_name: str,
            s_name: str
    ) -> Tuple[int, int]:
        """
        Fallback: find word-level indices when no tokenizer available.
        """
        words = prompt.split()

        io_idx = 0
        s_idx = 2
        for i, w in enumerate(words):
            # Strip punctuation
            clean = w.strip('.,!?;:')
            if clean == io_name and io_idx == 0:
                io_idx = i
            elif clean == s_name and s_idx == 2:
                s_idx = i

        return io_idx, s_idx
