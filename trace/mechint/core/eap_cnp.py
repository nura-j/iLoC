import torch
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader


def collate_cnp(xs):
    """Collate function for CNP dataset"""
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    # labels = torch.tensor(labels)
    labels = torch.tensor(labels)
    return clean, corrupted, labels


class CNPDataset(Dataset):
    """CNP dataset following TRACE conventions"""

    def __init__(self, intervention_data: List[Dict], tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.data = self._preprocess_data(intervention_data)
        if isinstance(self.data, dict) and hasattr(self.data, 'results'):
            self.data = self.data['results']

    def _preprocess_data(self, data: List[Dict]) -> List[Dict]:
        """Filter and validate intervention data"""
        processed = []
        violations = 0

        for item in data:
            # Filter by intervention type
            print(f'item {item}')
            if item['intervention_type'] not in self.config.intervention_types:
                continue

            # Validate single-token constraint
            target_tokens = self.tokenizer.encode(item['target_word'])
            if len(target_tokens) != 1:
                violations += 1
                continue

            processed.append(item)

        if violations > 0:
            print(f"Filtered {violations} multi-token items from CNP dataset")

        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item['clean'], item['corrupted'], [item['correct_idx'], item['incorrect_idx']]

    def to_dataloader(self):
        return DataLoader(self, batch_size=self.config.batch_size, collate_fn=collate_cnp)


def cnp_metric(logits: torch.Tensor, clean_logits: torch.Tensor,
               input_length: torch.Tensor, labels: torch.Tensor,
               mean=True, loss=False):
    """CNP metric for EAP analysis"""
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)
    # input_length can arrive on a different device than the logits
    # (get_attention_mask returns CPU tensors in some TransformerLens
    # versions); mixed-device fancy indexing raises on CUDA.
    input_length = input_length.to(logits.device)
    target_logits = logits[idx, input_length - 1]

    log_probs = torch.log_softmax(target_logits, dim=-1)
    # print(f'Type of logits: {(logits.dtype)}')
    # print(f'Type of labels1: {(labels.dtype)}')
    labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
    target_probs = torch.gather(log_probs, -1, labels)
    cnp_scores = target_probs[:, 0] - target_probs[:, 1]

    if loss:
        cnp_scores = -cnp_scores
    if mean:
        cnp_scores = cnp_scores.mean()
    # print(f'Type of cnp_scores: {type(cnp_scores)}')

    return cnp_scores