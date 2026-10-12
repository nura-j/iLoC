"""
split_datasets.py
==================
Create k disjoint prompt subsets per task for multi-circuit discovery.

Each split produces a separate CSV of ~50 prompts. Running EAP-IG on
different prompt subsets yields structurally different circuits for the
same task — enabling ILP to learn from multiple positive examples.

Usage:
    python split_datasets.py                        # default k=3
    python split_datasets.py --k 5 --model pythia1b
    python split_datasets.py --list                 # just show what's available
"""

import argparse
import os
import numpy as np
import pandas as pd

SEED = 42
SPLIT_SIZE = 50  # prompts per split (matches run_experiment.py max_examples)

# ── Dataset registry (all tasks × all models) ─────────────────────────────

_SEMROLE_BASE = "./semantic_role_generation"

_ROLES_ALL = [
    "Beneficiary", "Goal", "Instrument", "Location",
    "Path", "Source", "Time", "Topic",
]

_MODEL_PATHS = {
    "pythia1b": f"{_SEMROLE_BASE}/semroles_v1_filtered_1b",
    "pythia14m": f"{_SEMROLE_BASE}/semroles_v1_filtered_14m",
    "Llama3.21B": f"{_SEMROLE_BASE}/semroles_v1_Llama_3_2_1B_filtered",
}

_NON_ROLE_DATASETS = {
    "pythia1b": {
        "IOI": "./data/ioi_pythia_1b_filtered.csv",
        "GreaterThan": "./data/greater_than_filtered_pythia1b/both_correct/prompts.csv",
    },
    "pythia14m": {
        "IOI": "./data/ioi_pythia_14m_filtered.csv",
        "GreaterThan": "./data/greater_than_filtered_pythia14m/both_correct/prompts.csv",
    },
    "Llama3.21B": {
        "IOI": "./data/ioi_llama_1b.csv",
        "GreaterThan": "./data/greater_than_filtered_llama1b/both_correct/prompts.csv",
    },
}


def get_all_datasets(model: str):
    """Return {task_name: csv_path} for all available tasks for a model."""
    datasets = {}

    # Semantic roles
    role_base = _MODEL_PATHS.get(model)
    if role_base:
        for role in _ROLES_ALL:
            path = f"{role_base}/{role}/both_correct/prompts.csv"
            if os.path.isfile(path):
                datasets[role] = path

    # IOI + GreaterThan
    for task, path in _NON_ROLE_DATASETS.get(model, {}).items():
        if os.path.isfile(path):
            datasets[task] = path

    return datasets


def split_dataset(csv_path: str, k: int, split_size: int = SPLIT_SIZE,
                  seed: int = SEED):
    """
    Split a CSV into k disjoint subsets of split_size rows each.

    Returns list of DataFrames (may be fewer than k if dataset is small).
    """
    df = pd.read_csv(csv_path)
    n = len(df)

    # Shuffle deterministically
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    splits = []
    for i in range(k):
        start = i * split_size
        end = start + split_size
        if start >= n:
            break
        subset_idx = indices[start:min(end, n)]
        if len(subset_idx) < 20:  # too small to be useful
            break
        splits.append(df.iloc[subset_idx].reset_index(drop=True))

    return splits


def create_splits(model: str, k: int = 3, output_dir: str = None):
    """
    Create split CSVs for all tasks of a model.

    Writes to: {output_dir}/{task}_s{i}.csv
    Returns: {task_name: [list of split csv paths]}
    """
    if output_dir is None:
        output_dir = f"./analysis/{model}/splits"
    os.makedirs(output_dir, exist_ok=True)

    datasets = get_all_datasets(model)
    result = {}

    for task, csv_path in sorted(datasets.items()):
        df = pd.read_csv(csv_path)
        n_rows = len(df)
        max_k = min(k, n_rows // 20)  # at least 20 rows per split

        if max_k < 1:
            print(f"  {task:14} {n_rows:4} rows — SKIP (too small)")
            continue

        splits = split_dataset(csv_path, max_k)
        paths = []
        for i, split_df in enumerate(splits):
            out_path = os.path.join(output_dir, f"{task.lower()}_s{i}.csv")
            split_df.to_csv(out_path, index=False)
            paths.append(out_path)

        result[task] = paths
        print(f"  {task:14} {n_rows:4} rows -> {len(paths)} splits "
              f"of {SPLIT_SIZE} @ {output_dir}/")

    return result


def list_available(model: str):
    """Print available datasets and their sizes."""
    datasets = get_all_datasets(model)
    print(f"\nAvailable datasets for {model}:")
    total = 0
    for task, path in sorted(datasets.items()):
        n = sum(1 for _ in open(path)) - 1  # minus header
        max_splits = n // SPLIT_SIZE
        total += n
        print(f"  {task:14} {n:5} rows  (max {max_splits} splits of {SPLIT_SIZE})")
    print(f"  {'TOTAL':14} {total:5} rows")


def main():
    parser = argparse.ArgumentParser(description="Split datasets for multi-circuit discovery")
    parser.add_argument("--model", default="pythia1b")
    parser.add_argument("--k", type=int, default=3, help="Number of splits per task")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--all-models", action="store_true")
    args = parser.parse_args()

    if args.list:
        models = _MODEL_PATHS.keys() if args.all_models else [args.model]
        for m in models:
            list_available(m)
        return

    models = _MODEL_PATHS.keys() if args.all_models else [args.model]
    for m in models:
        print(f"\n{'='*60}")
        print(f"  Splitting datasets for {m} (k={args.k})")
        print(f"{'='*60}")
        result = create_splits(m, k=args.k)
        total_splits = sum(len(v) for v in result.values())
        print(f"\n  Total: {len(result)} tasks, {total_splits} split files")


if __name__ == "__main__":
    main()
