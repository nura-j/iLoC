"""
split_variants.py
=================
Split each filtered (task, variant) pool into k
disjoint subsets for multi-circuit discovery, mirroring split_datasets.py.

Reads : data/variants_filtered/{model_short}/{task}_v{j}/prompts.csv
Writes: analysis/{model_short}/splits_variants/{task}_v{j}_s{i}.csv

Usage:
    python split_variants.py --model pythia1b --k 3
    python split_variants.py --model Llama3.21B --k 3 --min-rows 20
"""

import argparse
import glob
import os

import pandas as pd

from split_datasets import split_dataset, SPLIT_SIZE


def main():
    parser = argparse.ArgumentParser(description="Split variant datasets")
    parser.add_argument("--model", required=True,
                        help="model_short, e.g. pythia1b | Llama3.21B")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--in-base", default="./data/variants_filtered")
    parser.add_argument("--split-size", type=int, default=SPLIT_SIZE)
    parser.add_argument("--min-rows", type=int, default=20,
                        help="Skip (task, variant) pools smaller than this")
    parser.add_argument("--out-dir", default=None,
                        help="Override ./analysis/{model}/splits_variants "
                             "(e.g. .../splits_extra for the extra pool)")
    args = parser.parse_args()

    in_root = os.path.join(args.in_base, args.model)
    out_dir = args.out_dir or f"./analysis/{args.model}/splits_variants"
    os.makedirs(out_dir, exist_ok=True)

    pools = sorted(glob.glob(os.path.join(in_root, "*_v*", "prompts.csv")))
    if not pools:
        raise SystemExit(f"No filtered variant pools under {in_root} — "
                         f"run filter_variants.py first")

    n_files = 0
    for path in pools:
        key = os.path.basename(os.path.dirname(path))  # e.g. location_v1
        df = pd.read_csv(path)
        if len(df) < args.min_rows:
            print(f"  {key:<20} {len(df):4} rows — SKIP (<{args.min_rows})")
            continue
        splits = split_dataset(path, args.k, split_size=args.split_size)
        for i, split_df in enumerate(splits):
            out_path = os.path.join(out_dir, f"{key}_s{i}.csv")
            split_df.to_csv(out_path, index=False)
            n_files += 1
        print(f"  {key:<20} {len(df):4} rows -> {len(splits)} splits")

    print(f"\n{n_files} split files in {out_dir}/")


if __name__ == "__main__":
    main()
