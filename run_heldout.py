"""
run_heldout.py
==============
Three-fold held-out evaluation over an existing multi-circuit pool
No model load — works entirely on stored results JSON.

Usage:
    python run_heldout.py --model pythia1b
    python run_heldout.py --model pythia14m --folds 3

Reads : analysis/{model}_multi/results/{task}_s{split}_results.json
Writes: analysis/{model}_multi/heldout/heldout_report.json
"""

import argparse
import glob
import json
import os
import re

from trace.mechint.heldout import three_fold_heldout


def load_records(results_dir: str):
    """{task: {split_idx: record}} from {task}_s{i}_results.json (no variants)."""
    records = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_s*_results.json"))):
        m = re.match(r"(.+)_s(\d+)_results\.json", os.path.basename(path))
        if not m or "_v" in m.group(1):
            continue
        task, split = m.group(1), int(m.group(2))
        with open(path) as f:
            records.setdefault(task, {})[split] = json.load(f)
    return records


def load_records_by_variant(results_dir: str):
    """
    {task: {variant_pos: record}} from {task}_v{v}_s{i}_results.json.

    One circuit per (task, variant): the lowest split index present. Variant
    keys are remapped to 0-based positions so fold i = i-th variant held out
    (leave-one-variant-out).
    """
    raw = {}
    for path in sorted(glob.glob(os.path.join(results_dir,
                                              "*_v*_s*_results.json"))):
        m = re.match(r"(.+)_v(\d+)_s(\d+)_results\.json",
                     os.path.basename(path))
        if not m:
            continue
        task, v, s = m.group(1), int(m.group(2)), int(m.group(3))
        cur = raw.setdefault(task, {})
        if v not in cur or s < cur[v][0]:
            with open(path) as f:
                cur[v] = (s, json.load(f))
    records = {}
    for task, variants in raw.items():
        records[task] = {pos: rec
                         for pos, (v, (s, rec))
                         in enumerate(sorted((v, sr) for v, sr
                                             in variants.items()))}
    return records


def main():
    parser = argparse.ArgumentParser(description="Three-fold held-out evaluation")
    parser.add_argument("--model", required=True,
                        help="analysis dir key, e.g. pythia14m | pythia1b | Llama3.21B")
    parser.add_argument("--folds", type=int, default=None,
                        help="Default: 3 for --by split, #variants for --by variant")
    parser.add_argument("--by", choices=["split", "variant"], default="split",
                        help="Fold unit: prompt split (default) or prompt "
                             "variant (leave-one-variant-out, §1.1c)")
    args = parser.parse_args()

    results_dir = f"./analysis/{args.model}_multi/results"
    if args.by == "variant":
        records = load_records_by_variant(results_dir)
    else:
        records = load_records(results_dir)
    if not records:
        raise SystemExit(f"No matching results JSON found in {results_dir} "
                         f"(--by {args.by})")

    n_folds = args.folds or (
        max(len(v) for v in records.values()) if args.by == "variant" else 3)

    print(f"Held-out evaluation: {args.model}  by={args.by}  "
          f"({len(records)} tasks, {sum(len(v) for v in records.values())} circuits, "
          f"{n_folds} folds)")

    report = three_fold_heldout(records, n_folds=n_folds)
    report["fold_unit"] = args.by
    args.folds = n_folds

    out_dir = f"./analysis/{args.model}_multi/heldout"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        "heldout_report.json" if args.by == "split"
        else "heldout_by_variant_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    agg = report["aggregate"]
    print(f"\n{'='*64}")
    print(f"HELD-OUT RESULTS ({args.folds} folds)  [{args.model}]")
    print(f"{'='*64}")
    print(f"\n  ILP held-out one-vs-rest (pooled across folds):")
    print(f"    {'task':<14} {'coverage':>8} {'pooled_F1':>10}   per-fold F1")
    for t, row in sorted(agg["ilp_heldout"].items()):
        pf = ", ".join(f"{v:.2f}" for v in row["per_fold_f1"])
        print(f"    {t:<14} {row['coverage']:>8.2f} {row['pooled_f1']:>10.3f}   [{pf}]")
    print(f"    mean pooled F1 = {agg['ilp_mean_pooled_f1']}")

    print(f"\n  Held-out separation (mean ± std over folds):")
    for m, s in agg["separation"].items():
        if s["mean"] is None:
            print(f"    {m:<16} n/a")
        else:
            print(f"    {m:<16} {s['mean']:.3f} ± {s['std']:.3f}   folds={s['folds']}")

    print(f"\n  Held-out classification accuracy (mean ± std over folds):")
    for m, s in agg["classification"].items():
        print(f"    {m:<16} {s['mean']:.3f} ± {s['std']:.3f}")

    if agg["skipped"]:
        print(f"\n  Skipped task-folds: {len(agg['skipped'])}")
        for s in agg["skipped"]:
            print(f"    {s}")

    print(f"\n  Report: {out_path}")


if __name__ == "__main__":
    main()
