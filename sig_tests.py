#!/usr/bin/env python3
"""
COMPASS significance tests for the paper.

T2  Mann-Whitney U (one-sided)
    Stable tasks (IOI, GT) have higher ILP confidence than sensitive tasks.

T6  Spearman correlation
    ILP confidence increases with model scale.
    x = model rank (0=Pythia-14M, 1=Pythia-1B, 2=LLaMA-3.2-1B)
    y = mean confidence per (task, model), pooled across tasks.
    Uses task-level means (not individual splits) to avoid inflation
    from the 3 non-independent prompt splits per task.

Usage
-----
    python sig_tests.py
    python sig_tests.py --out analysis/sig_tests.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats as scipy_stats

# ── Config ─────────────────────────────────────────────────────────────────────

ANALYSIS_DIR = Path("analysis")

MODELS = [
    ("pythia14m_multi",  "Pythia-14M",    0),
    ("pythia1b_multi",   "Pythia-1B",     1),
    ("Llama3.21B_multi", "LLaMA-3.2-1B",  2),
]

STABLE_TASKS = {"ioi", "greaterthan"}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_results(model_dir: str) -> dict[str, dict]:
    d = ANALYSIS_DIR / model_dir / "results"
    out = {}
    for p in sorted(d.glob("*_results.json")):
        key = p.stem.replace("_results", "")
        with open(p) as f:
            out[key] = json.load(f)
    return out


def get_records(all_results: dict, model_rank: int) -> list[dict]:
    """(task, confidence, model_rank) per circuit."""
    rows = []
    for key, r in all_results.items():
        task = key.rsplit("_", 1)[0]
        l3   = r.get("layer3") or {}
        conf = l3.get("tau_arch_confidence")
        if conf is not None:
            rows.append({"task": task, "confidence": float(conf),
                         "model_rank": model_rank})
    return rows


def sig(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


class _NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.bool_,)):    return bool(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        return super().default(o)

# ── T2: Mann-Whitney U ─────────────────────────────────────────────────────────

def run_t2(all_records: list[dict]) -> dict:
    """
    One-sided Mann-Whitney U per model.
    H1: stable tasks (IOI, GT) have stochastically higher confidence
        than all other tasks.
    """
    results = {}

    # Group by model_rank so we can test each model separately
    by_model = defaultdict(list)
    for r in all_records:
        by_model[r["model_rank"]].append(r)

    for rank, rows in sorted(by_model.items()):
        label = next(lbl for _, lbl, rk in MODELS if rk == rank)

        stable    = [r["confidence"] for r in rows if r["task"] in STABLE_TASKS]
        sensitive = [r["confidence"] for r in rows if r["task"] not in STABLE_TASKS]

        if len(stable) < 2 or len(sensitive) < 2:
            results[label] = {"error": "insufficient data"}
            continue

        U, p = scipy_stats.mannwhitneyu(stable, sensitive, alternative="greater")
        p = float(p)

        results[label] = {
            "stable_n":           len(stable),
            "stable_mean":        float(np.mean(stable)),
            "stable_std":         float(np.std(stable)),
            "sensitive_n":        len(sensitive),
            "sensitive_mean":     float(np.mean(sensitive)),
            "sensitive_std":      float(np.std(sensitive)),
            "U":                  float(U),
            "p_value":            p,
            "significant":        p < 0.05,
        }

    return results


# ── T6: Spearman correlation ───────────────────────────────────────────────────

def run_t6(all_records: list[dict]) -> dict:
    """
    Spearman r between model scale rank and ILP confidence.

    Uses per-task mean confidence per model as the unit of analysis
    (avoids inflating N with 3 non-independent prompt splits per task).
    Tasks with data in only one model are excluded.
    """
    # Compute mean confidence per (task, model_rank)
    by_task_model = defaultdict(list)
    for r in all_records:
        by_task_model[(r["task"], r["model_rank"])].append(r["confidence"])

    task_model_mean = {k: float(np.mean(v)) for k, v in by_task_model.items()}

    # Keep only tasks with data in ≥2 models (to contribute slope info)
    tasks_present = defaultdict(set)
    for task, rank in task_model_mean:
        tasks_present[task].add(rank)
    valid_tasks = {t for t, ranks in tasks_present.items() if len(ranks) >= 2}

    # Build (scale_rank, mean_conf) pairs — one per (task, model)
    pairs = [(rank, task_model_mean[(task, rank)])
             for task, rank in task_model_mean
             if task in valid_tasks]

    ranks_x = np.array([p[0] for p in pairs])
    confs_y = np.array([p[1] for p in pairs])

    r_stat, p_val = scipy_stats.spearmanr(ranks_x, confs_y)

    # Per-task: mean confidence at each scale
    per_task = {}
    model_labels = {rk: lbl for _, lbl, rk in MODELS}
    for task in sorted(valid_tasks):
        per_task[task] = {
            model_labels[rk]: task_model_mean[(task, rk)]
            for rk in sorted(model_labels)
            if (task, rk) in task_model_mean
        }

    # Per-model mean confidence (for reporting)
    per_model = {}
    for _, label, rank in MODELS:
        vals = [task_model_mean[(t, rank)]
                for t in valid_tasks if (t, rank) in task_model_mean]
        per_model[label] = {
            "n_tasks": len(vals),
            "mean_conf": float(np.mean(vals)) if vals else None,
        }

    return {
        "n_observations":    len(pairs),
        "n_tasks":           len(valid_tasks),
        "spearman_r":        float(r_stat),
        "p_value":           float(p_val),
        "significant":       float(p_val) < 0.05,
        "per_model":         per_model,
        "per_task":          per_task,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse_args()

    # Load all circuits across all models
    all_records = []
    for model_dir, label, rank in MODELS:
        try:
            res = load_results(model_dir)
        except FileNotFoundError:
            print(f"[SKIP] {label}")
            continue
        records = get_records(res, rank)
        all_records.extend(records)
        print(f"Loaded {len(records):3d} circuits — {label}")

    print()

    # ── T2 ────────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("T2  Mann-Whitney U: Stable vs Sensitive Task Confidence")
    print("    H0: confidence equal across task groups")
    print("    H1: stable (IOI, GT) > sensitive  [one-sided]")
    print("=" * 65)

    t2 = run_t2(all_records)
    for label, res in t2.items():
        if "error" in res:
            print(f"  {label:22s}  {res['error']}")
            continue
        print(
            f"  {label:22s}  "
            f"stable={res['stable_mean']:.3f}±{res['stable_std']:.3f} "
            f"(n={res['stable_n']})  "
            f"sensitive={res['sensitive_mean']:.3f}±{res['sensitive_std']:.3f} "
            f"(n={res['sensitive_n']})  "
            f"U={res['U']:.0f}  "
            f"p={res['p_value']:.4f} {sig(res['p_value'])}"
        )

    # ── T6 ────────────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("T6  Spearman r: Confidence vs Model Scale")
    print("    x = model rank (0=Py-14M, 1=Py-1B, 2=LLaMA)")
    print("    y = mean confidence per (task, model)")
    print("=" * 65)

    t6 = run_t6(all_records)

    print(
        f"\n  Spearman r = {t6['spearman_r']:.3f}  "
        f"p = {t6['p_value']:.4f} {sig(t6['p_value'])}  "
        f"(N = {t6['n_observations']} task×model pairs, "
        f"{t6['n_tasks']} tasks)\n"
    )

    # Per-model mean confidence
    model_labels = [lbl for _, lbl, _ in MODELS]
    header = "  ".join(f"{l:>12s}" for l in model_labels)
    print(f"  {'Task':18s}  {header}")
    print(f"  {'-'*18}  {'  '.join(['-'*12]*3)}")
    for task, row in sorted(t6["per_task"].items()):
        vals = "  ".join(
            f"{row.get(l, float('nan')):12.3f}" for l in model_labels
        )
        print(f"  {task:18s}  {vals}")
    print()
    per_m = t6["per_model"]
    means = "  ".join(
        f"{per_m.get(l, {}).get('mean_conf', float('nan')):12.3f}"
        for l in model_labels
    )
    print(f"  {'Mean':18s}  {means}")

    print()
    print("Legend: *** p<0.001  ** p<0.01  * p<0.05  n.s. not significant")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"T2": t2, "T6": t6}, f, indent=2, cls=_NpEnc)
        print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
