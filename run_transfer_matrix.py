"""
run_transfer_matrix.py
======================
Cross-model signature-retrieval test (the transfer engine's static phase,
paper Sec. 3.7 Steps 1-2) across every ordered model pair.

For each model, one tau_arch clause per task is learned from ALL of that
task's stored circuits (positives) against all other tasks' circuits
(negatives), using the same selection procedure as the pooled learner, and
summarised as an ILP signature. For every ordered model pair (source,
target) and every shared task, the source signature queries the target's
signature registry; retrieval is correct when the same task is among the
nearest signatures (ties in this distance are frequent and genuine).

Reports per-pair top-1 retrieval accuracy, within- vs cross-family
aggregates, and the same-task vs different-task distance ratio.

Usage:
    python run_transfer_matrix.py --models gpt2 gpt2large pythia14m ... \
        [--by variant] [--out analysis/transfer/transfer_matrix.json]
"""

import argparse
import glob
import itertools
import json
import os
import re
from collections import defaultdict

import numpy as np

from trace.mechint.heldout import _extract, default_task_family
from trace.mechint.storage.ilp.ilp_arch import (
    ArchitecturalILPLearner, _arch_signature,
)
from trace.mechint.storage.ilp.ilp_signature_distance import ilp_signature_distance

FAMILY = {"gpt2": "GPT-2", "gpt2large": "GPT-2",
          "pythia14m": "Pythia", "pythia410m": "Pythia",
          "pythia1b": "Pythia", "pythia6.9b": "Pythia",
          "Qwen2.50.5B": "Qwen", "Qwen2.51.5B": "Qwen",
          "Qwen2.53B": "Qwen", "Qwen2.57B": "Qwen",
          "Llama3.21B": "Llama", "Llama3.23B": "Llama"}


def load_task_circuits(model, by):
    """{task: [(feats, raw), ...]} pooled over variants and splits."""
    pat = (f"analysis/{model}_multi/results/*_v*_s*_results.json" if by == "variant"
           else f"analysis/{model}_multi/results/*_s*_results.json")
    out = defaultdict(list)
    for path in sorted(glob.glob(pat)):
        stem = os.path.basename(path).replace("_results.json", "")
        m = re.match(r"(.+?)(?:_v\d+)?_s\d+$", stem)
        if not m:
            continue
        if by == "split" and "_v" in stem:
            continue
        task = m.group(1)
        with open(path) as f:
            rec = json.load(f)
        out[task].append(_extract(rec))
    return dict(out)


def learn_signatures(model, task_circuits):
    """{task: S_ILP} via the standard one-vs-rest selection, quietly."""
    learner = ArchitecturalILPLearner()
    sigs = {}
    for task, circuits in task_circuits.items():
        pos_feats = [f for f, _ in circuits]
        pos_raw = [r for _, r in circuits]
        neg_feats = [f for o, cs in task_circuits.items() if o != task
                     for f, _ in cs]
        neg_raw = [r for o, cs in task_circuits.items() if o != task
                   for _, r in cs]
        if not pos_feats or not neg_feats:
            continue
        clause, _, _ = learner._select_best_clause(
            task, pos_feats, neg_feats, pos_raw, neg_raw)
        motifs, seen = [], set()
        for f in pos_feats:
            for mo in f.get("motifs", []):
                if mo not in seen:
                    motifs.append(mo)
                    seen.add(mo)
        sigs[task] = _arch_signature(clause, motifs=motifs)
    return sigs


def main():
    ap = argparse.ArgumentParser(description="Static transfer retrieval matrix")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--by", choices=["split", "variant"], default="variant")
    ap.add_argument("--delta", type=float, default=0.30,
                    help="delta_ILP acceptance threshold (Def. 5)")
    ap.add_argument("--out", default="analysis/transfer/transfer_matrix.json")
    args = ap.parse_args()

    registries = {}
    for m in args.models:
        tc = load_task_circuits(m, args.by)
        registries[m] = learn_signatures(m, tc)
        print(f"{m}: {len(registries[m])} task signatures "
              f"({sum(len(v) for v in tc.values())} circuits)")

    pair_rows = {}
    same_d, diff_d = [], []
    for src, tgt in itertools.permutations(args.models, 2):
        shared = sorted(set(registries[src]) & set(registries[tgt]))
        if not shared:
            continue
        hits, fam_hits, within = 0, 0, 0
        for t in shared:
            dists = {u: ilp_signature_distance(registries[src][t],
                                               registries[tgt][u])
                     for u in registries[tgt]}
            dmin = min(dists.values())
            nearest = [u for u, v in dists.items() if abs(v - dmin) < 1e-9]
            if t in nearest:
                hits += 1
            # operation-family retrieval: the paper's claim is family-level
            # (binding/selection/comparison), since roles collapse structurally
            if any(default_task_family(u) == default_task_family(t)
                   for u in nearest):
                fam_hits += 1
            if dists[t] < args.delta:
                within += 1
            same_d.append(dists[t])
            diff_d.extend(v for u, v in dists.items() if u != t)
        pair_rows[f"{src}__{tgt}"] = {
            "n_tasks": len(shared),
            "top1_acc": round(hits / len(shared), 3),
            "family_top1": round(fam_hits / len(shared), 3),
            "within_delta": round(within / len(shared), 3),
        }

    accs = [r["top1_acc"] for r in pair_rows.values()]
    fams = [r["family_top1"] for r in pair_rows.values()]
    wds = [r["within_delta"] for r in pair_rows.values()]
    fam_accs = defaultdict(list)
    for key, r in pair_rows.items():
        a, b = key.split("__")
        kind = ("within-family" if FAMILY.get(a) == FAMILY.get(b)
                else "cross-family")
        fam_accs[kind].append(r["top1_acc"])

    ratio = float(np.mean(diff_d)) / float(np.mean(same_d)) if same_d else None

    print(f"\n== Static transfer retrieval ({len(pair_rows)} ordered pairs) ==")
    print(f"  top-1 task retrieval     : mean {np.mean(accs):.3f} "
          f"(min {np.min(accs):.2f}, max {np.max(accs):.2f})")
    print(f"  top-1 FAMILY retrieval   : mean {np.mean(fams):.3f} "
          f"(min {np.min(fams):.2f}, max {np.max(fams):.2f})")
    for kind, vals in sorted(fam_accs.items()):
        print(f"    {kind:<14}: {np.mean(vals):.3f}  ({len(vals)} pairs)")
    print(f"  same-task within delta={args.delta}: mean {np.mean(wds):.3f}")
    print(f"  distance ratio (different-task / same-task): {ratio:.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"params": vars(args), "pairs": pair_rows,
                   "summary": {
                       "mean_top1": round(float(np.mean(accs)), 3),
                       "mean_family_top1": round(float(np.mean(fams)), 3),
                       "within_family_top1": round(float(np.mean(
                           fam_accs["within-family"])), 3) if fam_accs["within-family"] else None,
                       "cross_family_top1": round(float(np.mean(
                           fam_accs["cross-family"])), 3) if fam_accs["cross-family"] else None,
                       "mean_within_delta": round(float(np.mean(wds)), 3),
                       "diff_over_same_distance": round(ratio, 3),
                   }}, f, indent=2)
    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
