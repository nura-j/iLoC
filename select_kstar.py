"""
select_kstar.py
===============
Choose a faithfulness-matched edge budget (k*) per model from the k-sweep
curves recorded during discovery, and derive the k*-trimmed circuit pool.

Selection rule (default): k* = the smallest swept budget whose MEDIAN
faithfulness across the model's circuits is at least --tau (default 0.5).
If no budget reaches tau, k* = the largest swept budget and the attained
median faithfulness is reported. --tau 0.3 serves as the sensitivity check.

Derived pool: circuits are trimmed to their top-k* edges by |score| (top-k
circuits are nested, so this is exact, not an approximation) and written to

    analysis/{model}-k{k*}_multi/results/{same filenames}

so every downstream tool (run_heldout.py, run_transfer_matrix.py,
acl_augmentation.py, generate_clause_tables.py) runs unchanged by using the
pseudo-model key "{model}-k{k*}".

Metric provenance in derived records: faithfulness and circuit_performance
come from the sweep evaluation at k* (real measured values). DeltaT is
inherited from the exported budget (ablation was not re-evaluated per k)
and flagged with "delta_t_budget".

Usage:
    python select_kstar.py --model pythia6.9b            # report + derive
    python select_kstar.py --model pythia6.9b --tau 0.3  # sensitivity rule
    python select_kstar.py --model pythia6.9b --dry-run  # report only
"""

import argparse
import glob
import json
import os

import numpy as np


def load_sweeps(model):
    """{path: record} for circuits that carry a k_sweep."""
    out = {}
    for path in sorted(glob.glob(f"analysis/{model}_multi/results/*_results.json")):
        with open(path) as f:
            rec = json.load(f)
        if (rec.get("layer1") or {}).get("metrics", {}).get("k_sweep"):
            out[path] = rec
    return out


def median_curve(records):
    """{k(int): median faithfulness} over all circuits."""
    per_k = {}
    for rec in records.values():
        for k, v in rec["layer1"]["metrics"]["k_sweep"].items():
            per_k.setdefault(int(k), []).append(v["faithfulness"])
    return {k: float(np.median(vs)) for k, vs in sorted(per_k.items())}


def choose_kstar(curve, tau):
    for k, med in curve.items():
        if med >= tau:
            return k, med, True
    k = max(curve)
    return k, curve[k], False


def trim_circuit(rec, kstar):
    """Exact top-k* sub-circuit of the exported circuit (nested budgets)."""
    c = rec["layer1"]["circuit"]
    edges = sorted(c["edges"], key=lambda e: -abs(e.get("score") or 0.0))[:kstar]
    kept_nodes = {e["source"] for e in edges} | {e["target"] for e in edges}
    nodes = [n for n in c["nodes"] if n["name"] in kept_nodes]

    new = json.loads(json.dumps(rec))  # deep copy
    nc = new["layer1"]["circuit"]
    nc["edges"] = edges
    nc["nodes"] = nodes
    nc["num_edges"] = len(edges)
    nc["num_nodes"] = len(nodes)

    m = new["layer1"]["metrics"]
    sweep_at_k = rec["layer1"]["metrics"]["k_sweep"][str(kstar)]
    m["faithfulness"] = sweep_at_k["faithfulness"]
    m["circuit_performance"] = sweep_at_k["circuit_performance"]
    m["AccT"] = sweep_at_k["circuit_performance"]
    m["num_edges_in_circuit"] = len(edges)
    m["num_nodes_in_circuit"] = len(nodes)
    # ablation was evaluated at the exported budget only
    m["delta_t_budget"] = rec["layer1"]["metrics"].get("num_edges_in_circuit")
    # layer3 (if any) was learned at the exported budget — drop it so the
    # ILP stage re-learns on the trimmed pool
    new["layer3"] = None
    return new


def main():
    ap = argparse.ArgumentParser(description="Faithfulness-matched k* selection")
    ap.add_argument("--model", required=True,
                    help="analysis dir key, e.g. pythia6.9b")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="median-faithfulness target (default 0.5)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the curve and k* without writing the pool")
    args = ap.parse_args()

    records = load_sweeps(args.model)
    if not records:
        raise SystemExit(f"No circuits with k_sweep under "
                         f"analysis/{args.model}_multi/results — was the run "
                         f"made with --k-sweep?")

    curve = median_curve(records)
    kstar, med, reached = choose_kstar(curve, args.tau)

    print(f"{args.model}: {len(records)} circuits with sweep")
    print("  median faithfulness by budget:")
    for k, v in curve.items():
        mark = "  <- k*" if k == kstar else ""
        print(f"    k={k:<5} {v:+.3f}{mark}")
    status = ("meets" if reached else
              f"does NOT reach tau={args.tau}; using largest budget")
    print(f"  k* = {kstar} (median faithfulness {med:+.3f}, {status})")

    if args.dry_run:
        return

    out_key = f"{args.model}-k{kstar}"
    out_dir = f"analysis/{out_key}_multi/results"
    os.makedirs(out_dir, exist_ok=True)
    for path, rec in records.items():
        new = trim_circuit(rec, kstar)
        with open(os.path.join(out_dir, os.path.basename(path)), "w") as f:
            json.dump(new, f, indent=2)
    print(f"  derived pool: {out_dir} ({len(records)} circuits at k={kstar})")
    print(f"  downstream:   python run_heldout.py --model {out_key} --by variant")
    print(f"                python run_transfer_matrix.py --models ... {out_key}")


if __name__ == "__main__":
    main()
