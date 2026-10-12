"""
threshold_sensitivity.py
========================
Sensitivity sweep over every validation threshold, computed from stored
evaluation records only (no model runs):

  theta_behav x theta_causal : acceptance over the 338 stored live-transfer
                               evaluations (analysis/transfer)
  epsilon_DLA                : causal fraction per family from the stored
                               per-component DLA scores (analysis/cfs_scaled)
  tau                        : k* at 0.5 vs 0.3 (analysis/tables/kstar_summary)

Usage: python3 threshold_sensitivity.py
Output: analysis/tables/threshold_sensitivity.json (+ printed tables)
"""

import glob
import json

import numpy as np


def fam(m):
    for k in ("pythia", "gpt2", "Qwen", "Llama"):
        if m.startswith(k):
            return k


out = {}

c = json.load(open("analysis/transfer/live_transfer_scaled.json"))
rows = [(fam(p.split("_to_")[0]) == fam(p.split("_to_")[1]),
         e["acc_t"], e["delta_t"])
        for p, r in c.items() for e in r.values()]
TB = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
TC = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
grid, gaps = {}, []
for tb in TB:
    for tc in TC:
        acc = [(w, a >= tb and d >= tc) for w, a, d in rows]
        rate = float(np.mean([x for _, x in acc]))
        gap = abs(float(np.mean([x for w, x in acc if w]))
                  - float(np.mean([x for w, x in acc if not w])))
        grid[f"tb{tb}_tc{tc}"] = {"rate": round(rate, 4), "gap": round(gap, 4)}
        gaps.append(gap)
out["behav_causal_grid"] = {"n_evaluations": len(rows), "grid": grid,
                            "max_family_gap": round(max(gaps), 4)}
print(f"theta grid: rate {100*min(g['rate'] for g in grid.values()):.1f}"
      f"-{100*max(g['rate'] for g in grid.values()):.1f}%, "
      f"max family gap {100*max(gaps):.1f} pp")

scores = {}
for p in glob.glob("analysis/cfs_scaled/*/*.json"):
    f = fam(p.split("/")[-2])
    comps = json.load(open(p))["components"]
    scores.setdefault(f, []).extend(abs(v["dla_score"])
                                    for v in comps.values())
EPS = [0.01, 0.02, 0.05, 0.1, 0.2]
out["eps_dla"] = {f: {str(e): round(float(np.mean([s >= e for s in ss])), 3)
                      for e in EPS} for f, ss in scores.items()}
for f, d in out["eps_dla"].items():
    print(f"  eps_DLA {f:<7}: " + " ".join(f"{v:.2f}" for v in d.values()))

ks = json.load(open("analysis/tables/kstar_summary.json"))
out["tau"] = {m: {"kstar_tau05": ks[m]["kstar_tau05"],
                  "kstar_tau03": ks[m]["kstar_tau03"]} for m in ks}
same = sum(1 for m in ks if ks[m]["kstar_tau05"] == ks[m]["kstar_tau03"])
print(f"tau 0.5->0.3: unchanged {same}/12, one grid step smaller {12-same}/12")

json.dump(out, open("analysis/tables/threshold_sensitivity.json", "w"),
          indent=1)
print("-> analysis/tables/threshold_sensitivity.json")
