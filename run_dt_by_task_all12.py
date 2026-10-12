"""
run_dt_by_task_all12.py
=======================
Per-task causal contribution across ALL twelve models' k* pools,
extending the three-model preservation table of the paper. For every
(model, task) the median DeltaT over the pool's circuits (three variants
x three splits) is read from the archived validation records
(layer1/metrics/DeltaT in analysis/<key>_multi/results/*_v*_s*_results.json).
Validation: the per-model median over all circuits must reproduce the
DeltaT column of the held-out cross-family table.

Usage: python3 run_dt_by_task_all12.py
Output:
  paper/From_Functions_to_Logic/figures/fig_dt_by_task_12.pdf
  analysis/tables/dt_by_task_all12.json
"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps
from scipy.stats import friedmanchisquare

KS = json.load(open("analysis/tables/kstar_summary.json"))
MODELS = ["gpt2", "gpt2large", "pythia14m", "pythia410m", "pythia1b",
          "pythia6.9b", "Qwen2.50.5B", "Qwen2.51.5B", "Qwen2.53B",
          "Qwen2.57B", "Llama3.21B", "Llama3.23B"]
NAME = {"gpt2": "GPT-2", "gpt2large": "GPT-2-large", "pythia14m": "Pythia-14M",
        "pythia410m": "Pythia-410M", "pythia1b": "Pythia-1B",
        "pythia6.9b": "Pythia-6.9B", "Qwen2.50.5B": "Qwen2.5-0.5B",
        "Qwen2.51.5B": "Qwen2.5-1.5B", "Qwen2.53B": "Qwen2.5-3B",
        "Qwen2.57B": "Qwen2.5-7B", "Llama3.21B": "Llama-3.2-1B",
        "Llama3.23B": "Llama-3.2-3B"}
ORDER = ["ioi", "greaterthan", "beneficiary", "goal", "instrument", "location",
         "path", "source", "time", "topic"]
LBL = {"ioi": "IOI", "greaterthan": "GT", "beneficiary": "Ben", "goal": "Goal",
       "instrument": "Ins", "location": "Loc", "path": "Path", "source": "Src",
       "time": "Time", "topic": "Top"}
EXPECTED = {"gpt2": 3.78, "gpt2large": 3.66, "pythia14m": 3.22,
            "pythia410m": 3.79, "pythia1b": 3.65, "pythia6.9b": 3.72,
            "Qwen2.50.5B": 4.50, "Qwen2.51.5B": 4.85, "Qwen2.53B": 2.57,
            "Qwen2.57B": 4.64, "Llama3.21B": 4.28, "Llama3.23B": 4.31}

out = {}
for m in MODELS:
    key = f"{m}-k{KS[m]['kstar_tau05']}"
    vals = {}
    for path in sorted(glob.glob(f"analysis/{key}_multi/results/*_v*_s*_results.json")):
        stem = os.path.basename(path).replace("_results.json", "")
        task = re.match(r"(.+?)_v\d+_s\d+$", stem).group(1)
        with open(path) as f:
            rec = json.load(f)
        dt = rec.get("layer1", {}).get("metrics", {}).get("DeltaT")
        if dt is not None:
            vals.setdefault(task, []).append(float(dt))
    all_dt = [v for vs in vals.values() for v in vs]
    med_all = float(np.median(all_dt))
    flag = "" if abs(med_all - EXPECTED[m]) < 0.011 else "  <-- MISMATCH"
    print(f"{m}: {len(all_dt)} circuits, model median {med_all:.2f} "
          f"(expected {EXPECTED[m]:.2f}){flag}", flush=True)
    out[m] = {t: {"median": round(float(np.median(vs)), 3), "n": len(vs)}
              for t, vs in vals.items()}

# Concordance of the task ordering across models (Kendall's W via the
# Friedman statistic), over the eight tasks present on all twelve models.
complete = [t for t in ORDER if all(t in out[m] for m in MODELS)]
rows = [[out[m][t]["median"] for t in complete] for m in MODELS]
chi2, p_fried = friedmanchisquare(*np.array(rows).T)
W = chi2 / (len(MODELS) * (len(complete) - 1))
print(f"Kendall's W over {len(complete)} complete tasks x {len(MODELS)} models: "
      f"W = {W:.2f}, Friedman chi2 = {chi2:.1f}, p = {p_fried:.2e}")
summary = {"complete_tasks": complete, "kendalls_W": round(float(W), 4),
           "friedman_chi2": round(float(chi2), 2), "p": float(p_fried)}
json.dump({"per_model": out, "concordance": summary},
          open("analysis/tables/dt_by_task_all12.json", "w"), indent=1)

ps.apply()
M = np.full((len(MODELS), len(ORDER)), np.nan)
for i, m in enumerate(MODELS):
    for j, t in enumerate(ORDER):
        if t in out[m]:
            M[i, j] = out[m][t]["median"]

# Every (model, task) median is positive at k*, so the house sequential
# map applies (a diverging map would waste half its range).
vmax = float(np.nanmax(M))
fig, ax = plt.subplots(figsize=(10.5, 11))
Mm = np.ma.masked_invalid(M)
im = ax.imshow(Mm, cmap=ps.seq_cmap(ps.DARK_BLUE), vmin=0.0, vmax=vmax)
ax.grid(False)
for side in ("top", "right", "bottom", "left"):
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color(ps.CHARCOAL)
    ax.spines[side].set_linewidth(0.8)
ax.set_xticks(range(len(ORDER)), [LBL[t] for t in ORDER], rotation=45,
              ha="right", fontsize=12)
ax.set_yticks(range(len(MODELS)), [NAME[m] for m in MODELS], fontsize=12)
for i in range(len(MODELS)):
    for j in range(len(ORDER)):
        if np.isnan(M[i, j]):
            ax.text(j, i, "--", ha="center", va="center", fontsize=10,
                    color=ps.CHARCOAL)
        else:
            frac = M[i, j] / vmax
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=10.5,
                    color="white" if frac > 0.55 else ps.CHARCOAL)
cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label(r"median $\Delta_T$", fontsize=12)
fig.savefig("paper/From_Functions_to_Logic/figures/fig_dt_by_task_12.pdf",
            dpi=200, bbox_inches="tight")
print("-> figures/fig_dt_by_task_12.pdf, analysis/tables/dt_by_task_all12.json")
