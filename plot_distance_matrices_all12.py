"""
plot_distance_matrices_all12.py
===============================
ILP signature distance matrices for ALL twelve models of the k* pools,
replacing the nine-panel appendix figure so that the per-model matrix
figure covers the full inventory rather than a primary trio. Identical protocol to
plot_distance_matrices_scaled.py: signatures learned from each model's
k* pool with the registry protocol of the transfer matrix, pairwise
distances via ilp_signature_distance, house palette. One 4x3-panel figure.

Clause selection breaks genuine F1 ties by hypothesis-generation order,
which depends on set iteration (see ilp_arch.py, Template 5); the archived
nine-panel matrices were produced under PYTHONHASHSEED=0, so this script
re-executes itself with that seed to reproduce them exactly.

Usage: python3 plot_distance_matrices_all12.py
Output:
  paper/From_Functions_to_Logic/figures/appendix_distance_full_12.pdf
  analysis/tables/distance_matrices_kstar_12.json
"""

import json
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps
from run_transfer_matrix import load_task_circuits, learn_signatures
from trace.mechint.storage.ilp.ilp_signature_distance import ilp_signature_distance

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
LBL = {"ioi": "IOI", "greaterthan": "GT", "beneficiary": "Ben", "goal": "Goal",
       "instrument": "Ins", "location": "Loc", "path": "Path", "source": "Src",
       "time": "Time", "topic": "Top"}
ORDER = ["ioi", "greaterthan", "beneficiary", "goal", "instrument", "location",
         "path", "source", "time", "topic"]

out = {}
ps.apply()
fig, axes = plt.subplots(4, 3, figsize=(19, 24))
cmap = ps.seq_cmap(ps.DARK_BLUE)

for ax, m in zip(axes.flat, MODELS):
    key = f"{m}-k{KS[m]['kstar_tau05']}"
    reg = learn_signatures(key, load_task_circuits(key, "variant"))
    tasks = [t for t in ORDER if t in reg]
    n = len(tasks)
    D = np.zeros((n, n))
    for i, a in enumerate(tasks):
        for j, b in enumerate(tasks):
            if i != j:
                D[i, j] = float(ilp_signature_distance(reg[a], reg[b]))
    out[m] = {"tasks": tasks, "matrix": D.round(4).tolist()}

    im = ax.imshow(D, cmap=cmap, vmin=0.0, vmax=max(0.35, D.max()))
    ax.grid(False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(ps.CHARCOAL)
        ax.spines[side].set_linewidth(0.8)
    ax.set_xticks(range(n), [LBL[t] for t in tasks], rotation=45,
                  ha="right", fontsize=11)
    ax.set_yticks(range(n), [LBL[t] for t in tasks], fontsize=11)
    vmax = max(0.35, D.max())
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{D[i, j]:.2f}", ha="center", va="center",
                        fontsize=8.5,
                        color="white" if D[i, j] / vmax > 0.55 else ps.CHARCOAL)
    ax.set_title(f"{NAME[m]} ($k^*$={KS[m]['kstar_tau05']})", fontsize=14)
    print(f"{m}: {n} tasks, max d {D.max():.2f}", flush=True)

fig.subplots_adjust(hspace=0.35, wspace=0.3)
fig.savefig("paper/From_Functions_to_Logic/figures/appendix_distance_full_12.pdf",
            dpi=200, bbox_inches="tight")
json.dump(out, open("analysis/tables/distance_matrices_kstar_12.json", "w"),
          indent=1)
print("-> figures/appendix_distance_full_12.pdf, "
      "analysis/tables/distance_matrices_kstar_12.json")
