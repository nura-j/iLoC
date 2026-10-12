"""
plot_crossfamily_figures.py
===========================
Figures for the formal paper's cross-family scaled evaluation (App. C):

  fig_faith_budget.pdf   median faithfulness vs edge budget per model, with
                         the selected k* starred — the budget-matching story
  fig_separation.pdf     held-out IOI separation by method per model
                         (ILP vs feature vs WL vs attributed WL), log scale
  fig_transfer_heat.pdf  12x12 same-task retrieval heatmap (within delta)

Colours follow the shared figure schema (plot_style vivid). Reads the live
template-run pools and post-processing outputs; writes vector PDFs into
paper/From_Functions_to_Logic/figures/.
"""

import glob
import json
import re

import numpy as np
import matplotlib.pyplot as plt

import plot_style as ps

FIGDIR = "paper/From_Functions_to_Logic/figures"

# Colour schema shared across the paper figures (plot_style vivid)
BLUE = ps.DARK_BLUE
ORANGE = ps.ORANGE
SKY = ps.SKY_BLUE
VERMILION = ps.RED
BURGUNDY = ps.MAROON
GOLD = ps.GOLD
GREEN = ps.TEAL
MAGENTA = ps.PURPLE

ORDER = ["pythia14m", "gpt2", "Qwen2.50.5B", "pythia410m", "gpt2large",
         "pythia1b", "Qwen2.51.5B", "Llama3.21B", "Qwen2.53B", "Llama3.23B",
         "pythia6.9b", "Qwen2.57B"]
DISP = {"pythia14m": "Pythia-14M", "gpt2": "GPT-2", "Qwen2.50.5B": "Qwen-0.5B",
        "pythia410m": "Pythia-410M", "gpt2large": "GPT-2-large",
        "pythia1b": "Pythia-1B", "Qwen2.51.5B": "Qwen-1.5B",
        "Llama3.21B": "Llama-1B", "Qwen2.53B": "Qwen-3B",
        "Llama3.23B": "Llama-3B", "pythia6.9b": "Pythia-6.9B",
        "Qwen2.57B": "Qwen-7B"}
FAMILY_COLOR = {"pythia": BLUE, "gpt2": ORANGE, "qwen": GREEN,
                "llama": MAGENTA}


def fam_color(key):
    for fam, c in FAMILY_COLOR.items():
        if key.lower().startswith(fam):
            return c
    return ps.CHARCOAL


def sweep_curves():
    ks = json.load(open("analysis/tables/kstar_summary.json"))
    models = [m for m in ORDER if m != "pythia14m"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    seen_per_family = {}
    for key in models:
        per_k = {}
        for p in glob.glob(f"analysis/{key}_multi/results/*_v*_s*_results.json"):
            swp = (json.load(open(p))["layer1"]["metrics"] or {}).get("k_sweep") or {}
            for k, v in swp.items():
                per_k.setdefault(int(k), []).append(v["faithfulness"])
        xs = sorted(per_k)
        med = [float(np.median(per_k[k])) for k in xs]
        c = fam_color(key)
        i = seen_per_family.setdefault(c, 0)
        seen_per_family[c] += 1
        ls = ["-", "--", "-.", ":"][i % 4]
        ax.plot(xs, med, ls, color=c, lw=1.6, alpha=0.9,
                marker="o", ms=3.5, markeredgecolor="white",
                markeredgewidth=0.4, label=DISP[key])
        kstar = ks[key]["kstar_tau05"]
        if kstar in per_k:
            ax.plot([kstar], [float(np.median(per_k[kstar]))], "*",
                    color=GOLD, ms=14, markeredgecolor=ps.CHARCOAL,
                    markeredgewidth=0.6, zorder=6)
    ax.axhline(0.5, color=VERMILION, lw=1.1, ls="--", alpha=0.9)
    ax.text(52, 0.515, r"$\tau = 0.5$", color=VERMILION, fontsize=13)
    ax.set_xscale("log")
    ax.set_xlabel("edge budget $k$", fontsize=14)
    ax.set_ylabel("median faithfulness", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-0.05, 1.02)
    ax.legend(ncol=3, fontsize=10, loc="upper left", framealpha=0.92,
              frameon=True, edgecolor="#DDDDDD")
    fig.tight_layout()
    ps.save(fig, f"{FIGDIR}/fig_faith_budget", formats=("pdf",))
    print("fig_faith_budget.pdf")


def separation_bars():
    ks = json.load(open("analysis/tables/kstar_summary.json"))
    models = [m for m in ORDER if m != "pythia14m"]
    methods = [("ilp", "ILP", BLUE), ("feat_euclidean", "Feature", ORANGE),
               ("wl", "WL", BURGUNDY), ("wl_attributed", "Attr. WL", GREEN)]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    width = 0.2
    xs = np.arange(len(models))
    for j, (mkey, label, color) in enumerate(methods):
        means, stds = [], []
        for key in models:
            rep = json.load(open(f"analysis/{key}-k{ks[key]['kstar_tau05']}_multi"
                                 f"/heldout/heldout_by_variant_report.json"))
            d = (rep["aggregate"]["separation"].get(mkey)
                 if "ioi" in rep["tasks"] else None)
            m = d.get("mean") if d else None
            ok = m is not None and not np.isnan(m)
            means.append(m if ok else np.nan)
            stds.append(d.get("std", 0.0) if ok else 0.0)
        ax.bar(xs + (j - 1.5) * width, means, width, yerr=stds, capsize=1.5,
               color=color, label=label, error_kw={"lw": 0.7})
    ax.axhline(1.0, color=ps.CHARCOAL, lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_ylabel("held-out IOI separation", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xticks(xs, [DISP[m] for m in models], rotation=40, ha="right",
                  fontsize=12)
    ax.legend(ncol=4, fontsize=11, loc="upper left")
    fig.tight_layout()
    ps.save(fig, f"{FIGDIR}/fig_separation", formats=("pdf",))
    print("fig_separation.pdf")


def transfer_heatmap():
    tm = json.load(open("analysis/transfer/transfer_matrix_kstar.json"))
    strip = lambda k: re.sub(r"-k\d+$", "", k)
    idx = {m: i for i, m in enumerate(ORDER)}
    M = np.full((12, 12), np.nan)
    for pair, v in tm["pairs"].items():
        a, b = (strip(x) for x in pair.split("__"))
        M[idx[a], idx[b]] = v["within_delta"]
    np.fill_diagonal(M, 1.0)
    side = 6.4
    fig, ax = plt.subplots(figsize=(side + 1.2, side))
    im = ps.heatmap(ax, M, [DISP[m] for m in ORDER], vmax=1.0,
                    cmap=ps.seq_cmap(BLUE),
                    title="same-task signature within "
                          r"$\delta_{\mathrm{ILP}}$ (source $\to$ target)",
                    tick_fs=13, annot_fs=11, title_fs=15)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=12)
    fig.tight_layout()
    ps.save(fig, f"{FIGDIR}/fig_transfer_heat", formats=("pdf",))
    print("fig_transfer_heat.pdf")




def variance_bars():
    VS = json.load(open("analysis/tables/variance_sensitivity_12.json"))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    xs = np.arange(len(ORDER))
    means, stds, cols = [], [], []
    for m in ORDER:
        r = [v for v in VS["variance"][m].values() if v]
        means.append(np.mean(r)); stds.append(np.std(r)); cols.append(fam_color(m))
    ax.bar(xs, means, 0.62, yerr=stds, capsize=2, color=cols,
           error_kw={"lw": 0.8})
    ax.axhline(1.0, color=ps.CHARCOAL, lw=0.9, ls=":")
    ax.set_ylabel("between / within WL distance", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xticks(xs, [DISP[m] for m in ORDER], rotation=40, ha="right",
                  fontsize=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Pythia"),
                       Patch(color=ORANGE, label="GPT-2"),
                       Patch(color=GREEN, label="Qwen2.5"),
                       Patch(color=MAGENTA, label="Llama-3.2")],
              ncol=4, fontsize=11, loc="upper left")
    fig.tight_layout()
    ps.save(fig, f"{FIGDIR}/appendix_within_task_variance_12", formats=("pdf",))
    print("appendix_within_task_variance_12.pdf")


def sensitivity_curves():
    dists = json.load(open("analysis/tables/delta_sweep_dists.json"))
    FAMC = {"Pythia": BLUE, "GPT-2": ORANGE, "Qwen": GREEN, "Llama": MAGENTA}
    xs = [round(x, 2) for x in np.arange(0.02, 0.62, 0.02)]
    same = [(d, f) for d, s_, f in dists if s_]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    rec = [np.mean([d < x for d, _ in same]) for x in xs]
    acc = [np.mean([d < x for d, _, _ in dists]) for x in xs]
    ax.plot(xs, rec, "-", color=ps.CHARCOAL, lw=2.6,
            label="same-task acceptance (recall)")
    ax.plot(xs, acc, "-", color=GOLD, lw=2.4, label="all candidates accepted")
    for fam, c in FAMC.items():
        r = [np.mean([d < x for d, f in same if f == fam]) for x in xs]
        ax.plot(xs, r, "--", color=c, lw=1.4, alpha=0.9,
                label=f"recall, {fam} source")
    ax.axvline(0.30, color=VERMILION, lw=1.2, ls=":", alpha=0.9)
    ax.text(0.305, 0.06, r"$\delta_{\mathrm{ILP}}{=}0.30$", color=VERMILION,
            fontsize=12)
    ax.set_xlabel(r"acceptance threshold $\delta$", fontsize=14)
    ax.set_ylabel("fraction accepted", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=10, loc="lower right", frameon=True, framealpha=0.92,
              edgecolor="#DDDDDD")
    fig.tight_layout()
    ps.save(fig, f"{FIGDIR}/appendix_sensitivity_12", formats=("pdf",))
    print("appendix_sensitivity_12.pdf")


if __name__ == "__main__":
    ps.apply()
    sweep_curves()
    separation_bars()
    transfer_heatmap()
    variance_bars()
    sensitivity_curves()
