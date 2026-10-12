"""
plot_cfs_figures.py
===================
CFS/DLA figures in the house palette from the stored layer-2 CFS records.

Reads every results JSON with a populated layer2.cfs.components block:
  analysis/{model}/results/{task}_results.json          (primary trio)
  analysis/cfs_scaled/{model}/{task}.json               (scaled run, when present)

Outputs (paper/From_Functions_to_Logic/figures/):
  fig_dla_scatter_all.{pdf,png}   one panel per model: mean |DLA| attn vs MLP
  fig_dla_cluster_2d.{pdf,png}    PCA of per-circuit DLA feature vectors (2D)
  fig_dla_cluster_3d.{pdf,png}    same embedding, first three components (3D)

Usage: python3.11 plot_cfs_figures.py  (clustermap needs scipy)
"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps

FIGDIR = "paper/From_Functions_to_Logic/figures"
NAME = {"pythia14m": "Pythia-14M", "pythia410m": "Pythia-410M",
        "pythia1b": "Pythia-1B", "pythia6.9b": "Pythia-6.9B",
        "gpt2": "GPT-2", "gpt2large": "GPT-2-large",
        "Qwen2.50.5B": "Qwen2.5-0.5B", "Qwen2.51.5B": "Qwen2.5-1.5B",
        "Qwen2.53B": "Qwen2.5-3B", "Qwen2.57B": "Qwen2.5-7B",
        "Llama3.21B": "LLaMA-1B", "Llama3.23B": "LLaMA-3B"}
TASK_LABEL = {"ioi": "IOI", "greaterthan": "GT", "location": "Location",
              "time": "Time", "path": "Path", "beneficiary": "Beneficiary",
              "goal": "Goal", "instrument": "Instrument", "source": "Source",
              "topic": "Topic"}
TASK_COLOR = {"ioi": ps.RED, "greaterthan": ps.ORANGE, "location": ps.DARK_BLUE,
              "time": ps.TEAL, "path": ps.SKY_BLUE, "beneficiary": ps.PURPLE,
              "goal": ps.MAROON, "instrument": ps.GOLD, "source": ps.CHARCOAL,
              "topic": "#7A9E43"}
MODEL_MARK = {"pythia14m": "o", "pythia1b": "s", "Llama3.21B": "^",
              "pythia410m": "D", "gpt2": "v", "gpt2large": "P",
              "Qwen2.50.5B": "X", "Qwen2.51.5B": "*", "Qwen2.53B": "<",
              "Qwen2.57B": ">", "pythia6.9b": "h", "Llama3.23B": "d"}


def load_records():
    """[(model, task, comps_dict)] for every stored CFS block."""
    out = []
    for p in sorted(glob.glob("analysis/*/results/*_results.json")):
        m = re.match(r"analysis/([^/]+)/results/([a-z]+)_results\.json$", p)
        if not m or "_multi" in m.group(1) or "-k" in m.group(1):
            continue
        try:
            comps = (json.load(open(p)).get("layer2") or {}).get(
                "cfs", {}).get("components", {})
        except Exception:
            continue
        if comps:
            out.append((m.group(1), m.group(2), comps, "primary"))
    for p in sorted(glob.glob("analysis/cfs_scaled/*/*.json")):
        model, task = p.split("/")[-2], os.path.basename(p)[:-5]
        comps = json.load(open(p)).get("components", {})
        if comps:
            out.append((model, task, comps, "scaled"))
    return out


def features(comps):
    """Per-circuit DLA feature vector."""
    attn = [v["dla_score"] for k, v in comps.items() if k.startswith("a")]
    mlp = [v["dla_score"] for k, v in comps.items() if k.startswith("m")]
    alla = np.array(attn + mlp)
    aab = np.abs(attn) if attn else np.array([0.0])
    mab = np.abs(mlp) if mlp else np.array([0.0])
    marg = np.mean([v.get("causally_marginal", False) for v in comps.values()])
    return dict(
        attn_mean=float(np.mean(aab)), mlp_mean=float(np.mean(mab)),
        attn_std=float(np.std(aab)), mlp_std=float(np.std(mab)),
        attn_max=float(np.max(aab)), mlp_max=float(np.max(mab)),
        frac_neg=float(np.mean(alla < 0)) if len(alla) else 0.0,
        frac_marginal=float(marg))


def scatter_panels(records):
    models = sorted({m for m, _, _, _ in records},
                    key=lambda m: list(NAME).index(m))
    fig, axes = plt.subplots(1, len(models),
                             figsize=(4.8 * len(models), 4.6))
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        rows = [(t, features(c)) for m, t, c, _ in records if m == model]
        mx = max(max(f["attn_mean"], f["mlp_mean"]) for _, f in rows) * 1.2
        ax.plot([0, mx], [0, mx], color=ps.CHARCOAL, alpha=0.35,
                linewidth=1.0, linestyle="--", zorder=1)
        OFF = {("pythia1b", "location"): (8, 6), ("pythia1b", "path"): (-44, -18),
               ("pythia1b", "time"): (-40, 8), ("pythia14m", "time"): (8, -16),
               ("pythia14m", "ioi"): (8, 6)}
        for t, f in rows:
            ax.scatter(f["attn_mean"], f["mlp_mean"], s=110,
                       color=TASK_COLOR[t], zorder=4,
                       edgecolors="white", linewidths=0.9)
            dx, dy = OFF.get((model, t), (6, 6))
            ax.annotate(TASK_LABEL[t], (f["attn_mean"], f["mlp_mean"]),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=13, color=TASK_COLOR[t], fontweight="bold")
        ax.set_xlim(-0.05 * mx, mx)
        ax.set_ylim(-0.05 * mx, mx)
        ax.set_xlabel(r"Mean $|\delta_v|$ — attention heads", fontsize=14)
        if ax is axes[0]:
            ax.set_ylabel(r"Mean $|\delta_v|$ — MLP layers", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_title(NAME[model], fontsize=16)
    fig.tight_layout()
    return ps.save(fig, f"{FIGDIR}/fig_dla_scatter_all")


def pca(X):
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-12)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    evr = S**2 / np.sum(S**2)
    return Xc @ Vt.T, evr


def profile_features(f):
    """Scale-free DLA profile shape (raw magnitudes differ across models)."""
    tot = f["attn_mean"] + f["mlp_mean"] + 1e-12
    return [f["attn_mean"] / tot,                       # attention share
            f["attn_std"] / (f["attn_mean"] + 1e-12),   # attn dispersion
            f["mlp_std"] / (f["mlp_mean"] + 1e-12),     # MLP dispersion
            f["attn_max"] / (f["attn_mean"] + 1e-12),   # attn peakedness
            f["mlp_max"] / (f["mlp_mean"] + 1e-12),     # MLP peakedness
            f["frac_neg"], f["frac_marginal"]]


def cluster_figures(records):
    rows = [(m, t, features(c)) for m, t, c, _ in records]
    X = np.array([profile_features(f) for _, _, f in rows])
    Z, evr = pca(X)

    def draw(ax, dims):
        for (m, t, _), z in zip(rows, Z):
            args = dict(s=130, color=TASK_COLOR[t], marker=MODEL_MARK[m],
                        edgecolors="white", linewidths=0.9)
            ax.scatter(*[z[d] for d in dims], **args)

    # 2D
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    draw(ax, (0, 1))
    ax.set_xlabel(f"PC1 ({evr[0]:.0%} var.)", fontsize=14)
    ax.set_ylabel(f"PC2 ({evr[1]:.0%} var.)", fontsize=14)
    ax.tick_params(labelsize=12)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                          color=TASK_COLOR[t], label=TASK_LABEL[t])
               for t in dict.fromkeys(t for _, t, _ in rows)]
    handles += [plt.Line2D([], [], marker=MODEL_MARK[m], linestyle="",
                           markersize=9, color=ps.CHARCOAL, label=NAME[m])
                for m in dict.fromkeys(m for m, _, _ in rows)]
    ax.legend(handles=handles, fontsize=9.5, ncol=3, frameon=False,
              loc="best")
    fig.tight_layout()
    ps.save(fig, f"{FIGDIR}/fig_dla_cluster_2d")

    # 3D
    fig = plt.figure(figsize=(8.4, 7.0))
    ax = fig.add_subplot(projection="3d")
    draw(ax, (0, 1, 2))
    ax.set_xlabel(f"PC1 ({evr[0]:.0%})", fontsize=13, labelpad=8)
    ax.set_ylabel(f"PC2 ({evr[1]:.0%})", fontsize=13, labelpad=8)
    ax.set_zlabel(f"PC3 ({evr[2]:.0%})", fontsize=13, labelpad=8)
    ax.tick_params(labelsize=10)
    ax.view_init(elev=18, azim=-60)
    ax.legend(handles=handles, fontsize=10, ncol=2, frameon=False,
              loc="upper left", bbox_to_anchor=(0.0, 1.02))
    fig.tight_layout()
    ps.save(fig, f"{FIGDIR}/fig_dla_cluster_3d")


def scatter_grid(records, ncols=4):
    """All-model grid: one panel per model, task-coloured points, no text."""
    models = [m for m in NAME if any(r[0] == m for r in records)]
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.4 * ncols, 4.1 * nrows))
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    for ax, model in zip(axes.flat, models):
        rows = [(t, features(c)) for m, t, c, _ in records if m == model]
        mx = max(max(f["attn_mean"], f["mlp_mean"]) for _, f in rows) * 1.15
        mx = max(mx, 1e-3)
        ax.plot([0, mx], [0, mx], color=ps.CHARCOAL, alpha=0.35,
                linewidth=1.0, linestyle="--", zorder=1)
        for t, f in rows:
            ax.scatter(f["attn_mean"], f["mlp_mean"], s=100,
                       color=TASK_COLOR[t], zorder=4,
                       edgecolors="white", linewidths=0.9)
        ax.set_xlim(-0.05 * mx, mx)
        ax.set_ylim(-0.05 * mx, mx)
        ax.tick_params(labelsize=11)
        ax.set_title(NAME[model], fontsize=15)
    for ax in axes[-1] if nrows > 1 else [axes.flat[i] for i in range(len(models))]:
        ax.set_xlabel(r"Mean $|\delta_v|$ — attn", fontsize=13)
    for r in range(nrows):
        axes[r][0].set_ylabel(r"Mean $|\delta_v|$ — MLP", fontsize=13)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=10,
                          color=TASK_COLOR[t], label=TASK_LABEL[t])
               for t in TASK_COLOR
               if any(r[1] == t for r in records)]
    fig.legend(handles=handles, fontsize=13, ncol=5, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.045))
    fig.tight_layout()
    return ps.save(fig, f"{FIGDIR}/fig_dla_scatter_12")


ps.apply()
records = load_records()
print(f"{len(records)} CFS records:",
      sorted({m for m, _, _, _ in records}))


# ── Rich (full) CFS features: depth-resolved DLA + role mass ────────────
LAYERS = {"gpt2": 12, "gpt2large": 36, "pythia14m": 6, "pythia410m": 24,
          "pythia1b": 16, "pythia6.9b": 32, "Qwen2.50.5B": 24,
          "Qwen2.51.5B": 28, "Qwen2.53B": 36, "Qwen2.57B": 28,
          "Llama3.21B": 16, "Llama3.23B": 28}
ROLES6 = ["subject", "object", "verb", "function", "entity", "other"]
FAM_OF = lambda m: ("Pythia" if m.startswith("pythia") else
                    "GPT-2" if m.startswith("gpt2") else
                    "Qwen" if m.startswith("Q") else "Llama")
FAM_COLOR = {"Pythia": ps.DARK_BLUE, "GPT-2": ps.ORANGE,
             "Qwen": ps.TEAL, "Llama": ps.PURPLE}
TFAM_OF = lambda t: ("selection" if t == "ioi" else
                     "comparison" if t == "greaterthan" else "binding")


def rich_features(model, comps):
    """Depth-resolved, scale-free CFS profile (17 dims)."""
    L = LAYERS[model]
    a_prof, m_prof = np.zeros(4), np.zeros(4)
    neg = tot = marg = 0
    for name, v in comps.items():
        s = abs(v["dla_score"]); tot += 1
        neg += v["dla_score"] < 0
        marg += v["causally_marginal"]
        mm = re.match(r"[am](\d+)", name)
        if not mm:
            continue
        b = min(3, int(4 * int(mm.group(1)) / max(1, L)))
        (a_prof if name.startswith("a") else m_prof)[b] += s
    ashare = a_prof.sum() / (a_prof.sum() + m_prof.sum() + 1e-12)
    return np.concatenate([[ashare], a_prof / (a_prof.sum() + 1e-12),
                           m_prof / (m_prof.sum() + 1e-12),
                           ling_dist(comps),
                           [neg / max(tot, 1), marg / max(tot, 1)]])


FEATURE_LABELS = (["attn share"] + [f"attn Q{i}" for i in range(1, 5)]
                  + [f"MLP Q{i}" for i in range(1, 5)]
                  + ["subj", "obj", "verb", "func", "ent", "oth"]
                  + ["neg frac", "marginal"])


def clustermap(records):
    """Ward dendrogram + z-scored feature heatmap with task/family bands."""
    from scipy.cluster import hierarchy
    rows = [(m, t) for m, t, _, _ in records]
    X = np.array([rich_features(m, c) for m, t, c, _ in records])
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Z = hierarchy.linkage(Xs, method="ward")
    order = hierarchy.dendrogram(Z, no_plot=True)["leaves"]

    fig = plt.figure(figsize=(12.5, 15.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.1, 0.32, 0.32, 8.0],
                          wspace=0.03)
    axd = fig.add_subplot(gs[0])
    hierarchy.dendrogram(Z, orientation="left", ax=axd, no_labels=True,
                         color_threshold=0, above_threshold_color=ps.CHARCOAL)
    axd.invert_yaxis()
    axd.axis("off")

    n = len(rows)
    axt = fig.add_subplot(gs[1])
    axf = fig.add_subplot(gs[2])
    for ax, colours, title in (
            (axt, [TASK_COLOR[rows[i][1]] for i in order], "task"),
            (axf, [FAM_COLOR[FAM_OF(rows[i][0])] for i in order], "family")):
        for y, c in enumerate(colours):
            ax.add_patch(plt.Rectangle((0, y), 1, 1, color=c))
        ax.set_xlim(0, 1); ax.set_ylim(n, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, -0.004 * n, title, fontsize=12, rotation=45,
                ha="right", va="top", transform=ax.get_xaxis_transform())
        for side in ax.spines.values():
            side.set_visible(False)

    axh = fig.add_subplot(gs[3])
    V = np.clip(Xs[order], -2.5, 2.5)
    from matplotlib.colors import LinearSegmentedColormap
    div = LinearSegmentedColormap.from_list(
        "house_div", [ps.DARK_BLUE, "#FFFFFF", ps.RED])
    im = axh.imshow(V, aspect="auto", cmap=div, vmin=-2.5, vmax=2.5)
    axh.set_yticks([])
    axh.set_xticks(range(len(FEATURE_LABELS)), FEATURE_LABELS,
                   rotation=45, ha="right", fontsize=13)
    axh.grid(False)
    cb = fig.colorbar(im, ax=axh, fraction=0.025, pad=0.01)
    cb.set_label("feature z-score", fontsize=13)
    cb.ax.tick_params(labelsize=11)
    handles = ([plt.Line2D([], [], marker="s", linestyle="", markersize=11,
                           color=TASK_COLOR[t], label=TASK_LABEL[t])
                for t in TASK_COLOR]
               + [plt.Line2D([], [], marker="s", linestyle="", markersize=11,
                             color=FAM_COLOR[f], label=f) for f in FAM_COLOR])
    fig.legend(handles=handles, fontsize=12, ncol=7, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.015))
    return ps.save(fig, f"{FIGDIR}/fig_dla_clustermap")


def depth_profiles(records):
    """Mean |DLA| mass by depth quartile: rows attn/MLP, cols model family."""
    fams = ["GPT-2", "Pythia", "Qwen", "Llama"]
    tfams = [("selection", ps.RED), ("comparison", ps.ORANGE),
             ("binding", ps.DARK_BLUE)]
    prof = {}
    for m, t, c, _ in records:
        X = rich_features(m, c)
        prof.setdefault((FAM_OF(m), TFAM_OF(t), "a"), []).append(X[1:5])
        prof.setdefault((FAM_OF(m), TFAM_OF(t), "m"), []).append(X[5:9])
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.4), sharey="row")
    for j, fam in enumerate(fams):
        for i, kind in enumerate("am"):
            ax = axes[i][j]
            for tf, col in tfams:
                vals = prof.get((fam, tf, kind))
                if not vals:
                    continue
                mean = np.mean(vals, axis=0)
                sd = np.std(vals, axis=0)
                x = np.arange(4)
                ax.plot(x, mean, color=col, marker="o", markersize=7,
                        linewidth=2.2, label=tf)
                ax.fill_between(x, mean - sd, mean + sd, color=col,
                                alpha=0.13, linewidth=0)
            ax.set_xticks(range(4), ["Q1", "Q2", "Q3", "Q4"], fontsize=12)
            ax.tick_params(labelsize=12)
            if i == 0:
                ax.set_title(fam, fontsize=16)
        axes[1][j].set_xlabel("depth quartile", fontsize=14)
    axes[0][0].set_ylabel("attn $|\\delta_v|$ share", fontsize=14)
    axes[1][0].set_ylabel("MLP $|\\delta_v|$ share", fontsize=14)
    axes[0][0].legend(fontsize=12, frameon=False)
    fig.tight_layout()
    return ps.save(fig, f"{FIGDIR}/fig_dla_depth_profiles")


primary = [r for r in records if r[3] == "primary"]
scaled = [r for r in records if r[3] == "scaled"]
scatter_panels(primary)
if scaled:
    scatter_grid(scaled)
cluster_figures(scaled if scaled else primary)


ROLE_COLOR = {"subject": ps.DARK_BLUE, "object": ps.TEAL, "verb": ps.SKY_BLUE,
              "function": ps.ORANGE, "entity": ps.GOLD, "other": ps.RED}
TASK_ORDER = ["ioi", "greaterthan", "beneficiary", "goal", "instrument",
              "location", "path", "source", "time", "topic"]


def ling_dist(comps):
    """Attribution-weighted attention distribution over linguistic roles.

    P(role) = sum_v |delta_v| pi_v(role) / sum_v |delta_v| over causal
    components with a stored profile; the full per-component profile is
    used, not its argmax.
    """
    num = np.zeros(len(ROLES6))
    den = 0.0
    for name, v in comps.items():
        if v["causally_marginal"]:
            continue
        prof = v.get("attention_profile") or {}
        w = np.array([prof.get(r, 0.0) for r in ROLES6])
        if w.sum() <= 0:
            continue
        s = abs(v["dla_score"])
        num += s * (w / w.sum())
        den += s
    return num / (den + 1e-12)


def role_mass(model, comps):
    return ling_dist(comps)


def role_bars(records):
    """Stacked role-mass bars per task, one panel per model."""
    models = [m for m in NAME if any(r[0] == m for r in records)]
    ncols, nrows = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.9 * nrows))
    stats = {"ioi_other": [], "gt_entity": []}
    for ax, model in zip(axes.flat, models):
        rows = [(t, ling_dist(c)) for m, t, c, _ in records
                if m == model]
        rows.sort(key=lambda r: TASK_ORDER.index(r[0]))
        ys = np.arange(len(rows))
        for k, (task, mass) in enumerate(rows):
            left = 0.0
            for ri, role in enumerate(ROLES6):
                ax.barh(k, mass[ri], left=left, height=0.72,
                        color=ROLE_COLOR[role],
                        edgecolor="white", linewidth=0.4)
                left += mass[ri]
            if task == "ioi":
                stats["ioi_other"].append(mass[ROLES6.index("other")])
            if task == "greaterthan":
                stats["gt_entity"].append(mass[ROLES6.index("entity")])
        ax.set_yticks(ys, [TASK_LABEL[t] for t, _ in rows], fontsize=12)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=11)
        ax.set_title(NAME[model], fontsize=15)
        ax.grid(axis="y", alpha=0)
    for ax in axes[-1]:
        ax.set_xlabel("attribution-weighted linguistic-role distribution",
                      fontsize=13)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=12,
                          color=ROLE_COLOR[r], label=r) for r in ROLES6]
    fig.legend(handles=handles, fontsize=13, ncol=6, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.035))
    fig.tight_layout()
    print(f"  IOI other-share median {np.median(stats['ioi_other']):.2f} "
          f"(range {min(stats['ioi_other']):.2f}-{max(stats['ioi_other']):.2f})")
    print(f"  GT entity-share median {np.median(stats['gt_entity']):.2f} "
          f"(range {min(stats['gt_entity']):.2f}-{max(stats['gt_entity']):.2f})")
    return ps.save(fig, f"{FIGDIR}/fig_cfs_roles_12")




def task_profile_figure():
    """Task-specific role profiles (primary trio; per-family vocabularies).

    Reads analysis/{model}/cfs/{task}_task_profiles.json with DLA weights
    from {task}_cfs.json; the vocabularies differ per task family, so each
    family gets its own panel and the linguistic profiles stay separate.
    """
    trio = ["pythia14m", "pythia1b", "Llama3.21B"]
    groups = {"IOI": ["ioi"], "GT": ["greaterthan"],
              "Binding": ["location", "time", "path"]}
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.0),
                             gridspec_kw={"width_ratios": [1, 1, 2.1]})
    for ax, (fam, tasks) in zip(axes, groups.items()):
        bars, labels = [], []
        vocab = None
        for model in trio:
            for task in tasks:
                pp = f"analysis/{model}/cfs/{task}_task_profiles.json"
                cp = f"analysis/{model}/cfs/{task}_cfs.json"
                if not (os.path.isfile(pp) and os.path.isfile(cp)):
                    continue
                prof = json.load(open(pp))
                comps = json.load(open(cp))["components"]
                vocab = [r for r in prof["task_roles"]]
                num = np.zeros(len(vocab)); den = 0.0
                for name, pv in prof["profiles"].items():
                    cv = comps.get(name)
                    if cv is None or cv["causally_marginal"]:
                        continue
                    w = np.array([pv.get(r, 0.0) for r in vocab])
                    if w.sum() <= 0:
                        continue
                    s = abs(cv["dla_score"])
                    num += s * (w / w.sum()); den += s
                if den > 0:
                    bars.append(num / den)
                    labels.append(f"{NAME[model]}"
                                  + (f" {TASK_LABEL[task]}" if len(tasks) > 1
                                     else ""))
        if not bars:
            ax.set_visible(False)
            continue
        cols = ps.cycle(len(vocab))
        for k, (dist, lab) in enumerate(zip(bars, labels)):
            left = 0.0
            for ri in range(len(vocab)):
                ax.barh(k, dist[ri], left=left, height=0.7, color=cols[ri],
                        edgecolor="white", linewidth=0.4)
                left += dist[ri]
        ax.set_yticks(range(len(labels)), labels, fontsize=12)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=11)
        ax.set_title(f"{fam} vocabulary", fontsize=15)
        handles = [plt.Line2D([], [], marker="s", linestyle="",
                              markersize=10, color=cols[ri],
                              label=vocab[ri])
                   for ri in range(len(vocab))]
        ax.legend(handles=handles, fontsize=10, frameon=False,
                  loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
        ax.set_xlabel("attribution-weighted task-role distribution",
                      fontsize=12)
    fig.tight_layout()
    return ps.save(fig, f"{FIGDIR}/fig_cfs_task_profiles")


if scaled:
    clustermap(scaled)
    depth_profiles(scaled)
    role_bars(scaled)
task_profile_figure()
print("-> fig_dla_scatter_all, fig_dla_scatter_12, "
      "fig_dla_cluster_2d, fig_dla_cluster_3d")
