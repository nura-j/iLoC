#!/usr/bin/env python3
"""Significance tests for the twelve-model CFS claims.

Replicates the feature pipeline of COMPASS/plot_cfs_figures.py exactly
(features(), rich_features(), ling_dist(), LAYERS, ROLES6) over the 115
k* s0 circuits in analysis/cfs_scaled/.

Test A: IOI top-rank on DLA attention share, exact Poisson-binomial tail
        under a uniform-rank null (P(top)=1/k_m per model).
Test B: GT top-rank on mean MLP |delta| magnitude, same null.
Test C: cross-family IOI cohesion in the z-scored 17-dim clustermap
        feature space: observed mean pairwise distance among the 11 IOI
        circuits vs 10,000 random same-size subsets (seed 42).
        Same for the GT circuits.
"""
import os
import json, glob, os, re
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
LAYERS = {"gpt2": 12, "gpt2large": 36, "pythia14m": 6, "pythia410m": 24,
          "pythia1b": 16, "pythia6.9b": 32, "Qwen2.50.5B": 24,
          "Qwen2.51.5B": 28, "Qwen2.53B": 36, "Qwen2.57B": 28,
          "Llama3.21B": 16, "Llama3.23B": 28}
ROLES6 = ["subject", "object", "verb", "function", "entity", "other"]
BINDING = {"beneficiary", "goal", "instrument", "location", "path",
           "source", "time", "topic"}

def ling_dist(comps):
    num = np.zeros(len(ROLES6)); den = 0.0
    for name, v in comps.items():
        if v["causally_marginal"]:
            continue
        prof = v.get("attention_profile") or {}
        w = np.array([prof.get(r, 0.0) for r in ROLES6])
        if w.sum() <= 0:
            continue
        s = abs(v["dla_score"])
        num += s * (w / w.sum()); den += s
    return num / (den + 1e-12)

def rich_features(model, comps):
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

def mean_mags(comps):
    attn = [abs(v["dla_score"]) for k, v in comps.items() if k.startswith("a")]
    mlp = [abs(v["dla_score"]) for k, v in comps.items() if k.startswith("m")]
    return (float(np.mean(attn)) if attn else 0.0,
            float(np.mean(mlp)) if mlp else 0.0)

records = []
for p in sorted(glob.glob(f"{ROOT}/analysis/cfs_scaled/*/*.json")):
    model, task = p.split("/")[-2], os.path.basename(p)[:-5]
    comps = json.load(open(p)).get("components", {})
    if comps:
        records.append((model, task, comps))
print(f"circuits: {len(records)}")

# per-model attention share (mass-based, as in rich_features) and MLP magnitude
share, mlpmag = {}, {}
for m, t, c in records:
    f = rich_features(m, c)
    share.setdefault(m, {})[t] = f[0]
    mlpmag.setdefault(m, {})[t] = mean_mags(c)[1]

def poisson_binomial_tail(ps_list, k_obs):
    """Exact P(X >= k_obs) for independent Bernoulli(p_i)."""
    n = len(ps_list)
    dp = np.zeros(n + 1); dp[0] = 1.0
    for p in ps_list:
        dp[1:] = dp[1:] * (1 - p) + dp[:-1] * p
        dp[0] *= (1 - p)
    return float(dp[k_obs:].sum())

def top_rank_test(metric, target):
    models = [m for m in metric if target in metric[m]]
    succ, null_ps, detail = 0, [], []
    for m in models:
        k = len(metric[m])
        top = max(metric[m], key=metric[m].get)
        succ += top == target
        null_ps.append(1.0 / k)
        detail.append((m, k, top, round(metric[m][target], 3)))
    p = poisson_binomial_tail(null_ps, succ)
    return len(models), succ, p, detail

nA, sA, pA, dA = top_rank_test(share, "ioi")
print(f"\nTest A  IOI top attention share: {sA}/{nA} models, "
      f"Poisson-binomial P(X>={sA}) = {pA:.2e}")
for m, k, top, val in dA:
    print(f"   {m:14s} k={k:2d} top={top:12s} ioi_share={val}")
ioi_shares = [share[m]["ioi"] for m in share if "ioi" in share[m]]
bind_shares = [v for m in share for t, v in share[m].items() if t in BINDING]
print(f"   median IOI share {np.median(ioi_shares):.2f}, "
      f"median binding share {np.median(bind_shares):.2f}")

nB, sB, pB, dB = top_rank_test(mlpmag, "greaterthan")
print(f"\nTest B  GT top MLP magnitude: {sB}/{nB} models, "
      f"Poisson-binomial P(X>={sB}) = {pB:.2e}")
for m, k, top, val in dB:
    print(f"   {m:14s} k={k:2d} top={top}")

# Test C: cohesion in z-scored 17-dim space
X = np.array([rich_features(m, c) for m, t, c in records])
Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
tasks = np.array([t for _, t, _ in records])

def cohesion(task, n_perm=10000, seed=42):
    idx = np.where(tasks == task)[0]
    def mean_pd(ix):
        sub = Xs[ix]
        d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
        iu = np.triu_indices(len(ix), 1)
        return d[iu].mean()
    obs = mean_pd(idx)
    rng = np.random.RandomState(seed)
    null = np.array([mean_pd(rng.choice(len(Xs), size=len(idx), replace=False))
                     for _ in range(n_perm)])
    p = float((null <= obs).mean())
    return len(idx), obs, null.mean(), p

for task in ("ioi", "greaterthan"):
    n, obs, nullm, p = cohesion(task)
    print(f"\nTest C  {task}: n={n} circuits, mean pairwise distance "
          f"{obs:.2f} vs null {nullm:.2f}, p = {p:.4f} (10000 subsets, seed 42)")

# --- exact Ward-cut test (replaces the cohesion variant above for the
# clustermap claim: the claim is that the fixed 4-cluster cut isolates IOI) ---
def ward_cut_test():
    from scipy.cluster import hierarchy
    from math import comb
    Z = hierarchy.linkage(Xs, method="ward")
    lab4 = hierarchy.fcluster(Z, t=4, criterion="maxclust")
    sizes = [int((lab4 == c).sum()) for c in sorted(set(lab4))]
    ioi_idx = np.where(tasks == "ioi")[0]
    pure = len(set(lab4[ioi_idx])) == 1 and \
        sizes[sorted(set(lab4)).index(lab4[ioi_idx[0]])] == len(ioi_idx)
    n, k = len(tasks), len(ioi_idx)
    p_exact = sum(comb(s, k) for s in sizes if s >= k) / comb(n, k)
    print(f"\nWard 4-cut: sizes {sizes}, pure+complete IOI cluster: {pure}, "
          f"exact P(all {k} IOI in one cluster) = {p_exact:.3e}")
    return sizes, pure, p_exact

ward_cut_test()

# --- per-task dominant-role tally on the twelve-model k* pools (count-based,
# the Table cfs-summary definition: most frequent argmax pi_v among causal
# components), backing the per-task role-consistency claims in RQ8.1 ---
def dominant_role_tally():
    from collections import Counter
    res = {}
    for m, t, comps in records:
        roles = [v["dominant_role"] for v in comps.values()
                 if not v["causally_marginal"] and v.get("dominant_role")]
        if roles:
            top, n = Counter(roles).most_common(1)[0]
            res.setdefault(t, {})[m] = (top, n / len(roles))
    print("\nDominant-role tally (count-based, causal components):")
    for t in sorted(res):
        cnt = Counter(v[0] for v in res[t].values())
        print(f"  {t:12s} n={len(res[t]):2d}  "
              + ", ".join(f"{r}:{k}" for r, k in cnt.most_common()))
    return res

dominant_role_tally()

# --- circuit size as a fraction of model components (k* pools), backing the
# circuit-size caveat in the results: components per circuit / (L*(H+1)),
# with H inferred from the maximum head index present in the model's pool ---
def circuit_size_fractions():
    sizes, heads = {}, {}
    for m, t, comps in records:
        for name in comps:
            mm = re.match(r"a(\d+)\.h(\d+)", name)
            if mm:
                heads[m] = max(heads.get(m, 0), int(mm.group(2)) + 1)
        sizes.setdefault(m, []).append(len(comps))
    print("\nCircuit size fractions (components / model components):")
    for m in LAYERS:
        if m not in sizes:
            continue
        total = LAYERS[m] * (heads[m] + 1)
        med = float(np.median(sizes[m])) / total
        mx = max(sizes[m]) / total
        print(f"  {m:14s} med {med:.2f}  max {mx:.2f}  (model comps {total})")

circuit_size_fractions()

# --- per-model dominant role for each task (count-based, same definition as
# the tally above), backing the focused-pool role counts in the consolidated
# results: for each (model, task) the most frequent argmax pi_v among causal
# components is printed, so per-subset tallies can be read off directly ---
def dominant_role_by_model():
    from collections import Counter
    res = {}
    for m, t, comps in records:
        roles = [v["dominant_role"] for v in comps.values()
                 if not v["causally_marginal"] and v.get("dominant_role")]
        if roles:
            res.setdefault(t, {})[m] = Counter(roles).most_common(1)[0][0]
    print("\nDominant role per (task, model), count-based:")
    focused = ["gpt2large", "pythia410m", "pythia6.9b", "Qwen2.50.5B",
               "Qwen2.57B", "Llama3.21B", "Llama3.23B"]
    for t in sorted(res):
        row = ", ".join(f"{m}:{res[t][m]}" for m in LAYERS if m in res[t])
        print(f"  {t:12s} {row}")
    print("\nFocused seven-model pool tallies (count-based):")
    for t in sorted(res):
        cnt = Counter(res[t][m] for m in focused if m in res[t])
        n = sum(cnt.values())
        print(f"  {t:12s} n={n}  "
              + ", ".join(f"{r}:{k}" for r, k in cnt.most_common()))

dominant_role_by_model()
