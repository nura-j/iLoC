"""
pipeline.py
===========
High-level pipeline functions for the COMPASS framework.

Called by run_experiment.py — keeps the experiment script slim while
the heavy logic lives inside the package.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Graph reconstruction
# ---------------------------------------------------------------------------

def reconstruct_graph(model, json_path: str):
    """
    Reconstruct an EAP Graph from a results JSON + the model.

    Reads the circuit nodes/edges from the stored layer1.circuit,
    builds a fresh Graph from the model, then marks the stored
    nodes/edges as in_graph with their scores.
    """
    from .core.eap_graph import Graph

    with open(json_path) as f:
        rec = json.load(f)

    circuit = rec.get("layer1", {}).get("circuit", {})
    stored_nodes = {n["name"] for n in circuit.get("nodes", [])}
    stored_edges = {e["name"]: e.get("score", 0.0)
                    for e in circuit.get("edges", [])}

    graph = Graph.from_model(model)
    graph.reset(empty=True)

    for name in stored_nodes:
        if name in graph.nodes:
            graph.nodes[name].in_graph = True

    for edge_name, score in stored_edges.items():
        if edge_name in graph.edges:
            graph.edges[edge_name].in_graph = True
            graph.edges[edge_name].score = score

    return graph


# ---------------------------------------------------------------------------
# CFS pipeline (Layer 2)
# ---------------------------------------------------------------------------

def run_cfs_pipeline(
    model,
    tokenizer,
    tasks: Dict[str, Tuple],   # task_name -> (graph, owl_path, json_path, csv_path)
    cfs_dir: str,
    figs_dir: str,
    max_examples: int = 50,
    task_defs: Optional[List[Dict]] = None,  # task definitions with task_family
) -> Tuple[Dict, Dict]:
    """
    Run CFS computation + figures for all tasks.

    Computes two attention profiles per circuit:
      1. Linguistic (shared across tasks): subject, object, verb, function, entity, other
      2. Task-specific: role_filler/scaffold (roles), repeated_name (IOI), first_number (GT)

    The linguistic profile enables cross-task comparison (RQ2).
    The task-specific profile enables within-task characterisation (RQ1).

    Returns (cfs_all, ll_all) dicts keyed by task name.
    """
    import pandas as pd
    from .cfs import (
        compute_cfs, compute_logit_lens,
        save_cfs_json, save_cfs_prolog, save_logit_lens_json,
        add_cfs_to_results_json, add_cfs_to_ttl,
        plot_dla, plot_attention_profiles, plot_cfs_summary,
        plot_dla_comparison, plot_logit_lens, plot_logit_lens_overlay,
    )
    from .cfs.role_labellers import get_labellers

    # Build task_family lookup from task_defs
    _task_families = {}
    if task_defs:
        for td in task_defs:
            _task_families[td["task"]] = td.get("task_family", "")

    os.makedirs(cfs_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    cfs_all = {}
    ll_all = {}

    for task_name, (graph, owl, jpath, csv_path) in tasks.items():
        task_lower = task_name.lower()
        task_family = _task_families.get(task_name, "")
        print(f"\n  {task_name} — CFS Computation (family={task_family})")

        # Load examples with full metadata
        df = pd.read_csv(csv_path)
        examples = []
        examples_meta = []
        for _, row in df.head(max_examples).iterrows():
            examples.append({
                "clean": str(row.get("clean", row.get("prompt", ""))),
                "correct": str(row.get("target_word", row.get("answer", ""))),
            })
            examples_meta.append(row.to_dict())
        print(f"    Examples loaded: {len(examples)}")

        # Get labellers for this task family
        ling_roles, task_roles, ling_labeller, task_labeller = get_labellers(task_family)

        # 1. Linguistic CFS (shared roles for cross-task comparison)
        def _ling_lab(text, token_strs):
            return ling_labeller(text, token_strs, None)

        cfs = compute_cfs(model, graph, examples, tokenizer,
                         epsilon_dla=0.05,
                         token_roles=ling_roles,
                         role_labeller=_ling_lab)
        cfs_all[task_name] = cfs

        causal = cfs.causal_components()
        print(f"    CFS components : {len(cfs.components)} total, {len(causal)} causal")
        for comp in causal[:5]:
            dom = comp.dominant_role() or "—"
            print(f"      {comp.name}: δ={comp.dla_score:.4f}, dominant={dom}")

        # 2. Task-specific CFS (if labeller available)
        if task_labeller and task_roles:
            from .cfs.attention_profile import compute_attention_profiles
            from .cfs.dla import compute_dla

            dla = cfs.dla_result
            _meta_iter = iter(examples_meta)

            def _task_lab(text, token_strs):
                try:
                    meta = next(_meta_iter)
                except StopIteration:
                    meta = None
                return task_labeller(text, token_strs, meta)

            task_attn = compute_attention_profiles(
                model, graph, examples, tokenizer,
                dla_scores=dla.scores if dla else None,
                epsilon_dla=0.05,
                token_roles=task_roles,
                role_labeller=_task_lab,
            )

            # Save task-specific profiles alongside linguistic ones
            task_profiles = task_attn.profiles
            task_cfs_data = {
                "task_family": task_family,
                "task_roles": list(task_roles),
                "profiles": {
                    name: {r: round(v, 4) for r, v in prof.items()}
                    for name, prof in task_profiles.items()
                },
            }
            _task_path = os.path.join(cfs_dir, f"{task_lower}_task_profiles.json")
            with open(_task_path, "w") as f:
                json.dump(task_cfs_data, f, indent=2)
            print(f"    Task-specific profiles: {_task_path}")

            # Print task-specific dominant roles
            for comp in causal[:3]:
                tprof = task_profiles.get(comp.name, {})
                if tprof:
                    tdom = max(tprof, key=tprof.get, default="?")
                    print(f"      {comp.name}: task_dom={tdom} ({tprof.get(tdom,0):.2f})")

        # Save
        save_cfs_json(cfs, os.path.join(cfs_dir, f"{task_lower}_cfs.json"))
        save_cfs_prolog(cfs, os.path.join(cfs_dir, f"{task_lower}_cfs.pl"))
        add_cfs_to_results_json(jpath, cfs)
        if owl and os.path.isfile(owl):
            add_cfs_to_ttl(owl, cfs)

        # Figures
        plot_dla(cfs, task_name, os.path.join(figs_dir, f"{task_lower}_dla.png"))
        plot_attention_profiles(cfs, task_name, os.path.join(figs_dir, f"{task_lower}_attention.png"))
        plot_cfs_summary(cfs, task_name, os.path.join(figs_dir, f"{task_lower}_cfs_summary.png"))

        # Logit Lens
        print(f"\n  {task_name} — Logit Lens")
        ll = compute_logit_lens(model, examples, tokenizer)
        ll_all[task_name] = ll
        print(f"    Layers: {ll.n_layers}, crossing: L{ll.crossing_layer()}")
        save_logit_lens_json(ll, os.path.join(cfs_dir, f"{task_lower}_logit_lens.json"))
        plot_logit_lens(ll, task_name, os.path.join(figs_dir, f"{task_lower}_logit_lens.png"))

    # Cross-circuit comparisons
    if len(cfs_all) >= 2:
        plot_dla_comparison(cfs_all, os.path.join(figs_dir, "dla_comparison.png"))
        plot_logit_lens_overlay(ll_all, os.path.join(figs_dir, "logit_lens_overlay.png"))

    print(f"\n  CFS figures -> {figs_dir}/")
    return cfs_all, ll_all


# ---------------------------------------------------------------------------
# ILP_arch pipeline (Layer 3)
# ---------------------------------------------------------------------------

def run_ilp_arch_pipeline(
    json_paths: Dict[str, str],   # task_name -> json_path
    ilp_dir: str,
) -> Dict[str, Dict]:
    """
    Run ILP_arch for all tasks (one-vs-rest) and write layer3 into JSONs.

    Returns arch_results dict keyed by task_name (lowercase).
    """
    from .storage.ilp.ilp_arch import ArchitecturalILPLearner
    from .storage.results_export import add_ilp_architectural

    os.makedirs(ilp_dir, exist_ok=True)

    # Load records
    task_records = {}
    for task_name, jp in json_paths.items():
        with open(jp) as f:
            task_records[task_name.lower()] = (json.load(f), jp)

    learner = ArchitecturalILPLearner()
    arch_results = {}

    for target_task in list(task_records.keys()):
        target_rec, target_json = task_records[target_task]
        pos_records = [target_rec]
        neg_records = [rec for t, (rec, _) in task_records.items() if t != target_task]

        if not neg_records:
            print(f"  {target_task}: need at least one negative, skipping")
            continue

        result = learner.learn_from_json_records(
            target_type=target_task,
            positive_records=pos_records,
            negative_records=neg_records,
        )
        arch_results[target_task] = result

        feats = result["features"][0]
        clause = result["clause_text"]
        conf = result["confidence"]

        add_ilp_architectural(
            json_path=target_json,
            depth_mean=feats["depth_mean"],
            depth_min=feats["depth_min"],
            depth_max=feats["depth_max"],
            component_ratio=feats["component_ratio_str"],
            num_attn_heads=feats["num_attn"],
            num_mlp_blocks=feats["num_mlp"],
            motifs=result["motifs"],
            tau_arch_text=clause,
            tau_arch_confidence=conf,
            ilp_signature=result["ilp_signature"],
            exact=feats.get("exact", {}),
            loo_f1=result.get("loo_f1"),
        )
        print(f"  {target_task}: layer3 written — {clause[:60]}... (conf={conf:.3f})")

        # Save B_arch Prolog file
        pl_path = os.path.join(ilp_dir, f"{target_task}_arch_background.pl")
        with open(pl_path, "w") as f:
            f.write(f"% ILP_arch background knowledge for {target_task} circuit\n")
            f.write(f"% Learned clause:\n%   {clause}\n")
            f.write(f"% Confidence: {conf:.3f}\n\n")
            f.write(result["background_facts"])
            f.write("\n\n% Positive examples:\n")
            f.write("\n".join(result["positive_examples"]))
            f.write("\n\n% Negative examples:\n")
            f.write("\n".join(result["negative_examples"]))
        print(f"  B_arch: {pl_path}")

    return arch_results


# ---------------------------------------------------------------------------
# Distance report
# ---------------------------------------------------------------------------

def export_distance_report(
    arch_results: Dict[str, Dict],
    out_path: str,
    sem_results: Optional[Dict[str, Dict]] = None,
) -> Optional[str]:
    """
    Compute pairwise ILP signature distances and save distance_report.json.

    arch_results holds per-task ILP_arch results (with 'ilp_signature').
    sem_results, when given, holds per-task ILP_sem results; the semantic
    matrix is only computed from those — it is never a copy of the
    architectural one.

    Returns path to the report, or None if fewer than 2 tasks.
    """
    from .storage.ilp.ilp_signature_distance import (
        ilp_signature_distance, signature_distance_breakdown,
    )

    if len(arch_results) < 2:
        return None

    tasks = sorted(arch_results.keys())

    def _sig(task):
        return arch_results[task].get("ilp_signature", {})

    def _sem_sig(task):
        return (sem_results or {}).get(task, {}).get("ilp_signature", {})

    have_sem = bool(sem_results) and all(_sem_sig(t) for t in tasks)

    arch_matrix = {}
    sem_matrix = {} if have_sem else None
    for ta in tasks:
        arch_matrix[ta] = {}
        if have_sem:
            sem_matrix[ta] = {}
        for tb in tasks:
            if ta == tb:
                arch_matrix[ta][tb] = 0.0
                if have_sem:
                    sem_matrix[ta][tb] = 0.0
            else:
                arch_matrix[ta][tb] = round(
                    ilp_signature_distance(_sig(ta), _sig(tb)), 4)
                if have_sem:
                    sem_matrix[ta][tb] = round(
                        ilp_signature_distance(_sem_sig(ta), _sem_sig(tb)), 4)

    def _pair_entry(ta, tb):
        entry = {
            "d_arch": arch_matrix[ta][tb],
            "breakdown_arch": signature_distance_breakdown(_sig(ta), _sig(tb)),
        }
        if have_sem:
            entry["d_sem"] = sem_matrix[ta][tb]
        return entry

    report = {
        "tasks": tasks,
        "delta_arch": 0.40,
        "transfer_pairs": [],
        "architectural_matrix": arch_matrix,
        "per_pair": {
            f"{ta}_vs_{tb}": _pair_entry(ta, tb)
            for i, ta in enumerate(tasks) for tb in tasks[i+1:]
        },
    }
    if have_sem:
        report["delta_sem"] = 0.30
        report["semantic_matrix"] = sem_matrix

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Distance report: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# ΔT preservation analysis
# ---------------------------------------------------------------------------

def compute_delta_t_preservation(
    registry: List[Dict[str, Any]],
    task_names: List[str],
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    For each task that exists in 2+ models, compute |ΔT_src - ΔT_tgt|.

    Uses stored DeltaT from results JSON — no model inference needed.

    Returns report dict with pairs and summary.
    """
    dt_results = []

    for task in task_names:
        tn = task.lower()
        task_recs = {}
        for rec in registry:
            rec_task = (rec.get("meta", {}).get("task", "") or "").lower()
            rec_model = rec.get("meta", {}).get("model_name", "")
            if rec_task == tn:
                task_recs[rec_model] = rec

        if len(task_recs) < 2:
            continue

        models = sorted(task_recs.keys())
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                dt1 = (task_recs[m1].get("layer1", {})
                       .get("metrics", {}).get("DeltaT"))
                dt2 = (task_recs[m2].get("layer1", {})
                       .get("metrics", {}).get("DeltaT"))
                if dt1 is not None and dt2 is not None:
                    diff = abs(dt1 - dt2)
                    dt_results.append((task, m1, m2, dt1, dt2, diff))
                    print(f"  {task:12} {m1} -> {m2}:  "
                          f"ΔT_src={dt1:.4f}  ΔT_tgt={dt2:.4f}  "
                          f"|diff|={diff:.4f}")

    report = {
        "pairs": [
            {"task": t, "model_src": ms, "model_tgt": mt,
             "delta_t_src": ds, "delta_t_tgt": dt, "abs_diff": diff}
            for t, ms, mt, ds, dt, diff in dt_results
        ],
        "summary": {
            "n_pairs": len(dt_results),
            "mean_abs_diff": (sum(d[5] for d in dt_results) / len(dt_results)
                              if dt_results else 0.0),
            "max_abs_diff": max((d[5] for d in dt_results), default=0.0),
            "tasks_within_003": sum(1 for d in dt_results if d[5] <= 0.03),
        }
    }

    if out_path and dt_results:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        s = report["summary"]
        print(f"\n  Mean |ΔT_src - ΔT_tgt| = {s['mean_abs_diff']:.4f}")
        print(f"  Max  |ΔT_src - ΔT_tgt| = {s['max_abs_diff']:.4f}")
        print(f"  Pairs within 0.03: {s['tasks_within_003']}/{s['n_pairs']}")
        print(f"  Saved: {out_path}")

    return report


# ---------------------------------------------------------------------------
# Dual-pool analysis
# ---------------------------------------------------------------------------

def analyse_dual_pool(
    json_paths: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Determine which hypothesis pool (A=named motif, B=blind structural)
    produced each learned τ_arch clause, and whether Pool B found novel patterns.

    Reads the learned clauses from stored results JSON files.

    Returns list of dicts, one per task.
    """
    import re

    results = []
    for task_name, jp in json_paths.items():
        with open(jp) as f:
            rec = json.load(f)
        l3 = rec.get("layer3", {})
        clause = l3.get("tau_arch_text", "")
        if not clause:
            continue

        has_motif = "has_motif" in clause
        has_blind_pred = any(p in clause for p in ["edge(", "type(", "layer("])

        if has_motif:
            pool = "A"
            pool_desc = "named motif"
            novel = False
        elif has_blind_pred:
            pool = "B"
            pool_desc = "blind structural"
            novel = True
        else:
            pool = "B"
            pool_desc = "blind (aggregate only)"
            novel = False  # uses only component_ratio/size/layer_span — no novel pattern

        body = clause.split(":-")[1] if ":-" in clause else ""
        preds = sorted(set(re.findall(r"(\w+)\(", body)))
        n_lits = l3.get("ilp_signature", {}).get("avg_length", 0)

        results.append({
            "task": task_name,
            "pool": pool,
            "pool_desc": pool_desc,
            "novel": novel,
            "predicates": preds,
            "n_predicates": len(preds),
            "n_literals": n_lits,
            "confidence": l3.get("tau_arch_confidence"),
            "complexity": l3.get("ilp_signature", {}).get("complexity"),
            "clause": clause,
        })

    return results


# ---------------------------------------------------------------------------
# Subsumption matrix
# ---------------------------------------------------------------------------

def compute_subsumption_matrix(
    json_paths: Dict[str, str],
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute pairwise θ-subsumption for all learned τ_arch clauses.

    Reports both strict and relaxed subsumption:
    - Strict: exact constant matching (R > 0.63 ≠ R > 0.38)
    - Relaxed: numeric thresholds treated as wildcards (R > _ matches R > _)

    Also computes predicate-structure overlap (Jaccard over predicate names).

    Returns report with all three matrices and summary stats.
    """
    from .storage.ilp.subsumption import (
        subsumption_matrix, predicate_structure_overlap,
    )

    clauses = {}
    for task_name, jp in json_paths.items():
        with open(jp) as f:
            rec = json.load(f)
        clause = rec.get("layer3", {}).get("tau_arch_text", "")
        if clause:
            clauses[task_name.lower()] = clause

    if len(clauses) < 2:
        return {"matrix_strict": {}, "matrix_relaxed": {}, "summary": {}}

    matrix_strict = subsumption_matrix(clauses, relaxed=False)
    matrix_relaxed = subsumption_matrix(clauses, relaxed=True)
    pred_overlap = predicate_structure_overlap(clauses)
    tasks = sorted(clauses.keys())

    def _count_rels(matrix):
        rels = {"equivalent": 0, "generalises": 0, "specialises": 0, "incomparable": 0}
        for t1 in tasks:
            for t2 in tasks:
                if t1 < t2:
                    rels[matrix[t1][t2]] += 1
        return rels

    strict_rels = _count_rels(matrix_strict)
    relaxed_rels = _count_rels(matrix_relaxed)

    import re
    pred_sets = {}
    for t, c in clauses.items():
        body = c.split(":-")[1] if ":-" in c else ""
        pred_sets[t] = sorted(set(re.findall(r"(\w+)\(", body)))

    report = {
        "tasks": tasks,
        "clauses": clauses,
        "matrix_strict": matrix_strict,
        "matrix_relaxed": matrix_relaxed,
        "predicate_overlap": pred_overlap,
        "predicate_sets": pred_sets,
        "summary_strict": strict_rels,
        "summary_relaxed": relaxed_rels,
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Subsumption report saved: {out_path}")

    sym = {"self": "=", "equivalent": "≡", "generalises": "⊒",
           "specialises": "⊑", "incomparable": "∥"}

    for label, matrix, rels in [
        ("Strict", matrix_strict, strict_rels),
        ("Relaxed (numeric wildcards)", matrix_relaxed, relaxed_rels),
    ]:
        print(f"\n  θ-Subsumption — {label} ({len(tasks)} clauses):")
        print(f"    {'':>12}", end="")
        for t in tasks:
            print(f"{t:>12}", end="")
        print()
        for t1 in tasks:
            print(f"    {t1:>12}", end="")
            for t2 in tasks:
                print(f"{sym.get(matrix[t1][t2], '?'):>12}", end="")
            print()
        print(f"    {rels['equivalent']} equiv, {rels['generalises']} gen, "
              f"{rels['specialises']} spec, {rels['incomparable']} incomp")

    print(f"\n  Predicate-structure overlap (Jaccard over predicate names):")
    print(f"    {'':>12}", end="")
    for t in tasks:
        print(f"{t:>12}", end="")
    print()
    for t1 in tasks:
        print(f"    {t1:>12}", end="")
        for t2 in tasks:
            print(f"{pred_overlap[t1][t2]:12.2f}", end="")
        print()

    return report


# ---------------------------------------------------------------------------
# CFS clustering test
# ---------------------------------------------------------------------------

def cfs_clustering_test(
    cfs_dir: str,
    results_dir: str,
    task_families: Dict[str, str],
    out_path: Optional[str] = None,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Test whether CFS profiles cluster by task family.

    Builds a feature vector per circuit from CFS (DLA distribution,
    dominant role distribution) and structural features (edge type
    ratios, attention ratio), standardises, computes pairwise Euclidean
    distances, and runs a permutation test comparing within-family vs
    cross-family distances.

    Returns report dict with distance matrix, within/cross stats, p-value.
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    def _load_features(task):
        cfs_path = os.path.join(cfs_dir, f"{task}_cfs.json")
        res_path = os.path.join(results_dir, f"{task}_results.json")
        if not os.path.isfile(cfs_path) or not os.path.isfile(res_path):
            return None

        with open(cfs_path) as f:
            cfs = json.load(f)
        with open(res_path) as f:
            res = json.load(f)

        comps = cfs.get("components", {})
        causal = {k: v for k, v in comps.items()
                  if not v.get("causally_marginal", True)}
        dlas = [v["dla_score"] for v in causal.values()]
        n_causal = len(causal)

        mean_dla = float(np.mean(dlas)) if dlas else 0.0
        std_dla = float(np.std(dlas)) if len(dlas) > 1 else 0.0
        pos_pct = sum(1 for d in dlas if d > 0) / n_causal if n_causal else 0.0
        neg_pct = sum(1 for d in dlas if d < 0) / n_causal if n_causal else 0.0
        max_dla = max(dlas) if dlas else 0.0
        min_dla = min(dlas) if dlas else 0.0

        dom_counts = {r: 0 for r in ["subject", "object", "verb", "entity", "other"]}
        for v in causal.values():
            prof = v.get("attention_profile", {})
            dom = max(prof, key=prof.get, default="other") if prof else "other"
            if dom in dom_counts:
                dom_counts[dom] += 1
        dom_vec = [dom_counts[r] / n_causal if n_causal else 0.0
                   for r in ["subject", "object", "verb", "entity", "other"]]

        nodes = res.get("layer1", {}).get("circuit", {}).get("nodes", [])
        edges = res.get("layer1", {}).get("circuit", {}).get("edges", [])
        n_attn = len([n for n in nodes if n.get("type") == "attn"])
        attn_ratio = n_attn / len(nodes) if nodes else 0.0
        pct_marginal = (len(comps) - n_causal) / len(comps) if comps else 0.0

        et = {"aa": 0, "am": 0, "ma": 0, "mm": 0}
        for e in edges:
            src, tgt = e.get("source", ""), e.get("target", "")
            if tgt == "logits":
                continue
            st = "a" if (src.startswith("a") and ".h" in src) else "m"
            tt = "a" if (tgt.startswith("a") and ".h" in tgt) else "m"
            et[st + tt] += 1
        total_e = sum(et.values()) or 1
        edge_vec = [et[k] / total_e for k in ["aa", "am", "ma", "mm"]]

        faith = res.get("layer1", {}).get("metrics", {}).get("faithfulness", 0)
        delta_t = res.get("layer1", {}).get("metrics", {}).get("DeltaT", 0)

        return np.array([
            mean_dla, std_dla, pos_pct, neg_pct, max_dla, min_dla,
            n_causal, pct_marginal, attn_ratio,
        ] + dom_vec + edge_vec + [faith, delta_t])

    tasks, vectors, families = [], [], []
    for task, family in sorted(task_families.items()):
        vec = _load_features(task)
        if vec is not None:
            tasks.append(task)
            vectors.append(vec)
            families.append(family)

    if len(tasks) < 3:
        return {"error": "need at least 3 tasks"}

    V = np.array(vectors)
    mu, sigma = V.mean(axis=0), V.std(axis=0)
    sigma[sigma == 0] = 1
    V_norm = (V - mu) / sigma

    n = len(tasks)
    dist_matrix = squareform(pdist(V_norm, "euclidean"))

    within, cross = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i, j]
            if families[i] == families[j]:
                within.append(d)
            else:
                cross.append(d)

    within_mean = float(np.mean(within)) if within else 0.0
    cross_mean = float(np.mean(cross)) if cross else 0.0

    p_value = None
    if within and cross:
        observed = cross_mean - within_mean
        all_d = within + cross
        all_lab = ["w"] * len(within) + ["c"] * len(cross)
        rng = np.random.RandomState(seed)
        count = 0
        for _ in range(n_permutations):
            perm = rng.permutation(all_lab)
            pw = [all_d[k] for k in range(len(all_d)) if perm[k] == "w"]
            pc = [all_d[k] for k in range(len(all_d)) if perm[k] == "c"]
            if pw and pc and (np.mean(pc) - np.mean(pw)) >= observed:
                count += 1
        p_value = count / n_permutations

    report = {
        "tasks": tasks,
        "families": families,
        "distance_matrix": {
            tasks[i]: {tasks[j]: round(dist_matrix[i, j], 4)
                       for j in range(n)}
            for i in range(n)
        },
        "within_family_mean": round(within_mean, 4),
        "cross_family_mean": round(cross_mean, 4),
        "gap": round(cross_mean - within_mean, 4),
        "p_value": p_value,
        "n_permutations": n_permutations,
        "significant_at_005": p_value < 0.05 if p_value is not None else None,
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

    print(f"\n  CFS clustering test ({len(tasks)} circuits):")
    print(f"    Within-family mean distance: {within_mean:.3f}")
    print(f"    Cross-family mean distance:  {cross_mean:.3f}")
    print(f"    Gap (cross - within):        {cross_mean - within_mean:.3f}")
    if p_value is not None:
        sig = "significant" if p_value < 0.05 else "not significant"
        print(f"    Permutation p-value:         {p_value:.4f} ({sig} at α=0.05)")

    return report
