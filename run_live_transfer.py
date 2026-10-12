"""
run_live_transfer.py
=====================
Live transfer evaluation: take a circuit discovered on model A,
use its τ_arch to find a candidate in model B, then evaluate that
candidate on model B's evaluation suite.

This closes the RQ3 story: we don't just compare stored ΔT values
from independently discovered circuits — we actually transfer a
structural signature and verify it works on the target model.

Usage:
    python run_live_transfer.py --source pythia14m --target pythia1b
    python run_live_transfer.py --source pythia1b --target Llama3.21B
    python run_live_transfer.py --all  # all source→target pairs

Output:
    analysis/transfer/live_transfer_results.json
"""

import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser(description="Live transfer evaluation")
parser.add_argument("--source", default=None, help="Source model short name")
parser.add_argument("--target", default=None, help="Target model short name")
parser.add_argument("--all", action="store_true", help="Run all source→target pairs")
parser.add_argument("--tasks", nargs="*", default=None, help="Subset of tasks (default: all available)")
args = parser.parse_args()

# ── Model registry ────────────────────────────────────────────────────────

HF_MODELS = {
    "pythia14m": "EleutherAI/pythia-14m",
    "pythia1b": "EleutherAI/pythia-1b",
    "Llama3.21B": "meta-llama/Llama-3.2-1B",
}

DATASET_REGISTRY = {
    "pythia1b": {
        "IOI": "./data/ioi_pythia_1b_filtered.csv",
        "Location": "./semantic_role_generation/semroles_v1_filtered_1b/Location/both_correct/prompts.csv",
        "Time": "./semantic_role_generation/semroles_v1_filtered_1b/Time/both_correct/prompts.csv",
        "Path": "./semantic_role_generation/semroles_v1_filtered_1b/Path/both_correct/prompts.csv",
        "GreaterThan": "./data/greater_than_filtered_pythia1b/both_correct/prompts.csv",
    },
    "pythia14m": {
        "IOI": "./data/ioi_pythia_14m_filtered.csv",
        "Location": "./semantic_role_generation/semroles_v1_filtered_14m/Location/both_correct/prompts.csv",
        "Time": "./semantic_role_generation/semroles_v1_filtered_14m/Time/both_correct/prompts.csv",
        "Path": "./semantic_role_generation/semroles_v1_filtered_14m/Path/both_correct/prompts.csv",
        "GreaterThan": "./data/greater_than_filtered_pythia14m/both_correct/prompts.csv",
    },
    "Llama3.21B": {
        "IOI": "./data/ioi_llama_1b.csv",
        "Location": "./semantic_role_generation/semroles_v1_Llama_3_2_1B_filtered/Location/both_correct/prompts.csv",
        "Time": "./semantic_role_generation/semroles_v1_Llama_3_2_1B_filtered/Time/both_correct/prompts.csv",
        "Path": "./semantic_role_generation/semroles_v1_Llama_3_2_1B_filtered/Path/both_correct/prompts.csv",
        "GreaterThan": "./data/greater_than_filtered_llama1b/both_correct/prompts.csv",
    },
}

ALL_TASKS = ["IOI", "Location", "Time", "Path", "GreaterThan"]
TASKS = args.tasks or ALL_TASKS

OUTPUT_DIR = "./analysis/transfer"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Determine which pairs to run ──────────────────────────────────────────

if args.all:
    PAIRS = []
    for src in HF_MODELS:
        for tgt in HF_MODELS:
            if src != tgt:
                PAIRS.append((src, tgt))
elif args.source and args.target:
    PAIRS = [(args.source, args.target)]
else:
    print("Usage: --source X --target Y, or --all")
    sys.exit(1)

print(f"Live Transfer Evaluation")
print(f"  Pairs: {[(s,t) for s,t in PAIRS]}")
print(f"  Tasks: {TASKS}")
print(f"  Output: {OUTPUT_DIR}/")

# ── Helpers ───────────────────────────────────────────────────────────────

def load_source_circuit(model_short, task):
    """Load source circuit's results JSON."""
    jp = f"analysis/{model_short}/results/{task.lower()}_results.json"
    if not os.path.isfile(jp):
        return None, None
    with open(jp) as f:
        rec = json.load(f)
    return rec, jp


def get_dataset_path(model_short, task):
    """Get the evaluation dataset CSV for a model+task."""
    datasets = DATASET_REGISTRY.get(model_short, {})
    path = datasets.get(task, "")
    return path if path and os.path.isfile(path) else None


# ── Main loop ─────────────────────────────────────────────────────────────

all_results = {}

for src_model, tgt_model in PAIRS:
    pair_key = f"{src_model}_to_{tgt_model}"
    print(f"\n{'='*68}")
    print(f"  {src_model} → {tgt_model}")
    print(f"{'='*68}")

    # Load target model
    hf_id = HF_MODELS[tgt_model]
    print(f"\n  Loading target model: {hf_id}...")

    from transformers import AutoTokenizer
    import transformer_lens as tl
    import torch

    tgt_model_obj = tl.HookedTransformer.from_pretrained(hf_id, device='cpu')
    tgt_tokenizer = AutoTokenizer.from_pretrained(hf_id)
    tgt_model_obj.cfg.use_attn_result = True
    tgt_model_obj.cfg.use_split_qkv_input = True
    tgt_model_obj.cfg.use_hook_mlp_in = True
    if hasattr(tgt_model_obj.cfg, 'ungroup_grouped_query_attention'):
        tgt_model_obj.cfg.ungroup_grouped_query_attention = True
    print(f"  Loaded ({tgt_model_obj.cfg.n_layers}L, {tgt_model_obj.cfg.n_heads}H)")

    # Set up transfer engine with ALL models in registry
    from trace.mechint.transfer import TransferEngine

    engine = TransferEngine(
        registry_dir="./analysis",
        delta_ilp=0.30, delta_arch=0.40,
        theta_behav=0.50, theta_causal=0.10,
        struct_depth_tol=0.15, struct_size_tol=0.15,
    )
    n_reg = engine.reload_registry()
    print(f"  Registry: {n_reg} circuits loaded")

    pair_results = {}

    for task in TASKS:
        print(f"\n  --- {task}: {src_model} → {tgt_model} ---")

        # 1. Load source circuit
        src_rec, src_jp = load_source_circuit(src_model, task)
        if src_rec is None:
            print(f"    SKIP: no source circuit for {task}@{src_model}")
            continue

        src_metrics = src_rec.get("layer1", {}).get("metrics", {})
        src_delta_t = src_metrics.get("DeltaT", 0)
        src_acc_t = src_metrics.get("AccT", 0)
        print(f"    Source: AccT={src_acc_t:.4f}, ΔT={src_delta_t:.4f}")

        # 2. Get target dataset
        tgt_dataset = get_dataset_path(tgt_model, task)
        if not tgt_dataset:
            print(f"    SKIP: no dataset for {task}@{tgt_model}")
            continue

        # 3. Run static transfer to find candidates
        report = engine.find_transfer_candidates(
            source_json_path=src_jp,
            target_model_name=tgt_model,
        )

        struct_passed = [c for c in report.candidates if c.struct_pass]
        def5_passed = [c for c in report.candidates if c.def5_pass]
        print(f"    Static: {len(struct_passed)} struct pass, {len(def5_passed)} def5 pass")

        if not struct_passed:
            print(f"    No structural candidates — transfer fails (expected for cross-family)")
            pair_results[task] = {
                "status": "no_candidates",
                "src_delta_t": round(src_delta_t, 4),
                "src_acc_t": round(src_acc_t, 4),
                "n_candidates": 0,
            }
            continue

        # 4. For each candidate that passed structural filter,
        #    reconstruct its circuit in the target model and evaluate
        from trace.mechint.pipeline import reconstruct_graph
        from trace.mechint.core.eap_evaluate import evaluate_graph, evaluate_baseline
        from trace.mechint.core.eap_graph import Graph
        from trace.mechint.core.eap_cnp import cnp_metric
        from functools import partial

        metric_fn = partial(cnp_metric, loss=False, mean=False)

        # Build dataloader from target dataset
        try:
            from trace.mechint.core.analyzer import MechIntAnalyzer
            analyzer = MechIntAnalyzer(cnp_dataset_path=tgt_dataset)
            ds = analyzer.load_cnp_dataset(tgt_dataset, tgt_tokenizer)
            dataloader = ds.to_dataloader(10)
        except Exception as e:
            print(f"    ERROR building dataloader: {e}")
            continue

        # Baseline: full target model performance
        baseline_perf = evaluate_baseline(
            tgt_model_obj, dataloader, metric_fn
        ).mean().item()

        # Ablated: empty circuit
        empty = Graph.from_model(tgt_model_obj)
        empty.reset(empty=True)
        ablated_perf = evaluate_graph(
            tgt_model_obj, empty, dataloader, metric_fn, skip_clean=False
        ).mean().item()

        full_delta_t = baseline_perf - ablated_perf
        print(f"    Target baseline={baseline_perf:.4f}, ablated={ablated_perf:.4f}, "
              f"full ΔT={full_delta_t:.4f}")

        # 5. Evaluate each candidate
        candidate_results = []
        best_candidate = None
        best_delta_t = -float('inf')

        for ci, cand in enumerate(struct_passed):
            cand_jp = cand.json_path
            if not cand_jp or not os.path.isfile(cand_jp):
                continue

            cand_task = cand.task_name or "unknown"
            cand_model = cand.model_name or "unknown"

            # Reconstruct candidate circuit in TARGET model
            # The candidate's nodes/edges are from the target model's registry
            try:
                cand_graph = reconstruct_graph(tgt_model_obj, cand_jp)
            except Exception as e:
                print(f"    Candidate {ci} ({cand_task}@{cand_model}): graph failed: {e}")
                continue

            # Evaluate candidate circuit on target dataset
            try:
                cand_perf = evaluate_graph(
                    tgt_model_obj, cand_graph, dataloader, metric_fn
                ).mean().item()
                cand_delta_t = cand_perf - ablated_perf

                d_ilp = cand.def5_distance or 0
                passed_behav = cand_perf >= 0.50  # theta_behav
                passed_causal = cand_delta_t >= 0.10  # theta_causal

                print(f"    Candidate {ci} ({cand_task}@{cand_model}): "
                      f"AccT={cand_perf:.4f}, ΔT={cand_delta_t:.4f}, "
                      f"d_ILP={d_ilp:.3f}, "
                      f"behav={'PASS' if passed_behav else 'fail'}, "
                      f"causal={'PASS' if passed_causal else 'fail'}")

                cr = {
                    "candidate_task": cand_task,
                    "candidate_model": cand_model,
                    "candidate_json": cand_jp,
                    "acc_t": round(cand_perf, 4),
                    "delta_t": round(cand_delta_t, 4),
                    "d_ilp": round(d_ilp, 4),
                    "passed_behav": passed_behav,
                    "passed_causal": passed_causal,
                    "accepted": passed_behav and passed_causal,
                }
                candidate_results.append(cr)

                if passed_behav and passed_causal and cand_delta_t > best_delta_t:
                    best_delta_t = cand_delta_t
                    best_candidate = cr

            except Exception as e:
                print(f"    Candidate {ci} ({cand_task}@{cand_model}): eval failed: {e}")

        # 6. Report
        delta_t_gap = abs(src_delta_t - best_delta_t) if best_candidate else None

        pair_results[task] = {
            "status": "accepted" if best_candidate else "rejected",
            "src_model": src_model,
            "tgt_model": tgt_model,
            "src_delta_t": round(src_delta_t, 4),
            "src_acc_t": round(src_acc_t, 4),
            "tgt_baseline": round(baseline_perf, 4),
            "tgt_ablated": round(ablated_perf, 4),
            "n_candidates": len(struct_passed),
            "n_accepted": sum(1 for c in candidate_results if c["accepted"]),
            "candidates": candidate_results,
            "best_candidate": best_candidate,
            "delta_t_gap": round(delta_t_gap, 4) if delta_t_gap is not None else None,
        }

        if best_candidate:
            print(f"\n    ACCEPTED: {best_candidate['candidate_task']}@{best_candidate['candidate_model']}")
            print(f"      ΔT_src={src_delta_t:.4f}, ΔT_transferred={best_delta_t:.4f}, "
                  f"|gap|={delta_t_gap:.4f}")
        else:
            print(f"\n    REJECTED: no candidate passed both behavioural and causal validation")

    all_results[pair_key] = pair_results

    # Save per-pair results file
    pair_out = os.path.join(OUTPUT_DIR, f"live_transfer_results_{src_model}_{tgt_model}.json")
    with open(pair_out, "w") as f:
        json.dump({pair_key: pair_results}, f, indent=2)
    print(f"\n  Saved: {pair_out}")

    # Free target model
    del tgt_model_obj, tgt_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── Save combined results ─────────────────────────────────────────────────

out_path = os.path.join(OUTPUT_DIR, "live_transfer_results.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved combined: {out_path}")

# ── Summary ───────────────────────────────────────────────────────────────

print(f"\n{'='*68}")
print("  LIVE TRANSFER SUMMARY")
print(f"{'='*68}")
print(f"  {'Pair':30} {'Task':12} {'Status':10} {'ΔT_src':>8} {'ΔT_tgt':>8} {'|gap|':>8}")
print(f"  {'-'*80}")

for pair_key, tasks in all_results.items():
    for task, r in sorted(tasks.items()):
        status = r["status"]
        src_dt = r["src_delta_t"]
        tgt_dt = r["best_candidate"]["delta_t"] if r.get("best_candidate") else "---"
        gap = r.get("delta_t_gap", "---")
        if isinstance(tgt_dt, float):
            tgt_dt = f"{tgt_dt:.4f}"
        if isinstance(gap, float):
            gap = f"{gap:.4f}"
        print(f"  {pair_key:30} {task:12} {status:10} {src_dt:>8.4f} {tgt_dt:>8} {gap:>8}")

print(f"\n  Results saved to {out_path}")
