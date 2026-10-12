"""
run_random_baselines.py
========================
Generate random circuits matched to each real circuit's size,
measure AccT and ΔT, compare to real circuits.

Produces a permutation test: is the real circuit's ΔT significantly
higher than random circuits' ΔT?

Usage:
    python run_random_baselines.py --model EleutherAI/pythia-1b --n 50
    python run_random_baselines.py --model EleutherAI/pythia-14m --n 50
    python run_random_baselines.py --model meta-llama/Llama-3.2-1B --n 50
"""

import argparse
import json
import os
import sys
import numpy as np

parser = argparse.ArgumentParser(description="Random circuit baseline evaluation")
parser.add_argument("--model", default="EleutherAI/pythia-1b", help="HF model name")
parser.add_argument("--n", type=int, default=50, help="Number of random circuits per task")
parser.add_argument("--seed", type=int, default=42, help="Base random seed")
args = parser.parse_args()

MODEL = args.model
N_RANDOM = args.n
BASE_SEED = args.seed
MODEL_SHORT = MODEL.split("/")[-1].replace("-", "").replace("_", "")

# ── Resolve paths ────────────────────────────────────────────────────────

ANALYSIS_DIR = f"./analysis/{MODEL_SHORT}"
MULTI_DIR = f"./analysis/{MODEL_SHORT}_multi"
RESULTS_DIR = f"{ANALYSIS_DIR}/results"
MULTI_RESULTS_DIR = f"{MULTI_DIR}/results"
OUTPUT_DIR = f"{ANALYSIS_DIR}/baselines"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset registry
_DATASET_REGISTRY = {
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
DATASETS = _DATASET_REGISTRY.get(MODEL_SHORT, {})

# Also check splits directory for multi-circuit CSVs
SPLITS_DIR = f"{ANALYSIS_DIR}/splits"

# Find available tasks from BOTH original and multi results
ALL_TASKS_5 = ["IOI", "Location", "Time", "Path", "GreaterThan"]
ALL_TASKS_10 = ALL_TASKS_5 + ["Beneficiary", "Goal", "Instrument", "Source", "Topic"]

TASKS = []  # (label, results_json, dataset_csv)

# Original 5-task circuits
for task in ALL_TASKS_5:
    jp = os.path.join(RESULTS_DIR, f"{task.lower()}_results.json")
    csv = DATASETS.get(task, "")
    if os.path.isfile(jp) and os.path.isfile(csv):
        TASKS.append((task, jp, csv))

# Multi-circuit splits ({task}_s0, {task}_s1, {task}_s2)
for task in ALL_TASKS_10:
    for si in range(3):
        key = f"{task.lower()}_s{si}"
        jp = os.path.join(MULTI_RESULTS_DIR, f"{key}_results.json")
        # Try split CSV first, fall back to original dataset
        split_csv = os.path.join(SPLITS_DIR, f"{task.lower()}_s{si}.csv")
        orig_csv = DATASETS.get(task, "")
        csv = split_csv if os.path.isfile(split_csv) else orig_csv
        if os.path.isfile(jp) and csv and os.path.isfile(csv):
            TASKS.append((key, jp, csv))

if not TASKS:
    print(f"No tasks found in {RESULTS_DIR}/ or {MULTI_RESULTS_DIR}/")
    sys.exit(1)

print(f"Model: {MODEL} ({MODEL_SHORT})")
print(f"Tasks: {[t[0] for t in TASKS]}")
print(f"Random circuits per task: {N_RANDOM}")
print(f"Output: {OUTPUT_DIR}/")

# ── Load model ───────────────────────────────────────────────────────────

from transformers import AutoTokenizer
import transformer_lens as tl
import torch

print(f"\nLoading {MODEL}...")
model = tl.HookedTransformer.from_pretrained(MODEL, device='cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model.cfg.use_attn_result = True
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
if hasattr(model.cfg, 'ungroup_grouped_query_attention'):
    model.cfg.ungroup_grouped_query_attention = True
print(f"Model loaded ({model.cfg.n_layers}L, {model.cfg.n_heads}H)")

# ── Evaluate random circuits ─────────────────────────────────────────────

from trace.mechint.baselines import random_circuit_from_json
from trace.mechint.core.eap_evaluate import evaluate_graph, evaluate_baseline
from trace.mechint.core.eap_graph import Graph
from trace.mechint.core.eap_cnp import cnp_metric
from trace.mechint.pipeline import reconstruct_graph
from functools import partial

all_results = {}

for task_name, results_json, dataset_csv in TASKS:
    print(f"\n{'='*60}")
    print(f"  {task_name}")
    print(f"{'='*60}")

    # Load real circuit metrics
    with open(results_json) as f:
        rec = json.load(f)
    real_metrics = rec.get("layer1", {}).get("metrics", {})
    real_acc_t = real_metrics.get("AccT", 0)
    real_delta_t = real_metrics.get("DeltaT", 0)
    real_faith = real_metrics.get("faithfulness", 0)
    n_nodes = real_metrics.get("num_nodes_in_circuit", 10)
    n_edges = real_metrics.get("num_edges_in_circuit", 50)

    print(f"  Real circuit: AccT={real_acc_t:.4f}, ΔT={real_delta_t:.4f}, "
          f"faith={real_faith:.4f}, nodes={n_nodes}, edges={n_edges}")

    # Build dataloader
    try:
        from trace.mechint.core.analyzer import MechIntAnalyzer
        analyzer = MechIntAnalyzer(cnp_dataset_path=dataset_csv)
        ds = analyzer.load_cnp_dataset(dataset_csv, tokenizer)
        dataloader = ds.to_dataloader(10)
    except Exception as e:
        print(f"  ERROR building dataloader: {e}")
        continue

    metric_fn = partial(cnp_metric, loss=False, mean=False)

    # Baseline (full model)
    print("  Computing baseline...")
    baseline_perf = evaluate_baseline(model, dataloader, metric_fn).mean().item()

    # Ablated (empty circuit)
    empty = Graph.from_model(model)
    empty.reset(empty=True)
    ablated_perf = evaluate_graph(
        model, empty, dataloader, metric_fn, skip_clean=False
    ).mean().item()

    full_delta_t = baseline_perf - ablated_perf
    print(f"  Baseline={baseline_perf:.4f}, Ablated={ablated_perf:.4f}, "
          f"Full ΔT={full_delta_t:.4f}")

    # Real circuit performance
    real_graph = reconstruct_graph(model, results_json)
    real_circuit_perf = evaluate_graph(
        model, real_graph, dataloader, metric_fn
    ).mean().item()
    real_circuit_dt = real_circuit_perf - ablated_perf
    print(f"  Real circuit perf={real_circuit_perf:.4f}, "
          f"circuit ΔT (circ-ablated)={real_circuit_dt:.4f}")

    # Random circuits
    print(f"  Generating {N_RANDOM} random circuits (nodes={n_nodes}, edges={n_edges})...")
    random_perfs = []
    random_dts = []

    for i in range(N_RANDOM):
        rg = random_circuit_from_json(model, results_json, seed=BASE_SEED + i)
        try:
            rperf = evaluate_graph(
                model, rg, dataloader, metric_fn
            ).mean().item()
            rdt = rperf - ablated_perf
            random_perfs.append(rperf)
            random_dts.append(rdt)
        except Exception as e:
            if i == 0:
                print(f"    WARNING: eval failed: {e}")

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{N_RANDOM} done, "
                  f"mean random perf={np.mean(random_perfs):.4f}")

    if not random_perfs:
        print("  ERROR: all random circuits failed")
        continue

    # Permutation test: is real ΔT significantly > random ΔT?
    random_dts = np.array(random_dts)
    random_perfs = np.array(random_perfs)

    # One-sided p-value: fraction of random circuits with ΔT >= real
    p_value_dt = np.mean(random_dts >= real_circuit_dt)
    p_value_perf = np.mean(random_perfs >= real_circuit_perf)

    result = {
        "task": task_name,
        "model": MODEL_SHORT,
        "n_random": len(random_perfs),
        "real": {
            "AccT": round(real_acc_t, 4),
            "DeltaT": round(real_delta_t, 4),
            "circuit_perf": round(real_circuit_perf, 4),
            "circuit_dt": round(real_circuit_dt, 4),
            "faithfulness": round(real_faith, 4),
        },
        "baseline": round(baseline_perf, 4),
        "ablated": round(ablated_perf, 4),
        "random": {
            "perf_mean": round(float(np.mean(random_perfs)), 4),
            "perf_std": round(float(np.std(random_perfs)), 4),
            "dt_mean": round(float(np.mean(random_dts)), 4),
            "dt_std": round(float(np.std(random_dts)), 4),
            "perf_max": round(float(np.max(random_perfs)), 4),
            "dt_max": round(float(np.max(random_dts)), 4),
        },
        "permutation_test": {
            "p_value_dt": round(float(p_value_dt), 4),
            "p_value_perf": round(float(p_value_perf), 4),
            "significant_dt": bool(p_value_dt < 0.05),
            "significant_perf": bool(p_value_perf < 0.05),
        },
    }

    all_results[task_name] = result

    print(f"\n  Results:")
    print(f"    Real circuit ΔT:   {real_circuit_dt:.4f}")
    print(f"    Random mean ΔT:    {np.mean(random_dts):.4f} ± {np.std(random_dts):.4f}")
    print(f"    Random max ΔT:     {np.max(random_dts):.4f}")
    print(f"    p-value (ΔT):      {p_value_dt:.4f} "
          f"({'significant' if p_value_dt < 0.05 else 'NOT significant'})")
    print(f"    Real circuit perf: {real_circuit_perf:.4f}")
    print(f"    Random mean perf:  {np.mean(random_perfs):.4f} ± {np.std(random_perfs):.4f}")
    print(f"    p-value (perf):    {p_value_perf:.4f}")

# ── Save ─────────────────────────────────────────────────────────────────

out_path = os.path.join(OUTPUT_DIR, "random_circuit_baseline.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {out_path}")

# Summary table
print(f"\n{'='*70}")
print(f"  SUMMARY — Random Circuit Baseline ({MODEL_SHORT})")
print(f"{'='*70}")
print(f"  {'Task':12} {'Real ΔT':>10} {'Rand ΔT':>10} {'p-value':>10} {'Sig?':>6}")
print(f"  {'-'*50}")
for task_name, r in all_results.items():
    sig = "YES" if r["permutation_test"]["significant_dt"] else "no"
    print(f"  {task_name:12} {r['real']['circuit_dt']:10.4f} "
          f"{r['random']['dt_mean']:10.4f} "
          f"{r['permutation_test']['p_value_dt']:10.4f} {sig:>6}")

print(f"\n  All results saved to {out_path}")
