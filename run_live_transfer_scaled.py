"""
run_live_transfer_scaled.py
===========================
Live transfer validation at the twelve-model scale (option B design):
every model serves as TARGET with three SOURCES (one within-family, two
cross-family), giving 36 ordered pairs. For each shared task the source
signature queries the target's k* signature registry; the top-1 candidate
task's best circuit (highest stored faithfulness in the target's k* pool)
is reconstructed in the target model and evaluated LIVE on the target's
query-task prompts, with the original acceptance thresholds
(theta_behav = 0.50, theta_causal = 0.10 on the CNP metric scale).

Resumable: finished (source -> target) pairs found in the output are
skipped, so a killed session loses at most one pair.

Usage:
    python run_live_transfer_scaled.py                 # all 36 pairs
    python run_live_transfer_scaled.py --targets pythia1b Qwen2.57B
Output:
    analysis/transfer/live_transfer_scaled.json
    analysis/transfer/live_scaled_<target>.json (per target)
"""

import argparse
import glob
import json
import os
import re

parser = argparse.ArgumentParser(description="Scaled live transfer")
parser.add_argument("--targets", nargs="*", default=None)
parser.add_argument("--device", default="auto")
parser.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16"],
                    help="bfloat16 if fp32 inference OOMs on the 7B class")
parser.add_argument("--split", default="v1_s0",
                    help="query split suffix per task (default v1_s0)")
args = parser.parse_args()

HF = {"pythia14m": "EleutherAI/pythia-14m", "gpt2": "gpt2",
      "Qwen2.50.5B": "Qwen/Qwen2.5-0.5B", "pythia410m": "EleutherAI/pythia-410m",
      "gpt2large": "gpt2-large", "Qwen2.51.5B": "Qwen/Qwen2.5-1.5B",
      "pythia1b": "EleutherAI/pythia-1b", "Llama3.21B": "meta-llama/Llama-3.2-1B",
      "Qwen2.53B": "Qwen/Qwen2.5-3B", "Llama3.23B": "meta-llama/Llama-3.2-3B",
      "pythia6.9b": "EleutherAI/pythia-6.9b", "Qwen2.57B": "Qwen/Qwen2.5-7B"}

# target -> three sources (one within-family, two cross-family)
SOURCES = {
    "pythia14m":  ["pythia1b", "gpt2", "Qwen2.51.5B"],
    "gpt2":       ["gpt2large", "pythia410m", "Llama3.21B"],
    "Qwen2.50.5B": ["Qwen2.51.5B", "pythia410m", "gpt2"],
    "pythia410m": ["pythia1b", "Qwen2.51.5B", "gpt2large"],
    "gpt2large":  ["gpt2", "pythia6.9b", "Qwen2.53B"],
    "pythia1b":   ["pythia410m", "Llama3.21B", "Qwen2.51.5B"],
    "Qwen2.51.5B": ["Qwen2.50.5B", "pythia410m", "Llama3.21B"],
    "Llama3.21B": ["Llama3.23B", "pythia1b", "Qwen2.51.5B"],
    "Qwen2.53B":  ["Qwen2.57B", "Llama3.23B", "gpt2large"],
    "Llama3.23B": ["Llama3.21B", "pythia6.9b", "Qwen2.53B"],
    "pythia6.9b": ["pythia1b", "Qwen2.57B", "Llama3.23B"],
    "Qwen2.57B":  ["Qwen2.53B", "pythia6.9b", "Llama3.23B"],
}

BATCH = {"7": 2, "6.9": 2, "3B": 4, "3b": 4}
def batch_for(key):
    if "7B" in key or "6.9" in key: return 2
    if "3B" in key or "3b" in key: return 4
    return 8

THETA_BEHAV, THETA_CAUSAL, DELTA_ILP = 0.50, 0.10, 0.30
OUT_DIR = "analysis/transfer"
os.makedirs(OUT_DIR, exist_ok=True)

ks = json.load(open("analysis/tables/kstar_summary.json"))
KEY = {m: f"{m}-k{ks[m]['kstar_tau05']}" for m in HF}

# ── Signature registries (CPU, once) ─────────────────────────────────────
print("Learning k* signature registries (CPU)...")
from run_transfer_matrix import load_task_circuits, learn_signatures
from trace.mechint.storage.ilp.ilp_signature_distance import ilp_signature_distance

REG = {}
for m, key in KEY.items():
    REG[m] = learn_signatures(key, load_task_circuits(key, "variant"))
    print(f"  {m}: {len(REG[m])} task signatures", flush=True)


def best_circuit_json(model, task):
    """Target's k* circuit for `task` with the highest stored faithfulness."""
    best, best_f = None, -1e9
    for p in glob.glob(f"analysis/{KEY[model]}_multi/results/"
                       f"{task}_v*_s*_results.json"):
        f = json.load(open(p))["layer1"]["metrics"].get("faithfulness", -1e9)
        if f is not None and f > best_f:
            best, best_f = p, f
    return best


def query_csv(model, task):
    p = f"analysis/{model}/splits_variants/{task}_{args.split}.csv"
    return p if os.path.isfile(p) else None


# ── Main loop, grouped by target ─────────────────────────────────────────
import torch
from functools import partial
from transformers import AutoTokenizer
import transformer_lens as tl
from trace.mechint.pipeline import reconstruct_graph
from trace.mechint.core.eap_evaluate import evaluate_graph, evaluate_baseline
from trace.mechint.core.eap_graph import Graph
from trace.mechint.core.eap_cnp import cnp_metric
from trace.mechint.core.analyzer import MechIntAnalyzer

dev = args.device
if dev == "auto":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
metric_fn = partial(cnp_metric, loss=False, mean=False)

targets = args.targets or list(SOURCES)
combined_path = f"{OUT_DIR}/live_transfer_scaled.json"
combined = json.load(open(combined_path)) if os.path.isfile(combined_path) else {}

for tgt in targets:
    per_tgt_path = f"{OUT_DIR}/live_scaled_{tgt}.json"
    per_tgt = json.load(open(per_tgt_path)) if os.path.isfile(per_tgt_path) else {}
    todo = [s for s in SOURCES[tgt] if f"{s}_to_{tgt}" not in per_tgt]
    if not todo:
        print(f"[{tgt}] all pairs done — skip")
        combined.update(per_tgt)
        continue

    print(f"\n{'='*66}\n TARGET {tgt} ({HF[tgt]}), sources: {todo}\n{'='*66}",
          flush=True)
    import torch as _t
    model = tl.HookedTransformer.from_pretrained(
        HF[tgt], device=dev,
        dtype=_t.bfloat16 if args.dtype == "bfloat16" else _t.float32)
    tok = AutoTokenizer.from_pretrained(HF[tgt])
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    if hasattr(model.cfg, "ungroup_grouped_query_attention"):
        model.cfg.ungroup_grouped_query_attention = True
    model.eval()
    bs = batch_for(tgt)

    # per-(target, task) baseline/ablated cache
    cache = {}

    def task_env(task):
        if task in cache:
            return cache[task]
        csv = query_csv(tgt, task)
        if not csv:
            cache[task] = None
            return None
        an = MechIntAnalyzer(cnp_dataset_path=csv)
        ds = an.load_cnp_dataset(csv, tok)
        dl = ds.to_dataloader(bs)
        base = evaluate_baseline(model, dl, metric_fn).mean().item()
        empty = Graph.from_model(model)
        empty.reset(empty=True)
        abl = evaluate_graph(model, empty, dl, metric_fn,
                             skip_clean=False).mean().item()
        cache[task] = (dl, base, abl)
        return cache[task]

    for src in todo:
        pair = f"{src}_to_{tgt}"
        res = {}
        shared = sorted(set(REG[src]) & set(REG[tgt]))
        for task in shared:
            env = task_env(task)
            if env is None:
                continue
            dl, base, abl = env
            dists = {u: float(ilp_signature_distance(REG[src][task],
                                                     REG[tgt][u]))
                     for u in REG[tgt]}
            cand_task = min(dists, key=dists.get)
            cand_json = best_circuit_json(tgt, cand_task)
            if not cand_json:
                continue
            try:
                g = reconstruct_graph(model, cand_json)
                perf = evaluate_graph(model, g, dl, metric_fn).mean().item()
            except Exception as e:
                print(f"  [{pair}] {task}: eval failed: {e}", flush=True)
                continue
            dt = perf - abl
            entry = {
                "candidate_task": cand_task,
                "same_task": cand_task == task,
                "d_ilp": round(dists[cand_task], 4),
                "within_delta": dists[cand_task] < DELTA_ILP,
                "tgt_baseline": round(base, 4),
                "tgt_ablated": round(abl, 4),
                "acc_t": round(perf, 4),
                "delta_t": round(dt, 4),
                "passed_behav": perf >= THETA_BEHAV,
                "passed_causal": dt >= THETA_CAUSAL,
                "accepted": perf >= THETA_BEHAV and dt >= THETA_CAUSAL,
                "candidate_json": cand_json,
            }
            res[task] = entry
            print(f"  [{pair}] {task}: cand={cand_task} "
                  f"({'same' if entry['same_task'] else 'diff'}) "
                  f"acc={perf:.3f} dT={dt:.3f} "
                  f"{'ACCEPT' if entry['accepted'] else 'reject'}", flush=True)
        per_tgt[pair] = res
        with open(per_tgt_path, "w") as f:
            json.dump(per_tgt, f, indent=1)

    combined.update(per_tgt)
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=1)
    del model, tok, cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── summary ──────────────────────────────────────────────────────────────
n_pairs = len(combined)
rows = [(p, t, e) for p, r in combined.items() for t, e in r.items()]
acc = [e for _, _, e in rows if e["accepted"]]
same = [e for _, _, e in rows if e["accepted"] and e["same_task"]]
print(f"\n== LIVE TRANSFER SCALED: {n_pairs} pairs, {len(rows)} (pair,task) "
      f"evaluations ==")
if rows:
    print(f"  accepted: {len(acc)}/{len(rows)} "
          f"({100*len(acc)/len(rows):.0f}%); same-task among accepted: "
          f"{len(same)}/{len(acc) if acc else 1}")
print(f"Report: {combined_path}")
