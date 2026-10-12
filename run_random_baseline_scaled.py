"""
run_random_baseline_scaled.py
=============================
Random-circuit significance test for the twelve-model template evaluation
(fills the twelve-model gap of Table `random-baseline-summary`).

For every k* circuit on the s0 split of each task and variant, N size-matched
random subgraphs are sampled (same node and edge counts, uniform over the
model graph) and evaluated on the circuit's own discovery split. The real
circuit's stored performance is compared against the random distribution:
significant when at most alpha of the random subgraphs reach the real
circuit's Delta_T. Resumable per model.

Usage:
    python run_random_baseline_scaled.py                     # all 12 models
    python run_random_baseline_scaled.py --models pythia1b gpt2
    python run_random_baseline_scaled.py --n-random 25
Output:
    analysis/baselines/random_baseline_scaled.json
"""

import argparse
import glob
import json
import os
import re

parser = argparse.ArgumentParser(description="Scaled random-circuit baseline")
parser.add_argument("--models", nargs="*", default=None)
parser.add_argument("--n-random", type=int, default=25)
parser.add_argument("--alpha", type=float, default=0.05)
parser.add_argument("--device", default="auto")
parser.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16"],
                    help="bfloat16 if fp32 inference OOMs on the 7B class")
args = parser.parse_args()

HF = {"pythia14m": "EleutherAI/pythia-14m", "gpt2": "gpt2",
      "Qwen2.50.5B": "Qwen/Qwen2.5-0.5B", "pythia410m": "EleutherAI/pythia-410m",
      "gpt2large": "gpt2-large", "Qwen2.51.5B": "Qwen/Qwen2.5-1.5B",
      "pythia1b": "EleutherAI/pythia-1b", "Llama3.21B": "meta-llama/Llama-3.2-1B",
      "Qwen2.53B": "Qwen/Qwen2.5-3B", "Llama3.23B": "meta-llama/Llama-3.2-3B",
      "pythia6.9b": "EleutherAI/pythia-6.9b", "Qwen2.57B": "Qwen/Qwen2.5-7B"}


def batch_for(key):
    if "7B" in key or "6.9" in key: return 2
    if "3B" in key or "3b" in key: return 4
    return 8


ks = json.load(open("analysis/tables/kstar_summary.json"))
KEY = {m: f"{m}-k{ks[m]['kstar_tau05']}" for m in HF}
OUT = "analysis/baselines/random_baseline_scaled.json"
os.makedirs("analysis/baselines", exist_ok=True)
report = json.load(open(OUT)) if os.path.isfile(OUT) else {}

import torch
from functools import partial
from transformers import AutoTokenizer
import transformer_lens as tl
from trace.mechint.baselines import random_circuit
from trace.mechint.core.eap_evaluate import evaluate_graph, evaluate_baseline
from trace.mechint.core.eap_graph import Graph
from trace.mechint.core.eap_cnp import cnp_metric
from trace.mechint.core.analyzer import MechIntAnalyzer

dev = args.device
if dev == "auto":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
metric_fn = partial(cnp_metric, loss=False, mean=False)

for m in (args.models or list(HF)):
    if m in report:
        print(f"[{m}] already done — skip")
        continue
    circuits = sorted(glob.glob(f"analysis/{KEY[m]}_multi/results/"
                                f"*_v*_s0_results.json"))
    if not circuits:
        print(f"[{m}] no k* circuits — skip")
        continue
    print(f"\n== {m}: {len(circuits)} s0 circuits x {args.n_random} random ==",
          flush=True)
    import torch as _t
    model = tl.HookedTransformer.from_pretrained(
        HF[m], device=dev,
        dtype=_t.bfloat16 if args.dtype == "bfloat16" else _t.float32)
    tok = AutoTokenizer.from_pretrained(HF[m])
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    if hasattr(model.cfg, "ungroup_grouped_query_attention"):
        model.cfg.ungroup_grouped_query_attention = True
    model.eval()
    bs = batch_for(m)

    rows, cache = [], {}
    for cp in circuits:
        stem = os.path.basename(cp).replace("_results.json", "")
        task_v = re.match(r"(.+)_s0$", stem).group(1)
        csv = f"analysis/{m}/splits_variants/{task_v}_s0.csv"
        if not os.path.isfile(csv):
            continue
        rec = json.load(open(cp))
        c = rec["layer1"]["circuit"]
        met = rec["layer1"]["metrics"]
        if csv not in cache:
            an = MechIntAnalyzer(cnp_dataset_path=csv)
            ds = an.load_cnp_dataset(csv, tok)
            dl = ds.to_dataloader(bs)
            empty = Graph.from_model(model)
            empty.reset(empty=True)
            abl = evaluate_graph(model, empty, dl, metric_fn,
                                 skip_clean=False).mean().item()
            cache[csv] = (dl, abl)
        dl, abl = cache[csv]
        real_dt = met.get("circuit_performance", 0.0) - abl
        n_ge = 0
        for i in range(args.n_random):
            g = random_circuit(model, c["num_nodes"], c["num_edges"],
                               seed=1000 * i + len(rows))
            try:
                perf = evaluate_graph(model, g, dl, metric_fn).mean().item()
            except Exception as e:
                print(f"  {stem} rand{i}: {e}", flush=True)
                continue
            if perf - abl >= real_dt:
                n_ge += 1
        p = n_ge / args.n_random
        rows.append({"circuit": stem, "real_delta_t": round(real_dt, 4),
                     "p": round(p, 4), "significant": p < args.alpha})
        print(f"  {stem}: dT={real_dt:.3f} p={p:.3f} "
              f"{'SIG' if p < args.alpha else 'ns'}", flush=True)

    sig = sum(1 for r in rows if r["significant"])
    report[m] = {"n_circuits": len(rows), "n_random": args.n_random,
                 "significant": sig, "non_significant": len(rows) - sig,
                 "rate": round(sig / len(rows), 3) if rows else None,
                 "rows": rows}
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[{m}] {sig}/{len(rows)} significant "
          f"({100*sig/max(len(rows),1):.0f}%) -> {OUT}", flush=True)
    del model, tok, cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nSummary:")
for m, r in report.items():
    if r.get("rate") is not None:
        print(f"  {m:<13} {r['significant']}/{r['n_circuits']} ({r['rate']:.0%})")
