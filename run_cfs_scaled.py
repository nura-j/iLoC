"""
run_cfs_scaled.py
=================
CFS (layer 2) for the twelve-model template evaluation: per-component DLA
scores and attribution-weighted attention profiles for every k* s0 circuit.
The template pipeline stored layer2 = null, so the CFS summary table and
DLA figures cover only the primary trio; this fills the gap.

For each (model, task): the v1_s0 k* circuit is reconstructed from its
results JSON, CFS is computed on up to 50 examples of the same split with
the task family's role labellers, and the result is written to
analysis/cfs_scaled/{model}/{task}.json (CFSResult.to_dict() plus meta).
Resumable per (model, task).

Usage (GPU server):
    CUDA_VISIBLE_DEVICES=0 python run_cfs_scaled.py
    python run_cfs_scaled.py --models pythia410m gpt2
Afterwards: python plot_cfs_figures.py picks the records up automatically.
"""

import argparse
import glob
import json
import os
import re

parser = argparse.ArgumentParser(description="Scaled CFS extraction")
parser.add_argument("--models", nargs="*", default=None)
parser.add_argument("--device", default="auto")
parser.add_argument("--max-examples", type=int, default=50)
parser.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16"])
args = parser.parse_args()

HF = {"pythia14m": "EleutherAI/pythia-14m", "gpt2": "gpt2",
      "Qwen2.50.5B": "Qwen/Qwen2.5-0.5B", "pythia410m": "EleutherAI/pythia-410m",
      "gpt2large": "gpt2-large", "Qwen2.51.5B": "Qwen/Qwen2.5-1.5B",
      "pythia1b": "EleutherAI/pythia-1b", "Llama3.21B": "meta-llama/Llama-3.2-1B",
      "Qwen2.53B": "Qwen/Qwen2.5-3B", "Llama3.23B": "meta-llama/Llama-3.2-3B",
      "pythia6.9b": "EleutherAI/pythia-6.9b", "Qwen2.57B": "Qwen/Qwen2.5-7B"}

ks = json.load(open("analysis/tables/kstar_summary.json"))
KEY = {m: f"{m}-k{ks[m]['kstar_tau05']}" for m in HF}
OUT = "analysis/cfs_scaled"

import pandas as pd
import torch
import transformer_lens as tl
from transformers import AutoTokenizer
from trace.mechint.pipeline import reconstruct_graph
from trace.mechint.cfs import compute_cfs
from trace.mechint.cfs.role_labellers import get_labellers

dev = args.device
if dev == "auto":
    dev = "cuda" if torch.cuda.is_available() else "cpu"

for m in (args.models or list(HF)):
    paths = sorted(glob.glob(f"analysis/{KEY[m]}_multi/results/"
                             f"*_v1_s0_results.json"))
    todo = []
    for p in paths:
        task = re.match(r"(.+)_v1_s0_results\.json$",
                        os.path.basename(p)).group(1)
        if not os.path.isfile(f"{OUT}/{m}/{task}.json"):
            todo.append((task, p))
    if not todo:
        print(f"[{m}] all tasks done — skip")
        continue

    print(f"\n== {m}: {len(todo)} tasks ==", flush=True)
    model = tl.HookedTransformer.from_pretrained(
        HF[m], device=dev,
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)
    tok = AutoTokenizer.from_pretrained(HF[m])
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    if hasattr(model.cfg, "ungroup_grouped_query_attention"):
        model.cfg.ungroup_grouped_query_attention = True
    model.eval()
    os.makedirs(f"{OUT}/{m}", exist_ok=True)

    for task, jpath in todo:
        csv = f"analysis/{m}/splits_variants/{task}_v1_s0.csv"
        if not os.path.isfile(csv):
            print(f"  {task}: no split csv — skip")
            continue
        rec = json.load(open(jpath))
        family = (rec.get("meta") or {}).get("task_family", "")
        df = pd.read_csv(csv)
        examples = [{"clean": str(r.get("clean", r.get("prompt", ""))),
                     "correct": str(r.get("target_word", r.get("answer", "")))}
                    for _, r in df.head(args.max_examples).iterrows()]
        ling_roles, _, ling_labeller, _ = get_labellers(family)

        def _lab(text, token_strs):
            return ling_labeller(text, token_strs, None)

        try:
            graph = reconstruct_graph(model, jpath)
            cfs = compute_cfs(model, graph, examples, tok,
                              epsilon_dla=0.05, token_roles=ling_roles,
                              role_labeller=_lab)
        except Exception as e:
            print(f"  {task}: FAILED {e}", flush=True)
            continue
        out = cfs.to_dict()
        out["meta"] = {"model": m, "task": task, "task_family": family,
                       "kstar": ks[m]["kstar_tau05"], "split": "v1_s0",
                       "n_examples": len(examples)}
        with open(f"{OUT}/{m}/{task}.json", "w") as f:
            json.dump(out, f, indent=1)
        caus = sum(1 for c in out["components"].values()
                   if not c["causally_marginal"])
        print(f"  {task}: {len(out['components'])} components, "
              f"{caus} causal", flush=True)

    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nDone. Results written to analysis/cfs_scaled/")
