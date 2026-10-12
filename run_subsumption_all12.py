"""
run_subsumption_all12.py
========================
Systematic theta-subsumption analysis over ALL twelve models' k* pools,
extending the single-model worked example of the paper. For each model,
one clause per task is selected with the standard one-vs-rest procedure
(the same clause sets as plot_distance_matrices_all12.py; clause-selection
ties resolve by generation order, so PYTHONHASHSEED=0 is pinned to match
the archived registries), and the pairwise relation is computed under
strict and relaxed (numeric-wildcard) theta-subsumption.

Usage: python3 run_subsumption_all12.py
Output: analysis/tables/subsumption_all12.json (+ console summary)
"""

import json
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from run_transfer_matrix import load_task_circuits
from trace.mechint.storage.ilp.ilp_arch import ArchitecturalILPLearner
from trace.mechint.storage.ilp.subsumption import subsumption_matrix

KS = json.load(open("analysis/tables/kstar_summary.json"))
MODELS = ["gpt2", "gpt2large", "pythia14m", "pythia410m", "pythia1b",
          "pythia6.9b", "Qwen2.50.5B", "Qwen2.51.5B", "Qwen2.53B",
          "Qwen2.57B", "Llama3.21B", "Llama3.23B"]

def learn_clauses(model_key, task_circuits):
    """{task: clause_text} via the standard one-vs-rest selection."""
    learner = ArchitecturalILPLearner()
    clauses = {}
    for task, circuits in task_circuits.items():
        pos_feats = [f for f, _ in circuits]
        pos_raw = [r for _, r in circuits]
        neg_feats = [f for o, cs in task_circuits.items() if o != task
                     for f, _ in cs]
        neg_raw = [r for o, cs in task_circuits.items() if o != task
                   for _, r in cs]
        if not pos_feats or not neg_feats:
            continue
        clause, _, _ = learner._select_best_clause(
            task, pos_feats, neg_feats, pos_raw, neg_raw)
        clauses[task] = clause
    return clauses

def summarise(matrix):
    """Unordered-pair counts: ordered (one strictly generalises the other),
    equivalent, incomparable; plus the most-general tasks (each of whose
    clause generalises or equals every other clause)."""
    names = sorted(matrix)
    ordered = equivalent = incomparable = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rel = matrix[a][b]
            if rel == "equivalent":
                equivalent += 1
            elif rel in ("generalises", "specialises"):
                ordered += 1
            else:
                incomparable += 1
    top = [a for a in names
           if all(matrix[a][b] in ("generalises", "equivalent")
                  for b in names if b != a)]
    return {"n_tasks": len(names), "pairs_ordered": ordered,
            "pairs_equivalent": equivalent, "pairs_incomparable": incomparable,
            "most_general": top}

out = {}
for m in MODELS:
    key = f"{m}-k{KS[m]['kstar_tau05']}"
    clauses = learn_clauses(key, load_task_circuits(key, "variant"))
    strict = subsumption_matrix(clauses, relaxed=False)
    relaxed = subsumption_matrix(clauses, relaxed=True)
    out[m] = {"clauses": clauses,
              "strict": strict, "relaxed": relaxed,
              "strict_summary": summarise(strict),
              "relaxed_summary": summarise(relaxed)}
    ss, rs = out[m]["strict_summary"], out[m]["relaxed_summary"]
    print(f"{m}: n={rs['n_tasks']}  strict ord/eq/inc = "
          f"{ss['pairs_ordered']}/{ss['pairs_equivalent']}/{ss['pairs_incomparable']}  "
          f"relaxed ord/eq/inc = "
          f"{rs['pairs_ordered']}/{rs['pairs_equivalent']}/{rs['pairs_incomparable']}  "
          f"top(relaxed) = {rs['most_general']}", flush=True)

json.dump(out, open("analysis/tables/subsumption_all12.json", "w"), indent=1)
tot = lambda mode, k: sum(out[m][mode][k] for m in MODELS)
for mode in ("strict_summary", "relaxed_summary"):
    o, e, i = (tot(mode, "pairs_ordered"), tot(mode, "pairs_equivalent"),
               tot(mode, "pairs_incomparable"))
    print(f"TOTAL {mode}: ordered {o}, equivalent {e}, incomparable {i}, "
          f"of {o + e + i} pairs")
print("-> analysis/tables/subsumption_all12.json")
