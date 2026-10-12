"""Probe: does the current PYTHONHASHSEED reproduce the archived distance
matrices for the three tie-affected models? Exit 0 iff all three match."""
import json
import sys

import numpy as np

from run_transfer_matrix import load_task_circuits, learn_signatures
from trace.mechint.storage.ilp.ilp_signature_distance import ilp_signature_distance

KS = json.load(open("analysis/tables/kstar_summary.json"))
OLD = json.load(open("analysis/tables/distance_matrices_kstar.json"))
ORDER = ["ioi", "greaterthan", "beneficiary", "goal", "instrument", "location",
         "path", "source", "time", "topic"]

ok = True
for m in ["gpt2large", "pythia6.9b", "Qwen2.51.5B"]:
    key = f"{m}-k{KS[m]['kstar_tau05']}"
    reg = learn_signatures(key, load_task_circuits(key, "variant"))
    tasks = [t for t in ORDER if t in reg]
    n = len(tasks)
    D = np.zeros((n, n))
    for i, a in enumerate(tasks):
        for j, b in enumerate(tasks):
            if i != j:
                D[i, j] = float(ilp_signature_distance(reg[a], reg[b]))
    match = (tasks == OLD[m]["tasks"]
             and np.allclose(D, np.array(OLD[m]["matrix"]), atol=5e-4))
    print(f"  {m}: {'match' if match else 'DIFFER'}", flush=True)
    ok = ok and match
sys.exit(0 if ok else 1)
