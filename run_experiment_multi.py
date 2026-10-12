"""
run_experiment_multi.py
========================
COMPASS multi-circuit pipeline — Strategy A+B combined.

Strategy A: All 12 tasks (10 semantic roles + IOI + GT)
Strategy B: k splits per task → multiple circuits per class

This produces up to 12×k circuits per model, enabling ILP to learn from
multiple positive examples and breaking the F1=1.0 circularity.

Usage:
    # Step 1: create splits (no GPU needed)
    python split_datasets.py --model pythia1b --k 3

    # Step 2: run multi-circuit pipeline
    python run_experiment_multi.py --model pythia1b --k 3
    python run_experiment_multi.py --model pythia1b --k 3 --skip-eap  # reuse cached
    python run_experiment_multi.py --model pythia1b --k 1             # Strategy A only

Output structure:
    analysis/{model}_multi/results/{task}_s{split}_results.json
    analysis/{model}_multi/ilp/                                  # ILP with multi-pos
    analysis/{model}_multi/baselines/                            # baseline comparison
"""

import argparse
import gc
import json
import os
import sys

# ── Args ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="COMPASS multi-circuit pipeline")
parser.add_argument("--model", default="EleutherAI/pythia-1b",
                    help="HF model name (default: EleutherAI/pythia-1b)")
parser.add_argument("--k", type=int, default=3,
                    help="Number of splits per task (1 = Strategy A only)")
parser.add_argument("--skip-eap", action="store_true",
                    help="Skip EAP; load existing circuits from JSON")
parser.add_argument("--skip-cfs", action="store_true",
                    help="Skip CFS computation")
parser.add_argument("--top-k", type=int, default=200,
                    help="Top-K edges for EAP (default: 200)")
parser.add_argument("--roles", nargs="*", default=None,
                    help="Subset of roles to run (default: all available)")
parser.add_argument("--variants", nargs="*", default=None,
                    help="Variant keys (e.g. v1 v2 v3): run (task x variant) "
                         "discovery from splits_variants/; skips the pooled "
                         "ILP/baseline stages (use run_heldout.py --by variant)")
parser.add_argument("--splits-dir", default=None,
                    help="Override the split CSV directory")
parser.add_argument("--out-suffix", default="",
                    help="Analysis dir suffix: '_extra' writes to "
                         "analysis/{model}_extra_multi (keeps the extra "
                         "task pool separate from the core results)")
parser.add_argument("--k-sweep", default=None,
                    help="Comma-separated edge budgets to evaluate faithfulness "
                         "at (e.g. 200,500,1000,2000); the exported circuit "
                         "uses --top-k")
parser.add_argument("--batch-size", type=int, default=10,
                    help="EAP dataloader batch size (reduce for large models)")
parser.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"],
                    help="Model dtype (bfloat16 halves memory; last resort)")
parser.add_argument("--device", default="cpu",
                    help="cpu | cuda | mps | auto")
args = parser.parse_args()

BASE_MODEL = args.model
TOP_K = args.top_k
N_SPLITS = args.k
SKIP_EAP = args.skip_eap
SKIP_CFS = args.skip_cfs
VARIANTS = args.variants or []

MODEL_SHORT = BASE_MODEL.split("/")[-1].replace("-", "").replace("_", "")

# ── Task definitions ──────────────────────────────────────────────────────

TASK_FAMILY_SEMROLES = "SemanticRoles"
TASK_FAMILY_IOI = "IOI"
TASK_FAMILY_GT = "GreaterThan"

ALL_TASKS = [
    # Original 5
    {"task": "IOI",         "task_family": TASK_FAMILY_IOI,      "role": "IOI",
     "operation_type": "selection",  "logical_category": "PREDICATE"},
    {"task": "GreaterThan", "task_family": TASK_FAMILY_GT,       "role": "GreaterThan",
     "operation_type": "comparison", "logical_category": "OPERATOR"},
    # All 10 semantic roles
    {"task": "Location",    "task_family": TASK_FAMILY_SEMROLES, "role": "Location",
     "operation_type": "binding",    "logical_category": "ROLE"},
    {"task": "Time",        "task_family": TASK_FAMILY_SEMROLES, "role": "Time",
     "operation_type": "binding",    "logical_category": "ROLE"},
    {"task": "Path",        "task_family": TASK_FAMILY_SEMROLES, "role": "Path",
     "operation_type": "binding",    "logical_category": "ROLE"},
    {"task": "Beneficiary", "task_family": TASK_FAMILY_SEMROLES, "role": "Beneficiary",
     "operation_type": "binding",    "logical_category": "ROLE"},
    {"task": "Goal",        "task_family": TASK_FAMILY_SEMROLES, "role": "Goal",
     "operation_type": "binding",    "logical_category": "ROLE"},
    {"task": "Instrument",  "task_family": TASK_FAMILY_SEMROLES, "role": "Instrument",
     "operation_type": "binding",    "logical_category": "ROLE"},
    # Manner and Material removed — ill-defined roles with inconsistent circuits
    {"task": "Source",       "task_family": TASK_FAMILY_SEMROLES, "role": "Source",
     "operation_type": "binding",    "logical_category": "ROLE"},
    {"task": "Topic",       "task_family": TASK_FAMILY_SEMROLES, "role": "Topic",
     "operation_type": "binding",    "logical_category": "ROLE"},
    # Extra tasks — separate pool (data/variants_extra);
    # only run when named via --roles, so the core pipeline never sees them
    {"task": "ColoredObjects", "task_family": "ColoredObjects",
     "role": "ColoredObjects", "operation_type": "binding",
     "logical_category": "ATTRIBUTE", "extra": True},
    {"task": "SVA", "task_family": "SVA", "role": "SVA",
     "operation_type": "agreement", "logical_category": "FEATURE",
     "extra": True},
]

# Filter tasks if --roles specified; extra tasks run only when asked for
if args.roles:
    allowed = set(r.lower() for r in args.roles)
    ALL_TASKS = [t for t in ALL_TASKS if t["task"].lower() in allowed]
else:
    ALL_TASKS = [t for t in ALL_TASKS if not t.get("extra")]

# Variant mode: expand each task into task_vN pseudo-tasks; split files are
# named {task}_vN_s{i}.csv and results {task}_vN_s{i}_results.json
if VARIANTS:
    ALL_TASKS = [{**t, "task": f"{t['task']}_{v}"}
                 for t in ALL_TASKS for v in VARIANTS]

# ── Output dirs ───────────────────────────────────────────────────────────

ANALYSIS_ROOT = "./analysis"
MODEL_DIR = f"{ANALYSIS_ROOT}/{MODEL_SHORT}{args.out_suffix}_multi"
SPLITS_DIR = args.splits_dir or (
    f"{ANALYSIS_ROOT}/{MODEL_SHORT}/splits_variants" if VARIANTS
    else f"{ANALYSIS_ROOT}/{MODEL_SHORT}/splits")
JSON_OUTPUT_DIR = f"{MODEL_DIR}/results"
OWL_OUTPUT_DIR = f"{MODEL_DIR}/circuits"
CFS_OUTPUT_DIR = f"{MODEL_DIR}/cfs"
ILP_DIR = f"{MODEL_DIR}/ilp"
BASELINES_DIR = f"{MODEL_DIR}/baselines"

for d in [JSON_OUTPUT_DIR, OWL_OUTPUT_DIR, CFS_OUTPUT_DIR, ILP_DIR, BASELINES_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Config: {BASE_MODEL} ({MODEL_SHORT}), k={N_SPLITS} splits/task")
print(f"Output: {MODEL_DIR}/")
print(f"Tasks:  {[t['task'] for t in ALL_TASKS]}")


# ── Resolve split CSVs ───────────────────────────────────────────────────

def find_split_csvs(task_name: str):
    """Find split CSV files for a task, or fall back to the original."""
    splits = []
    for i in range(N_SPLITS):
        path = os.path.join(SPLITS_DIR, f"{task_name.lower()}_s{i}.csv")
        if os.path.isfile(path):
            splits.append(path)
    return splits


# Check splits exist
_missing_splits = []
for t in ALL_TASKS:
    csvs = find_split_csvs(t["task"])
    if not csvs:
        _missing_splits.append(t["task"])

if _missing_splits:
    print(f"\nWARNING: No splits found for: {_missing_splits}")
    print(f"Run first:  python split_datasets.py --model {MODEL_SHORT} --k {N_SPLITS}")
    if not SKIP_EAP:
        print("Continuing with available splits only...")


# ── Load model ────────────────────────────────────────────────────────────

from transformers import AutoTokenizer
import torch
import transformer_lens as tl

_device = args.device
if _device == "auto":
    _device = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available()
               else "cpu")

import importlib.metadata as _md
print(f"\ntorch {torch.__version__}, transformer_lens {_md.version('transformer_lens')}")
print(f"Loading {BASE_MODEL} on {_device}...")
# NOTE: freezing parameter gradients (keeping only the embedding as a
# grad-requiring leaf) was tested as a memory optimisation and REJECTED:
# it measurably changes EAP-IG scores on a deterministic backend
# (max range shifts ~1.4x), so all parameters keep requires_grad=True.
_dtype = getattr(torch, args.dtype)
model = tl.HookedTransformer.from_pretrained(BASE_MODEL, device=_device,
                                             dtype=_dtype)
_param_device = str(next(model.parameters()).device)
if str(model.cfg.device) != _param_device:
    print(f"WARNING: model.cfg.device={model.cfg.device!r} but parameters are on "
          f"{_param_device!r} — syncing cfg.device to the parameters")
    model.cfg.device = _param_device
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model.cfg.use_attn_result = True
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True
print(f"Model loaded ({model.cfg.n_layers}L, {model.cfg.n_heads}H)")


# ── Circuit discovery: k splits per task ──────────────────────────────────

from trace.mechint import analyze_and_export, reconstruct_graph
from trace.mechint.storage.results_export import export_task_json

# {task_name: [(split_idx, json_path, graph, results), ...]}
_circuits = {}
_json_paths_all = {}  # flat: "task_s0" -> json_path

for _t in ALL_TASKS:
    tn = _t["task"]
    split_csvs = find_split_csvs(tn)

    if not split_csvs:
        print(f"\n[SKIP] {tn}: no split CSVs")
        continue

    _circuits[tn] = []

    for si, csv_path in enumerate(split_csvs):
        key = f"{tn.lower()}_s{si}"
        jp = os.path.join(JSON_OUTPUT_DIR, f"{key}_results.json")

        if os.path.isfile(jp):
            print(f"[CACHED] {key}")
            _circuits[tn].append((si, jp))
            _json_paths_all[key] = jp
        elif SKIP_EAP:
            print(f"[SKIP]   {key}: no cached result")
        else:
            print(f"[NEW]    {key} <- {os.path.basename(csv_path)}")
            try:
                res, graph, owl = analyze_and_export(
                    model=model, cnp_dataset_path=csv_path,
                    task_family=_t["task_family"], role=_t["role"],
                    model_name=MODEL_SHORT, tokenizer=tokenizer,
                    top_k=TOP_K, ig_steps=3, compute_validation=False,
                    output_dir=OWL_OUTPUT_DIR,
                    operation_type=_t["operation_type"],
                    logical_category=_t["logical_category"],
                    # unique per (task, variant, split) so TTLs never collide
                    experiment_id=f"{_t['task_family'].lower()}_{key}",
                    eap_batch_size=args.batch_size,
                    k_sweep=([int(x) for x in args.k_sweep.split(",")]
                             if args.k_sweep else None),
                )
                # Export with split-indexed name
                jp_out = export_task_json(
                    task=tn, task_family=_t["task_family"], role=_t["role"],
                    model_name=MODEL_SHORT, results=res, graph=graph,
                    owl_path=owl, out_dir=JSON_OUTPUT_DIR,
                    operation_type=_t["operation_type"],
                    logical_category=_t["logical_category"],
                    filename_override=f"{key}_results.json",
                )
                _circuits[tn].append((si, jp_out))
                _json_paths_all[key] = jp_out
                faith = res.get("faithfulness", "N/A")
                nodes = res.get("num_nodes_in_circuit", "?")
                print(f"         Faith={faith}, nodes={nodes}")
                # Everything needed downstream is in the results JSON.
                # A 32-layer model's Graph holds millions of edge objects;
                # retaining one per circuit (~90) exhausted host RAM and
                # got the pythia-6.9b run OOM-killed on the final circuit.
                del graph, res
            except Exception as e:
                print(f"         ERROR: {e}")
                import traceback; traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Summary
total_circuits = sum(len(v) for v in _circuits.values())
tasks_with_circuits = [tn for tn, cs in _circuits.items() if cs]
print(f"\nCircuit inventory: {total_circuits} circuits across "
      f"{len(tasks_with_circuits)} tasks")
for tn in sorted(_circuits.keys()):
    n = len(_circuits[tn])
    if n:
        print(f"  {tn:14}: {n} circuit(s)")

if total_circuits < 3:
    print("\nToo few circuits for ILP. Exiting.")
    sys.exit(0)

if VARIANTS:
    print(f"\n{'='*68}")
    print("VARIANTS MODE — circuit discovery complete.")
    print("Pooled generalisation analysis runs separately:")
    print(f"  python run_heldout.py --model {MODEL_SHORT} --by variant")
    print(f"{'='*68}")
    sys.exit(0)


# ── ILP_arch with multiple positives ──────────────────────────────────────

from trace.mechint.storage.ilp.ilp_arch import ArchitecturalILPLearner
from trace.mechint.storage.results_export import add_ilp_architectural

print(f"\n{'='*68}")
print(f"ILP_arch — multi-positive learning ({total_circuits} circuits)")
print(f"{'='*68}")

# Load all records
_all_records = {}  # key -> (record, json_path)
for key, jp in _json_paths_all.items():
    with open(jp) as f:
        _all_records[key] = (json.load(f), jp)

# Map task -> family for grouping
_task_to_family = {t["task"]: t["task_family"] for t in ALL_TASKS}

learner = ArchitecturalILPLearner()
arch_results = {}

for target_task in sorted(tasks_with_circuits):
    # Positive records: all splits for this task
    pos_keys = [k for k in _all_records if k.startswith(target_task.lower() + "_s")]
    neg_keys = [k for k in _all_records if not k.startswith(target_task.lower() + "_s")]

    if not pos_keys or not neg_keys:
        print(f"  {target_task}: need both pos and neg, skipping")
        continue

    pos_records = [_all_records[k][0] for k in pos_keys]
    neg_records = [_all_records[k][0] for k in neg_keys]

    print(f"\n  {target_task}: {len(pos_records)} positive, {len(neg_records)} negative")

    try:
        result = learner.learn_from_json_records(
            target_type=target_task.lower(),
            positive_records=pos_records,
            negative_records=neg_records,
        )
        arch_results[target_task.lower()] = result

        clause = result["clause_text"]
        conf = result["confidence"]
        loo = result.get("loo_f1")
        print(f"    clause: {clause[:80]}...")
        print(f"    confidence: {conf:.3f}  (F1 on {len(pos_records)}+ vs {len(neg_records)}-)")
        print(f"    loo_f1: {loo:.3f}" if loo is not None
              else "    loo_f1: None (<3 positives — consistency check only)")

        # Write layer3 into each positive circuit's JSON
        for k in pos_keys:
            rec, jp = _all_records[k]
            feats = result["features"][0] if result["features"] else {}
            add_ilp_architectural(
                json_path=jp,
                depth_mean=feats.get("depth_mean", 0),
                depth_min=feats.get("depth_min", 0),
                depth_max=feats.get("depth_max", 0),
                component_ratio=feats.get("component_ratio_str", ""),
                num_attn_heads=feats.get("num_attn", 0),
                num_mlp_blocks=feats.get("num_mlp", 0),
                motifs=result.get("motifs", []),
                tau_arch_text=clause,
                tau_arch_confidence=conf,
                ilp_signature=result.get("ilp_signature", {}),
                exact=feats.get("exact", {}),
                loo_f1=loo,
            )
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback; traceback.print_exc()


# ── Distance report ───────────────────────────────────────────────────────

from trace.mechint.pipeline import export_distance_report

if len(arch_results) >= 2:
    print(f"\n{'='*68}")
    print("Distance Report (task-level, using first split per task)")
    print(f"{'='*68}")

    export_distance_report(
        arch_results,
        os.path.join(JSON_OUTPUT_DIR, "distance_report.json"),
    )


# ── Subsumption matrix ───────────────────────────────────────────────────

from trace.mechint.pipeline import compute_subsumption_matrix

# Use first split per task for subsumption
_first_split_paths = {}
for tn in sorted(tasks_with_circuits):
    key = f"{tn.lower()}_s0"
    if key in _json_paths_all:
        _first_split_paths[tn] = _json_paths_all[key]

if len(_first_split_paths) >= 2:
    print(f"\n{'='*68}")
    print("Subsumption Matrix")
    print(f"{'='*68}")

    compute_subsumption_matrix(
        _first_split_paths,
        os.path.join(JSON_OUTPUT_DIR, "subsumption_matrix.json"),
    )


# ── Baselines on the expanded circuit set ─────────────────────────────────

print(f"\n{'='*68}")
print("Baselines — expanded circuit set")
print(f"{'='*68}")

from trace.mechint.baselines_structural import full_baseline_comparison

# Load all circuit records for baselines
baseline_records = {}
baseline_families = {}
for key, (rec, jp) in _all_records.items():
    baseline_records[key] = rec
    # Extract task name from key (e.g., "location_s0" -> "Location")
    task_name = key.rsplit("_s", 1)[0].title()
    # Handle multi-word tasks
    for t in ALL_TASKS:
        if t["task"].lower() == key.rsplit("_s", 1)[0].lower():
            baseline_families[key] = t["task_family"]
            break

# Load ILP distance matrix if available
ilp_matrix = None
dr_path = os.path.join(JSON_OUTPUT_DIR, "distance_report.json")
if os.path.isfile(dr_path):
    with open(dr_path) as f:
        dr = json.load(f)
    ilp_matrix = dr.get("architectural_matrix", dr.get("semantic_matrix", {}))

try:
    report = full_baseline_comparison(
        baseline_records, baseline_families,
        ilp_distance_matrix=ilp_matrix,
        wl_iterations=3,
    )

    # Print key results
    rf = report["classification"]["random_forest"]
    stump = report["classification"]["decision_stump"]

    print(f"\n  Random Forest LOO: {rf['accuracy']:.2f} "
          f"({rf['correct']}/{rf['total']})")
    print(f"  Decision Stump:    {stump['accuracy']:.2f} "
          f"({stump['correct']}/{stump['total']})")

    sep = report.get("separation_scores", {})
    if sep:
        print(f"\n  Separation scores:")
        for method, score in sorted(sep.items()):
            print(f"    {method:>20}: {score:.3f}x")

    # Save
    import numpy as np

    def _serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    out_path = os.path.join(BASELINES_DIR, "baseline_comparison.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=_serialize)
    print(f"\n  Baseline report: {out_path}")

except Exception as e:
    print(f"  Baseline error: {e}")
    import traceback; traceback.print_exc()


# ── CFS clustering test (should-do #6) ───────────────────────────────────

from trace.mechint.pipeline import cfs_clustering_test

# Build task families for all split keys
_clustering_families = {}
for key in _json_paths_all:
    task_base = key.rsplit("_s", 1)[0].lower()
    for t in ALL_TASKS:
        if t["task"].lower() == task_base:
            _clustering_families[key] = t["task_family"]
            break

if len(_json_paths_all) >= 3 and not SKIP_CFS:
    print(f"\n{'='*68}")
    print("CFS Clustering Test (expanded circuit set)")
    print(f"{'='*68}")

    try:
        cfs_clustering_test(
            cfs_dir=CFS_OUTPUT_DIR,
            results_dir=JSON_OUTPUT_DIR,
            task_families=_clustering_families,
            out_path=os.path.join(JSON_OUTPUT_DIR, "cfs_clustering.json"),
        )
    except Exception as e:
        print(f"  CFS clustering error: {e}")


# ── δ_ILP sensitivity sweep (should-do #7) ───────────────────────────────

from trace.mechint.baselines_structural import delta_ilp_sensitivity

if ilp_matrix and len(ilp_matrix) >= 3:
    print(f"\n{'='*68}")
    print("delta_ILP Sensitivity Sweep")
    print(f"{'='*68}")

    # Task families for the task-level distance matrix
    _ilp_families = {}
    for t in ALL_TASKS:
        _ilp_families[t["task"].lower()] = t["task_family"]

    try:
        sweep = delta_ilp_sensitivity(ilp_matrix, _ilp_families)

        print(f"  Pairs: {sweep['n_pairs']} "
              f"({sweep['n_same_family']} same-family, "
              f"{sweep['n_cross_family']} cross-family)")
        print(f"\n  {'d_ILP':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Acc':>8} "
              f"{'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
        for r in sweep["thresholds"]:
            marker = " <-- best" if r["delta"] == sweep["best_threshold"] else ""
            print(f"  {r['delta']:8.2f} {r['precision']:8.4f} {r['recall']:8.4f} "
                  f"{r['f1']:8.4f} {r['accuracy']:8.4f} "
                  f"{r['tp']:4} {r['fp']:4} {r['tn']:4} {r['fn']:4}{marker}")
        print(f"\n  Best threshold: d_ILP = {sweep['best_threshold']} "
              f"(F1 = {sweep['best_f1']})")

        sweep_path = os.path.join(BASELINES_DIR, "delta_ilp_sensitivity.json")
        with open(sweep_path, "w") as f:
            json.dump(sweep, f, indent=2)
        print(f"  Saved: {sweep_path}")
    except Exception as e:
        print(f"  Sensitivity sweep error: {e}")


# ── Random circuit baselines (should-do #8) ──────────────────────────────

from trace.mechint.baselines import random_circuit

print(f"\n{'='*68}")
print("Random Circuit Baselines")
print(f"{'='*68}")

import numpy as np
from trace.mechint.baselines_structural import extract_structural_features

N_RANDOM = 10
random_baseline_results = {}

for tn in sorted(tasks_with_circuits):
    key = f"{tn.lower()}_s0"
    jp = _json_paths_all.get(key)
    if not jp:
        continue

    print(f"\n  {tn}: generating {N_RANDOM} size-matched random circuits...")
    try:
        with open(jp) as f:
            rec = json.load(f)
        real_metrics = rec.get("layer1", {}).get("metrics", {})
        real_faith = real_metrics.get("faithfulness", "N/A")
        real_delta_t = real_metrics.get("DeltaT", "N/A")
        n_nodes = real_metrics.get("num_nodes_in_circuit", 10)
        n_edges = real_metrics.get("num_edges_in_circuit", 50)

        random_feats = []
        for seed_i in range(N_RANDOM):
            rg = random_circuit(model, n_nodes, n_edges, seed=42 + seed_i)

            # Build a minimal results dict for feature extraction
            r_nodes = [
                {"name": n,
                 "type": "attn" if (n.startswith("a") and ".h" in n) else "mlp",
                 "layer": int(n.split(".")[0].replace("a", "").replace("m", "") or 0)}
                for n in sorted(rg.nodes)
                # only attention/MLP components carry a layer index;
                # 'input'/'logits' pseudo-nodes are not circuit components
                if rg.nodes[n].in_graph
                and n.split(".")[0].lstrip("am").isdigit()
            ]
            r_edges = [
                {"source": e.split("->")[0].strip(),
                 "target": e.split("->")[1].strip(),
                 "name": e, "score": rg.edges[e].score}
                for e in sorted(rg.edges)
                if rg.edges[e].in_graph
            ]
            fake_rec = {
                "layer1": {"circuit": {"nodes": r_nodes, "edges": r_edges,
                                        "n_layers": model.cfg.n_layers}},
                "layer3": {"motifs": []},
            }
            random_feats.append(extract_structural_features(fake_rec))

        real_feats = extract_structural_features(rec)
        real_attn = real_feats["component_ratio_attn"]
        rand_attns = [rf["component_ratio_attn"] for rf in random_feats]
        real_density = real_feats["density"]
        rand_densities = [rf["density"] for rf in random_feats]

        random_baseline_results[tn] = {
            "real_faith": real_faith,
            "real_delta_t": real_delta_t,
            "real_attn_ratio": round(real_attn, 4),
            "rand_attn_ratio_mean": round(float(np.mean(rand_attns)), 4),
            "rand_attn_ratio_std": round(float(np.std(rand_attns)), 4),
            "real_density": round(real_density, 4),
            "rand_density_mean": round(float(np.mean(rand_densities)), 4),
            "n_random": N_RANDOM,
        }

        print(f"    Real:   attn_ratio={real_attn:.3f}  density={real_density:.4f}  "
              f"faith={real_faith}  dT={real_delta_t}")
        print(f"    Random: attn_ratio={np.mean(rand_attns):.3f}"
              f"+/-{np.std(rand_attns):.3f}  "
              f"density={np.mean(rand_densities):.4f}"
              f"+/-{np.std(rand_densities):.4f}")
    except Exception as e:
        print(f"    ERROR: {e}")

if random_baseline_results:
    rand_path = os.path.join(BASELINES_DIR, "random_circuit_baselines.json")
    with open(rand_path, "w") as f:
        json.dump(random_baseline_results, f, indent=2)
    print(f"\n  Random baselines saved: {rand_path}")


# ── Summary ───────────────────────────────────────────────────────────────

print(f"\n{'='*68}")
print("DONE — Multi-circuit pipeline")
print(f"{'='*68}")
print(f"  Model:    {BASE_MODEL} ({MODEL_SHORT})")
print(f"  Tasks:    {len(tasks_with_circuits)}")
print(f"  Circuits: {total_circuits} ({N_SPLITS} splits/task)")
print(f"  ILP:      {len(arch_results)} learned signatures")
print(f"  Output:   {MODEL_DIR}/")

# Print per-task ILP results
if arch_results:
    print(f"\n  ILP Results:")
    print(f"    {'Task':14} {'Conf':>6} {'Cmplx':>6} {'Clause':50}")
    print(f"    {'-'*14} {'-'*6} {'-'*6} {'-'*50}")
    for task, res in sorted(arch_results.items()):
        conf = res.get("confidence", 0)
        sig = res.get("ilp_signature", {})
        cmplx = sig.get("complexity", 0)
        clause = res.get("clause_text", "")[:50]
        print(f"    {task:14} {conf:6.3f} {cmplx:6.1f} {clause}")
