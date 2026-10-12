"""
run_experiment.py
=================
COMPASS pipeline — circuit discovery, CFS, ILP_arch, transfer.

    python run_experiment.py              # full pipeline
    python run_experiment.py --skip-eap   # skip EAP, load cached circuits

Change BASE_MODEL to run for a different model. Results are stored per-model
under analysis/{model}/.
"""

# ── Config ────────────────────────────────────────────────────────────────
import argparse
import os
import json

_parser = argparse.ArgumentParser(description="COMPASS full pipeline")
_parser.add_argument("--skip-eap", action="store_true",
                     help="Skip EAP; load existing circuits from JSON")
_args, _ = _parser.parse_known_args()
SKIP_EAP = _args.skip_eap

BASE_MODEL = "meta-llama/Llama-3.2-1B"
TOP_K      = 200

# Model short name: "EleutherAI/pythia-1b" → "pythia1b"
MODEL_SHORT = BASE_MODEL.split("/")[-1].replace("-", "").replace("_", "")

# Task families
TASK_FAMILY_SEMROLES = "SemanticRoles"
TASK_FAMILY_IOI      = "IOI"
TASK_FAMILY_GT       = "GreaterThan"

# Per-model dataset paths
_DATASET_REGISTRY = {
    "pythia1b": {
        "Location":    "./semantic_role_generation/"
                       "semroles_v1_filtered_1b/Location/both_correct/prompts.csv",
        "Time":        "./semantic_role_generation/"
                       "semroles_v1_filtered_1b/Time/both_correct/prompts.csv",
        "Path":        "./semantic_role_generation/"
                       "semroles_v1_filtered_1b/Path/both_correct/prompts.csv",
        "IOI":         "./data/ioi_pythia_1b_filtered.csv",
        "GreaterThan": "./data/greater_than_filtered_pythia1b/both_correct/prompts.csv",
    },
    "pythia14m": {
        "Location":    "./semantic_role_generation/"
                       "semroles_v1_filtered_14m/Location/both_correct/prompts.csv",
        "Time":        "./semantic_role_generation/"
                       "semroles_v1_filtered_14m/Time/both_correct/prompts.csv",
        "Path":        "./semantic_role_generation/"
                       "semroles_v1_filtered_14m/Path/both_correct/prompts.csv",
        "IOI":         "./data/ioi_pythia_14m_filtered.csv",
        "GreaterThan": "./data/greater_than_filtered_pythia14m/both_correct/prompts.csv",
    },
    "Llama3.21B": {
        "Location": "./semantic_role_generation/"
                    "semroles_v1_Llama_3_2_1B_filtered/Location/both_correct/prompts.csv",
        "Time": "./semantic_role_generation/"
                    "semroles_v1_Llama_3_2_1B_filtered/Time/both_correct/prompts.csv",
        "Path": "./semantic_role_generation/"
                    "semroles_v1_Llama_3_2_1B_filtered/Path/both_correct/prompts.csv",
        "IOI": "./data/ioi_llama_1b.csv",
        "GreaterThan": "./data/greater_than_filtered_llama1b/both_correct/prompts.csv",
    }
}
DATASETS = _DATASET_REGISTRY.get(MODEL_SHORT, _DATASET_REGISTRY["pythia1b"])

# Task definitions
_TASKS = [
    {"task": "IOI",         "task_family": TASK_FAMILY_IOI,      "role": "IOI",
     "operation_type": "selection",  "logical_category": "PREDICATE"},
    {"task": "Location",    "task_family": TASK_FAMILY_SEMROLES, "role": "Location",
     "operation_type": "binding",    "logical_category": "ROLE"},
    {"task": "Time",        "task_family": TASK_FAMILY_SEMROLES, "role": "Time",
     "operation_type": "binding",    "logical_category": "ROLE"},
    {"task": "Path",        "task_family": TASK_FAMILY_SEMROLES, "role": "Path",
     "operation_type": "binding",    "logical_category": "ROLE"},
    {"task": "GreaterThan", "task_family": TASK_FAMILY_GT,       "role": "GreaterThan",
     "operation_type": "comparison", "logical_category": "OPERATOR"},
]

# Output directories (per-model data, shared figures)
ANALYSIS_ROOT    = "./analysis"
MODEL_DIR        = f"{ANALYSIS_ROOT}/{MODEL_SHORT}"
OWL_OUTPUT_DIR   = f"{MODEL_DIR}/circuits"
JSON_OUTPUT_DIR  = f"{MODEL_DIR}/results"
CFS_OUTPUT_DIR   = f"{MODEL_DIR}/cfs"
ILP_DIR          = f"{MODEL_DIR}/ilp"
FIGURES_DIR      = f"{ANALYSIS_ROOT}/figures"
CIRCUIT_FIGS_DIR = f"{FIGURES_DIR}/circuits/{MODEL_SHORT}"
L1_FIGS_DIR      = f"{FIGURES_DIR}/layer1/{MODEL_SHORT}"
L2_FIGS_DIR      = f"{FIGURES_DIR}/layer2/{MODEL_SHORT}"
L3_FIGS_DIR      = f"{FIGURES_DIR}/layer3/{MODEL_SHORT}"
TRANSFER_FIGS_DIR= f"{FIGURES_DIR}/transfer"
TABLES_DIR       = f"{ANALYSIS_ROOT}/tables"
LAYERS_DIR       = f"{FIGURES_DIR}/unified/{MODEL_SHORT}"

for d in [OWL_OUTPUT_DIR, JSON_OUTPUT_DIR, CFS_OUTPUT_DIR, ILP_DIR,
          CIRCUIT_FIGS_DIR, L1_FIGS_DIR, L2_FIGS_DIR, L3_FIGS_DIR,
          TRANSFER_FIGS_DIR, TABLES_DIR, LAYERS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Config: {BASE_MODEL} ({MODEL_SHORT}), output -> {MODEL_DIR}/")


# ── Load model ────────────────────────────────────────────────────────────
from transformers import AutoTokenizer
import transformer_lens as tl

print(f"\nLoading {BASE_MODEL}...")
model = tl.HookedTransformer.from_pretrained(BASE_MODEL, device='cpu')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model.cfg.use_attn_result = True
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True
print(f"Model loaded ({model.cfg.n_layers}L, {model.cfg.n_heads}H, d={model.cfg.d_model})")


# ── Circuit discovery (or load cached) ────────────────────────────────────
from trace.mechint import analyze_and_export, reconstruct_graph

_results, _graphs, _owls = {}, {}, {}

for _t in _TASKS:
    tn = _t["task"]
    jp = os.path.join(JSON_OUTPUT_DIR, f"{tn.lower()}_results.json")

    if os.path.isfile(jp):
        print(f"\n[CACHED] {tn}")
        _graphs[tn] = reconstruct_graph(model, jp)
        with open(jp) as f:
            rec = json.load(f)
        _results[tn] = rec.get("layer1", {}).get("metrics", {})
        _results[tn]["method"] = rec.get("layer0", {}).get("attribution_method")
        _results[tn]["top_k"] = rec.get("layer0", {}).get("top_k")
        _owls[tn] = rec.get("layer0", {}).get("owl_path") or rec.get("meta", {}).get("owl_path")
        print(f"  Faith={_results[tn].get('faithfulness', 'N/A')}")
    else:
        if SKIP_EAP:
            print(f"\n[SKIP] {tn}: no cached result")
            continue
        print(f"\n[NEW] {tn}")
        res, graph, owl = analyze_and_export(
            model=model, cnp_dataset_path=DATASETS[tn],
            task_family=_t["task_family"], role=_t["role"],
            model_name=MODEL_SHORT, tokenizer=tokenizer,
            top_k=TOP_K, ig_steps=3, compute_validation=False,
            output_dir=OWL_OUTPUT_DIR,
            operation_type=_t["operation_type"],
            logical_category=_t["logical_category"],
        )
        _results[tn], _graphs[tn], _owls[tn] = res, graph, owl
        print(f"  Faith={res['faithfulness']:.4f}, nodes={res['num_nodes_in_circuit']}")

# Filter to active tasks
_TASKS = [t for t in _TASKS if t["task"] in _results]
print(f"\nActive tasks: {[t['task'] for t in _TASKS]}")


# ── Circuit topology PNGs ─────────────────────────────────────────────────
for tn, g in _graphs.items():
    g.to_image(os.path.join(CIRCUIT_FIGS_DIR, f"{tn.lower()}_circuit.png"))


# ── JSON export ───────────────────────────────────────────────────────────
from trace.mechint.storage.results_export import export_task_json, export_comparison_table

_json_paths, _records = {}, {}
for _t in _TASKS:
    tn = _t["task"]
    _json_paths[tn] = export_task_json(
        task=tn, task_family=_t["task_family"], role=_t["role"],
        model_name=MODEL_SHORT, results=_results[tn], graph=_graphs[tn],
        owl_path=_owls[tn], out_dir=JSON_OUTPUT_DIR,
        operation_type=_t["operation_type"],
        logical_category=_t["logical_category"],
    )
for tn, jp in _json_paths.items():
    with open(jp) as f:
        _records[tn] = json.load(f)


# ── Layer 2: CFS ──────────────────────────────────────────────────────────
from trace.mechint import run_cfs_pipeline

print("\n" + "=" * 68)
print("Layer 2 — CFS")
print("=" * 68)

_task_graphs = {
    t["task"]: (_graphs[t["task"]], _owls[t["task"]],
                _json_paths[t["task"]], DATASETS[t["task"]])
    for t in _TASKS
}
cfs_all, ll_all = run_cfs_pipeline(
    model, tokenizer, _task_graphs, CFS_OUTPUT_DIR, L2_FIGS_DIR,
    task_defs=_TASKS,
)


# ── Layer 3: ILP_arch ─────────────────────────────────────────────────────
from trace.mechint import run_ilp_arch_pipeline, export_distance_report

print("\n" + "=" * 68)
print("Layer 3 — ILP_arch")
print("=" * 68)

arch_results = run_ilp_arch_pipeline(_json_paths, ILP_DIR)

# Distance report
export_distance_report(
    arch_results,
    os.path.join(JSON_OUTPUT_DIR, "distance_report.json"),
)


# ── Layer 3 figures ───────────────────────────────────────────────────────
from plot_layer3_architectural import (
    plot_clause_card, plot_motif_heatmap,
    plot_distance_matrix_signature, plot_distance_matrix_circuit,
    plot_distance_comparison, plot_component_ratios,
    plot_summary_table as plot_l3_summary_table,
    export_summary_latex as export_l3_summary_latex,
    _load_all_arch,
)

print("\n" + "=" * 68)
print("Layer 3 — Figures")
print("=" * 68)

archs = _load_all_arch()
if archs:
    for a in archs:
        plot_clause_card(a, os.path.join(L3_FIGS_DIR, f"{a['role']}_clause.png"))
    plot_motif_heatmap(archs, os.path.join(L3_FIGS_DIR, "motif_heatmap.png"))
    plot_distance_matrix_signature(os.path.join(L3_FIGS_DIR, "distance_signature.png"))
    plot_distance_matrix_circuit(os.path.join(L3_FIGS_DIR, "distance_circuit.png"))
    plot_distance_comparison(os.path.join(L3_FIGS_DIR, "distance_comparison.png"))
    plot_component_ratios(archs, os.path.join(L3_FIGS_DIR, "component_ratios.png"))
    plot_l3_summary_table(archs, os.path.join(L3_FIGS_DIR, "summary_table.png"))
    export_l3_summary_latex(archs, os.path.join(TABLES_DIR, "layer3_summary.tex"))


# ── Comparison table ──────────────────────────────────────────────────────
_comp_records = {}
for t in _TASKS:
    with open(_json_paths[t["task"]]) as f:
        _comp_records[t["task"]] = json.load(f)

comparison_path = export_comparison_table(_comp_records, JSON_OUTPUT_DIR)
print(f"\nComparison table: {comparison_path}")


# ── Layer 1 figures ───────────────────────────────────────────────────────
from plot_layer1_circuits import (
    plot_single_circuit, plot_component_counts, plot_depth_profile,
    plot_faithfulness, plot_summary_table as plot_l1_summary_table,
    export_summary_latex as export_l1_summary_latex,
)
from test_layer1_structural import (
    _discover_ttl_files, _parse_ttl_id, _load_rdf,
    _find_circuit_uri, extract_layer1,
)

print("\n" + "=" * 68)
print("Layer 1 — Figures")
print("=" * 68)

_l1_circuits = []
for ttl_path in _discover_ttl_files():
    info = _parse_ttl_id(ttl_path)
    rdf = _load_rdf(ttl_path)
    circuit_uri = _find_circuit_uri(rdf)
    l1 = extract_layer1(rdf, circuit_uri)
    _l1_circuits.append((info, l1))
    plot_single_circuit(info, l1, os.path.join(L1_FIGS_DIR, f"{info['role_raw']}_topology.png"))

if _l1_circuits:
    plot_component_counts(_l1_circuits, os.path.join(L1_FIGS_DIR, "component_counts.png"))
    plot_depth_profile(_l1_circuits, os.path.join(L1_FIGS_DIR, "depth_profile.png"))
    plot_faithfulness(_l1_circuits, os.path.join(L1_FIGS_DIR, "faithfulness.png"))
    plot_l1_summary_table(_l1_circuits, os.path.join(L1_FIGS_DIR, "summary_table.png"))
    export_l1_summary_latex(_l1_circuits, os.path.join(TABLES_DIR, "layer1_summary.tex"))


# ── Unified layer figures ─────────────────────────────────────────────────
from visualise_layers import (
    load_circuit_data, plot_layer0, plot_layer1, plot_layer2, plot_layer3,
    plot_summary_table,
)

viz_data = {t["task"]: load_circuit_data(t["task"], OWL_OUTPUT_DIR, JSON_OUTPUT_DIR)
            for t in _TASKS}
plot_layer0(viz_data, LAYERS_DIR)
plot_layer1(viz_data, LAYERS_DIR)
plot_layer2(viz_data, LAYERS_DIR)
plot_layer3(viz_data, LAYERS_DIR)
plot_summary_table(viz_data, LAYERS_DIR)
print(f"\nUnified figures -> {LAYERS_DIR}/")


# ── Transfer Engine ───────────────────────────────────────────────────────
from trace.mechint import TransferEngine, compute_delta_t_preservation

print("\n" + "=" * 68)
print("Transfer Engine (§3.7)")
print("=" * 68)

engine = TransferEngine(
    registry_dir=ANALYSIS_ROOT,
    delta_ilp=0.30, delta_arch=0.40,
    theta_behav=0.50, theta_causal=0.10,
    struct_depth_tol=0.15, struct_size_tol=0.15,
)
print(f"  Registry: {engine.reload_registry()} records from {ANALYSIS_ROOT}/")

_registry_models = sorted(set(
    r.get("meta", {}).get("model_name", "unknown") for r in engine._registry
))
print(f"  Models: {_registry_models}")

# Within-model transfer
print("\n  WITHIN-MODEL TRANSFER")
transfer_reports = {}
for t in _TASKS:
    jp = _json_paths[t["task"]]
    report = engine.find_transfer_candidates(jp, target_model_name=MODEL_SHORT)
    transfer_reports[t["task"].lower()] = report
    top = [c for c in report.candidates if c.struct_pass][:3]
    for c in top:
        d5 = "PASS" if c.def5_pass else ("?" if c.def5_pass is None else "FAIL")
        print(f"    [{d5}] {t['task']} -> {c.task_name}  d={c.def5_distance or 'N/A'}")

# Cross-scale transfer
if len(_registry_models) > 1:
    print("\n  CROSS-SCALE TRANSFER")
    for t in _TASKS:
        for tgt in _registry_models:
            if tgt == MODEL_SHORT:
                continue
            report = engine.find_transfer_candidates(_json_paths[t["task"]], target_model_name=tgt)
            print(f"    {t['task']}: {MODEL_SHORT} -> {tgt}: "
                  f"{len([c for c in report.candidates if c.struct_pass])} struct, "
                  f"{len([c for c in report.candidates if c.def5_pass])} def5")

    # ΔT preservation
    print("\n  ΔT PRESERVATION")
    compute_delta_t_preservation(
        engine._registry,
        [t["task"] for t in _TASKS],
        os.path.join(JSON_OUTPUT_DIR, "transfer_delta_t.json"),
    )



# ── Analysis: Dual-pool, Subsumption, CFS clustering ─────────────────
from trace.mechint import analyse_dual_pool, compute_subsumption_matrix, cfs_clustering_test

print("\n" + "=" * 68)
print("Analysis — Dual-pool, Subsumption, CFS clustering")
print("=" * 68)

# Dual-pool analysis
print("\n  DUAL-POOL ANALYSIS")
pool_results = analyse_dual_pool(_json_paths)
for r in pool_results:
    print(f"    {r['task']:12} Pool {r['pool']} ({r['pool_desc']})")

# Subsumption matrix (strict + relaxed)
print("\n  SUBSUMPTION MATRIX")
compute_subsumption_matrix(
    _json_paths,
    os.path.join(JSON_OUTPUT_DIR, "subsumption_matrix.json"),
)

# CFS clustering test
print("\n  CFS CLUSTERING TEST")
_task_families = {}
for t in _TASKS:
    _task_families[t["task"].lower()] = t["task_family"]
cfs_clustering_test(
    cfs_dir=CFS_OUTPUT_DIR,
    results_dir=JSON_OUTPUT_DIR,
    task_families=_task_families,
    out_path=os.path.join(JSON_OUTPUT_DIR, "cfs_clustering.json"),
)


# ── Done ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("DONE")
print("=" * 68)
print(f"  Model output -> {MODEL_DIR}/")
print(f"  Figures      -> {FIGURES_DIR}/")
for tn, jp in _json_paths.items():
    print(f"    {os.path.basename(jp)}")
