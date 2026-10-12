"""
recompute_cfs_profiles.py
=========================
Recompute CFS attention profiles with spaCy linguistic labelling
and task-specific labelling.

Loads each model once, recomputes both linguistic and task-specific
attention profiles for all its tasks, and updates the results JSONs
in-place (layer2.cfs.components) and writes task-specific profile JSONs.

Usage:
    conda run -n formal_trace python recompute_cfs_profiles.py --model pythia1b
    conda run -n formal_trace python recompute_cfs_profiles.py --model Llama3.21B
    conda run -n formal_trace python recompute_cfs_profiles.py  # all 3 models
"""

import argparse
import json
import os
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_MODEL_MAP = {
    "pythia14m":  "EleutherAI/pythia-14m",
    "pythia1b":   "EleutherAI/pythia-1b",
    "Llama3.21B": "meta-llama/Llama-3.2-1B",
}

_DATASET_REGISTRY = {
    "pythia1b": {
        "Location":    "./semantic_role_generation/semroles_v1_filtered_1b/Location/both_correct/prompts.csv",
        "Time":        "./semantic_role_generation/semroles_v1_filtered_1b/Time/both_correct/prompts.csv",
        "Path":        "./semantic_role_generation/semroles_v1_filtered_1b/Path/both_correct/prompts.csv",
        "IOI":         "./data/ioi_pythia_1b_filtered.csv",
        "GreaterThan": "./data/greater_than_filtered_pythia1b/both_correct/prompts.csv",
    },
    "pythia14m": {
        "Location":    "./semantic_role_generation/semroles_v1_filtered_14m/Location/both_correct/prompts.csv",
        "Time":        "./semantic_role_generation/semroles_v1_filtered_14m/Time/both_correct/prompts.csv",
        "Path":        "./semantic_role_generation/semroles_v1_filtered_14m/Path/both_correct/prompts.csv",
        "IOI":         "./data/ioi_pythia_14m_filtered.csv",
        "GreaterThan": "./data/greater_than_filtered_pythia14m/both_correct/prompts.csv",
    },
    "Llama3.21B": {
        "Location":    "./semantic_role_generation/semroles_v1_Llama_3_2_1B_filtered/Location/both_correct/prompts.csv",
        "Time":        "./semantic_role_generation/semroles_v1_Llama_3_2_1B_filtered/Time/both_correct/prompts.csv",
        "Path":        "./semantic_role_generation/semroles_v1_Llama_3_2_1B_filtered/Path/both_correct/prompts.csv",
        "IOI":         "./data/ioi_llama_1b.csv",
        "GreaterThan": "./data/greater_than_filtered_llama1b/both_correct/prompts.csv",
    },
}

_TASK_FAMILIES = {
    "IOI":         "IOI",
    "Location":    "SemanticRoles",
    "Time":        "SemanticRoles",
    "Path":        "SemanticRoles",
    "GreaterThan": "GreaterThan",
}

_TASKS = ["IOI", "Location", "Time", "Path", "GreaterThan"]

ANALYSIS_ROOT = "./analysis"
MAX_EXAMPLES = 50


# ---------------------------------------------------------------------------
# spaCy availability check
# ---------------------------------------------------------------------------

def _check_spacy():
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("[spaCy] Available — using dependency parsing for linguistic labels.")
        return True
    except ImportError:
        print("[spaCy] NOT installed. Run: pip install spacy && python -m spacy download en_core_web_sm")
        print("        Falling back to heuristic (function/entity/other only).")
        return False
    except OSError:
        print("[spaCy] Installed but en_core_web_sm model missing. Run: python -m spacy download en_core_web_sm")
        print("        Falling back to heuristic.")
        return False


# ---------------------------------------------------------------------------
# Main per-model recompute
# ---------------------------------------------------------------------------

def recompute_model(model_short: str, spacy_ok: bool):
    hf_name = _MODEL_MAP[model_short]
    results_dir = os.path.join(ANALYSIS_ROOT, model_short, "results")
    cfs_dir = os.path.join(ANALYSIS_ROOT, model_short, "cfs")
    os.makedirs(cfs_dir, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"Model: {model_short} ({hf_name})")
    print(f"{'='*68}")

    # Load model + tokenizer
    from transformers import AutoTokenizer
    import transformer_lens as tl

    print(f"Loading {hf_name}...")
    model = tl.HookedTransformer.from_pretrained(hf_name, device="cpu")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    print(f"Loaded ({model.cfg.n_layers}L, {model.cfg.n_heads}H)")

    from trace.mechint.pipeline import reconstruct_graph
    from trace.mechint.cfs.attention_profile import compute_attention_profiles
    from trace.mechint.cfs.role_labellers import get_labellers, LINGUISTIC_ROLES

    datasets = _DATASET_REGISTRY.get(model_short, _DATASET_REGISTRY["pythia1b"])

    for task_name in _TASKS:
        csv_path = datasets.get(task_name)
        json_path = os.path.join(results_dir, f"{task_name.lower()}_results.json")

        if not os.path.isfile(json_path):
            print(f"\n  [{task_name}] No results JSON — skipping.")
            continue
        if not csv_path or not os.path.isfile(csv_path):
            print(f"\n  [{task_name}] No prompt CSV at {csv_path} — skipping.")
            continue

        task_family = _TASK_FAMILIES[task_name]
        print(f"\n  [{task_name}] family={task_family}")

        # Load graph
        graph = reconstruct_graph(model, json_path)

        # Load examples + metadata
        df = pd.read_csv(csv_path).head(MAX_EXAMPLES)
        examples = []
        examples_meta = []
        for _, row in df.iterrows():
            examples.append({
                "clean":   str(row.get("clean",   row.get("prompt", ""))),
                "correct": str(row.get("target_word", row.get("answer", ""))),
            })
            examples_meta.append(row.to_dict())
        print(f"    {len(examples)} examples loaded.")

        # Get labellers
        ling_roles, task_roles, ling_labeller, task_labeller = get_labellers(task_family)

        # -- 1. Linguistic profiles (spaCy if available, heuristic fallback) --
        def _ling_lab(text, token_strs):
            return ling_labeller(text, token_strs, None)

        print("    Computing linguistic profiles...")
        ling_attn = compute_attention_profiles(
            model, graph, examples, tokenizer,
            epsilon_dla=0.05,
            token_roles=ling_roles,
            role_labeller=_ling_lab,
        )

        # -- 2. Task-specific profiles --
        task_profiles = {}
        if task_labeller and task_roles:
            print("    Computing task-specific profiles...")
            _meta_iter = iter(examples_meta)

            def _task_lab(text, token_strs):
                try:
                    meta = next(_meta_iter)
                except StopIteration:
                    meta = None
                return task_labeller(text, token_strs, meta)

            task_attn = compute_attention_profiles(
                model, graph, examples, tokenizer,
                epsilon_dla=0.05,
                token_roles=task_roles,
                role_labeller=_task_lab,
            )
            task_profiles = task_attn.profiles

            # Save task-specific profiles JSON
            task_json = {
                "task_family": task_family,
                "task_roles":  list(task_roles),
                "profiles": {
                    name: {r: round(v, 4) for r, v in prof.items()}
                    for name, prof in task_profiles.items()
                },
            }
            task_path = os.path.join(cfs_dir, f"{task_name.lower()}_task_profiles.json")
            with open(task_path, "w") as f:
                json.dump(task_json, f, indent=2)
            print(f"    Task profiles -> {task_path}")

        # -- 3. Patch results JSON in-place --
        with open(json_path) as f:
            record = json.load(f)

        if "layer2" not in record or "cfs" not in record.get("layer2", {}):
            print(f"    WARNING: no layer2.cfs in {json_path} — skipping patch.")
            continue

        cfs_section = record["layer2"]["cfs"]
        components = cfs_section.get("components", {})
        updated_attn = 0
        updated_task = 0
        fixed_mlp = 0

        for comp_name, comp_data in components.items():
            is_attn = comp_name.startswith("a") and ".h" in comp_name

            if is_attn and comp_name in ling_attn.profiles:
                # Update linguistic profile
                new_prof = ling_attn.profiles[comp_name]
                comp_data["attention_profile"] = {r: round(v, 4) for r, v in new_prof.items()}
                # Fix dominant_role from new profile
                total = sum(new_prof.values())
                if total > 0:
                    comp_data["dominant_role"] = max(new_prof, key=new_prof.get)
                else:
                    comp_data["dominant_role"] = None
                updated_attn += 1

                # Add task-specific profile if available
                if comp_name in task_profiles:
                    tp = task_profiles[comp_name]
                    comp_data["task_attention_profile"] = {r: round(v, 4) for r, v in tp.items()}
                    updated_task += 1
            else:
                # MLP or non-attn component — no meaningful attention profile
                # Fix the spurious "subject" dominant_role from argmax-of-zeros
                prof = comp_data.get("attention_profile", {})
                total = sum(prof.values())
                if total == 0.0:
                    comp_data["dominant_role"] = None
                    fixed_mlp += 1

        record["layer2"]["cfs"]["components"] = components

        with open(json_path, "w") as f:
            json.dump(record, f, indent=2)

        print(f"    Updated: {updated_attn} attn profiles, "
              f"{updated_task} task profiles, "
              f"{fixed_mlp} MLP dominant_role=null fixes.")
        print(f"    Saved -> {json_path}")

    # Explicitly free GPU/CPU memory
    del model
    import gc
    gc.collect()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=list(_MODEL_MAP.keys()),
        help="Which model to recompute (default: all)"
    )
    args = parser.parse_args()

    spacy_ok = _check_spacy()

    models = [args.model] if args.model else list(_MODEL_MAP.keys())
    for m in models:
        recompute_model(m, spacy_ok)

    print("\nDone.")
