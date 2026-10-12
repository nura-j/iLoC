#!/usr/bin/env bash
#
# run_all.sh — Full COMPASS pipeline: splits, multi-circuit discovery,
#              baselines, and comparison across all three models.
#
# Usage:
#   ./run_all.sh                  # full pipeline (all 3 models, k=3 splits)
#   ./run_all.sh --k 1            # Strategy A only (12 tasks, 1 circuit each)
#   ./run_all.sh --model pythia1b # single model
#
# Prerequisites:
#   pip install scikit-learn scipy transformer_lens transformers rdflib
#
set -euo pipefail

K="${K:-3}"
MODEL=""
SKIP_EAP=""

# ── Parse args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --k)        K="$2"; shift 2 ;;
        --model)    MODEL="$2"; shift 2 ;;
        --skip-eap) SKIP_EAP="--skip-eap"; shift ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Model registry: short_name -> HF model ID
declare -A HF_MODELS=(
    [pythia14m]="EleutherAI/pythia-14m"
    [pythia1b]="EleutherAI/pythia-1b"
    [Llama3.21B]="meta-llama/Llama-3.2-1B"
)

if [[ -n "$MODEL" ]]; then
    MODELS=("$MODEL")
else
    MODELS=("pythia14m" "pythia1b" "Llama3.21B")
fi

echo "========================================================================"
echo "  COMPASS Full Pipeline"
echo "  Models: ${MODELS[*]}"
echo "  Splits: k=${K} per task"
echo "========================================================================"

# ── Step 0: Verify dependencies ──────────────────────────────────────────

echo ""
echo "[Step 0] Checking dependencies..."
python -c "import sklearn, scipy, transformer_lens, transformers, rdflib" 2>/dev/null \
    || { echo "ERROR: Missing dependencies. Run:"; \
         echo "  pip install scikit-learn scipy transformer_lens transformers rdflib"; \
         exit 1; }
echo "  OK"

# ── Step 1: Create dataset splits (no GPU, instant) ──────────────────────

echo ""
echo "========================================================================"
echo "[Step 1] Creating dataset splits (k=${K})"
echo "========================================================================"

for m in "${MODELS[@]}"; do
    echo ""
    python split_datasets.py --model "$m" --k "$K"
done

# ── Step 2: Multi-circuit discovery + ILP + baselines (needs model) ──────

echo ""
echo "========================================================================"
echo "[Step 2] Multi-circuit pipeline (EAP + CFS + ILP + baselines)"
echo "========================================================================"

for m in "${MODELS[@]}"; do
    hf_id="${HF_MODELS[$m]}"
    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "  Model: ${m} (${hf_id})"
    echo "────────────────────────────────────────────────────────────────────"
    python run_experiment_multi.py \
        --model "$hf_id" \
        --k "$K" \
        --skip-cfs \
        $SKIP_EAP
done

# ── Step 3: Run baselines on the ORIGINAL 5-task circuits (for comparison)

echo ""
echo "========================================================================"
echo "[Step 3] Baselines on original 5-task circuits"
echo "========================================================================"

for m in "${MODELS[@]}"; do
    # Only run if original results exist
    if [[ -d "./analysis/${m}/results" ]]; then
        echo ""
        python run_baselines.py --model "$m" --save
    fi
done

# ── Step 4: Summary ──────────────────────────────────────────────────────

echo ""
echo "========================================================================"
echo "  DONE"
echo "========================================================================"
echo ""
echo "  Results:"
for m in "${MODELS[@]}"; do
    orig="./analysis/${m}/results"
    multi="./analysis/${m}_multi/results"
    base_orig="./analysis/${m}/baselines"
    base_multi="./analysis/${m}_multi/baselines"

    echo ""
    echo "  ${m}:"
    [[ -d "$orig" ]]       && echo "    Original (5 tasks):   ${orig}/"
    [[ -d "$multi" ]]      && echo "    Multi (12×${K}):       ${multi}/"
    [[ -d "$base_orig" ]]  && echo "    Baselines (original): ${base_orig}/"
    [[ -d "$base_multi" ]] && echo "    Baselines (multi):    ${base_multi}/"
done

echo ""
echo "  Key files to check:"
echo "    analysis/{model}_multi/results/distance_report.json"
echo "    analysis/{model}_multi/baselines/baseline_comparison.json"
echo "    analysis/{model}/baselines/baseline_comparison.json"
echo ""
echo "  Compare ILP F1 between original (should be 1.0) and multi"
echo "  (should be <1.0 if splits produce structurally different circuits)."
