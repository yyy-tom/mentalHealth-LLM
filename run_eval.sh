#!/bin/bash
# -----------------------------------------------------------------------
# Mental Health LLM — Full Evaluation Script
#
# For running on a rented GPU machine (RTX 5090 / A100 / H100).
# No SLURM needed — just run directly:
#
#   bash run_eval.sh                          # default: qwen-ft only
#   bash run_eval.sh qwen-ft gemma-ft         # specific models
#   bash run_eval.sh all                      # all three models
#
# Prerequisites on the rented machine:
#   1. Clone the repo and cd into it
#   2. Install deps:  pip install -r requirements.txt
#   3. Set judge key: export DEEPSEEK_API_KEY=sk-...
#   4. Ensure model weights are at paths in config.yaml model_registry
#
# Estimated wall time (RTX 5090, 32GB VRAM):
#   42 cases x 8 passes x ~4 turns x ~3s/turn + judge API = ~5-6h per model
#   With 4-bit quant: 7B uses ~5GB VRAM, 9B uses ~6GB — plenty of headroom
# -----------------------------------------------------------------------

set -euo pipefail

# -----------------------------------------------------------------------
# Resolve python binary (some systems only have python3)
# -----------------------------------------------------------------------
if command -v python &>/dev/null; then
    PYTHON=python
elif command -v python3 &>/dev/null; then
    PYTHON=python3
else
    echo "ERROR: neither python nor python3 found in PATH"
    exit 1
fi
echo "Using: $($PYTHON --version) at $(which $PYTHON)"

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
CONFIG="evaluation/harness/config.yaml"
OUTPUT_DIR="evaluation/results/full_eval"
TEST_SUITE="all"
ALL_MODELS=("qwen-ft" "gemma-ft" "mistral-ft")

# Parse args
if [[ $# -gt 0 && "$1" == "all" ]]; then
    MODELS=("${ALL_MODELS[@]}")
elif [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=("qwen-ft")
fi

# -----------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------
echo "======================================================================"
echo "Mental Health LLM — Full Evaluation"
echo "Host        : $(hostname)"
echo "Models      : ${MODELS[*]}"
echo "Start time  : $(date)"
echo "======================================================================"

# GPU check
echo ""
echo "--- GPU info ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. CUDA may not be available."
fi
echo ""

# Judge API key
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is not set."
    echo "Run: export DEEPSEEK_API_KEY=sk-..."
    exit 1
fi

# Test cases exist
CASE_COUNT=$(find evaluation/cases -name '*.json' 2>/dev/null | wc -l)
if [[ "$CASE_COUNT" -eq 0 ]]; then
    echo "ERROR: No test cases found in evaluation/cases/"
    exit 1
fi
echo "Found ${CASE_COUNT} case files in evaluation/cases/"

# Create output dirs
mkdir -p "${OUTPUT_DIR}" logs

# -----------------------------------------------------------------------
# Router benchmark (CPU only, fast — run first)
# -----------------------------------------------------------------------
echo ""
echo "--- Router benchmark ---"
$PYTHON scripts/benchmark_router.py \
    --output "${OUTPUT_DIR}/router_benchmark.json" \
    2>&1 | tee logs/router_benchmark.log \
    || echo "WARNING: router benchmark failed, continuing..."

# -----------------------------------------------------------------------
# Main evaluation — single model load per model
# -----------------------------------------------------------------------
echo ""
echo "--- Full evaluation (single model load) ---"
$PYTHON scripts/evaluation/run_full_eval_single_load.py \
    --model "${MODELS[@]}" \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --test-suite "${TEST_SUITE}" \
    2>&1 | tee logs/full_eval.log

# -----------------------------------------------------------------------
# Unit tests (CPU only, after GPU freed)
# -----------------------------------------------------------------------
echo ""
echo "--- Unit tests ---"
$PYTHON -m pytest tests/test_tools.py tests/test_orchestration.py -q --tb=short \
    2>&1 | tee logs/unit_tests.log \
    || echo "WARNING: some unit tests failed"

# -----------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "Complete    : $(date)"
echo "Results in  : ${OUTPUT_DIR}/"
echo "Logs in     : logs/"
echo "======================================================================"
