#!/bin/bash
# -----------------------------------------------------------------------
# Multi-GPU Evaluation — splits test cases across available GPUs
#
# Usage:
#   bash run_eval_multi_gpu.sh qwen-ft           # one model, all GPUs
#   bash run_eval_multi_gpu.sh qwen-ft gemma-ft   # two models sequentially
#   NUM_GPUS=4 bash run_eval_multi_gpu.sh qwen-ft # limit to 4 GPUs
#
# Each GPU loads the model independently and evaluates a slice of cases.
# Results are merged at the end into a single output directory.
# -----------------------------------------------------------------------

set -euo pipefail

if command -v python &>/dev/null; then
    PYTHON=python
elif command -v python3 &>/dev/null; then
    PYTHON=python3
else
    echo "ERROR: python not found"
    exit 1
fi

CONFIG="evaluation/harness/config.yaml"
OUTPUT_BASE="evaluation/results/multi_gpu"
MERGED_BASE="evaluation/results/full_eval"
TEST_SUITE="all"
JUDGE_WORKERS=2

# Detect GPUs
if [[ -z "${NUM_GPUS:-}" ]]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
fi
echo "Using $NUM_GPUS GPUs"

# Count total cases
TOTAL_CASES=$($PYTHON -c "
import json, sys
sys.path.insert(0, '.')
from evaluation.harness.config import HarnessConfig
from evaluation.harness.runner import EvaluationHarness
config = HarnessConfig.from_yaml('$CONFIG')
harness = EvaluationHarness(config)
cases = harness._load_test_cases('$TEST_SUITE')
print(len(cases))
")
echo "Total cases: $TOTAL_CASES"

# Parse models
if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=("qwen-ft")
fi

mkdir -p "$OUTPUT_BASE" "$MERGED_BASE" logs

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Model: $MODEL — $NUM_GPUS GPUs — $TOTAL_CASES cases"
    echo "======================================================================"

    # Calculate case slices
    CASES_PER_GPU=$(( (TOTAL_CASES + NUM_GPUS - 1) / NUM_GPUS ))

    PIDS=()
    GPU_DIRS=()

    for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
        CASE_START=$((GPU_ID * CASES_PER_GPU))
        CASE_END=$(( (GPU_ID + 1) * CASES_PER_GPU ))
        if [[ $CASE_END -gt $TOTAL_CASES ]]; then
            CASE_END=$TOTAL_CASES
        fi
        if [[ $CASE_START -ge $TOTAL_CASES ]]; then
            echo "  GPU $GPU_ID: no cases to process, skipping"
            continue
        fi

        GPU_DIR="$OUTPUT_BASE/${MODEL}_gpu${GPU_ID}"
        GPU_DIRS+=("$GPU_DIR")
        mkdir -p "$GPU_DIR"

        echo "  GPU $GPU_ID: cases [$CASE_START, $CASE_END) → $GPU_DIR"

        $PYTHON scripts/evaluation/run_full_eval_single_load.py \
            --model "$MODEL" \
            --config "$CONFIG" \
            --output-dir "$GPU_DIR" \
            --test-suite "$TEST_SUITE" \
            --judge-workers "$JUDGE_WORKERS" \
            --gpu-id "$GPU_ID" \
            --case-start "$CASE_START" \
            --case-end "$CASE_END" \
            > "logs/eval_${MODEL}_gpu${GPU_ID}.log" 2>&1 &

        PIDS+=($!)
    done

    # Wait for all GPUs to finish
    echo ""
    echo "Waiting for ${#PIDS[@]} GPU processes..."
    FAILED=0
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "  GPU $i: done"
        else
            echo "  GPU $i: FAILED (check logs/eval_${MODEL}_gpu${i}.log)"
            FAILED=$((FAILED + 1))
        fi
    done

    if [[ $FAILED -gt 0 ]]; then
        echo "WARNING: $FAILED GPU(s) failed. Merging available results."
    fi

    # Merge results
    MERGED_DIR="$MERGED_BASE/$MODEL"
    echo ""
    echo "Merging results → $MERGED_DIR"

    # Build input-dirs arg from GPU dirs that actually have results
    MERGE_ARGS=()
    for GPU_DIR in "${GPU_DIRS[@]}"; do
        if ls "$GPU_DIR"/*.raw.json &>/dev/null; then
            MERGE_ARGS+=("$GPU_DIR/$MODEL")
        fi
    done

    if [[ ${#MERGE_ARGS[@]} -gt 0 ]]; then
        $PYTHON scripts/evaluation/merge_gpu_results.py \
            --input-dirs "${MERGE_ARGS[@]}" \
            --output-dir "$MERGED_DIR" \
            --model "$MODEL" \
            --config "$CONFIG"
        echo "Merged results saved to $MERGED_DIR"
    else
        echo "ERROR: No results to merge for $MODEL"
    fi

    echo "Done: $MODEL"
done

echo ""
echo "======================================================================"
echo "All models complete."
echo "Per-GPU results: $OUTPUT_BASE/"
echo "Merged results:  $MERGED_BASE/"
echo "Logs:            logs/"
echo "======================================================================"
