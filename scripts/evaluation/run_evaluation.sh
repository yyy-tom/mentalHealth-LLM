#!/bin/bash
# ============================================================
# LLM-as-a-Judge Evaluation Pipeline
# ============================================================
#
# End-to-end workflow:
#   Step 1: Build stratified test set (200 samples)
#   Step 2: Generate responses from trained models (GPU required)
#   Step 3: Score responses via LLM judge API
#   Step 4: Aggregate results into tables
#
# Steps 1-2 need GPU + datasets (run on GCP).
# Steps 3-4 need only API key + CPU (can run locally).
#
# Usage (run inside tmux on the VM):
#   bash scripts/evaluation/run_evaluation.sh
#
# To skip steps that are already done:
#   SKIP_BUILD=1 SKIP_GENERATE=1 bash scripts/evaluation/run_evaluation.sh
#
# To change judge model:
#   JUDGE_MODEL=claude bash scripts/evaluation/run_evaluation.sh
#
# To run multiple judges:
#   JUDGE_MODELS="gpt-4o deepseek gemini" bash scripts/evaluation/run_evaluation.sh
#
# ============================================================

set -euo pipefail

# ---------- Configuration ----------
PROJECT_DIR="${PROJECT_DIR:-/workspace/LLM}"
DATASETS_DIR="${DATASETS_DIR:-datasets}"
TEST_SET="evaluation/test_set.json"
RESPONSES_DIR="evaluation/responses"
SCORES_DIR="evaluation/scores"
RESULTS_DIR="evaluation/results"
# Backward-compatible: JUDGE_MODEL (single) -> JUDGE_MODELS (multi)
if [ -n "${JUDGE_MODEL:-}" ] && [ -z "${JUDGE_MODELS:-}" ]; then
    JUDGE_MODELS="$JUDGE_MODEL"
else
    JUDGE_MODELS="${JUDGE_MODELS:-gpt-4o}"
fi
JUDGE_RUNS="${JUDGE_RUNS:-3}"

# Skip flags (set to 1 to skip a step)
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"

echo "============================================================"
echo "  LLM-as-a-Judge Evaluation Pipeline"
echo "============================================================"
echo "  Project dir  : $PROJECT_DIR"
echo "  Datasets dir : $DATASETS_DIR"
echo "  Judge models : $JUDGE_MODELS"
echo "  Judge runs   : $JUDGE_RUNS"
echo "  Start time   : $(date)"
echo "============================================================"
echo ""

cd "$PROJECT_DIR" || { echo "ERROR: $PROJECT_DIR not found"; exit 1; }

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Install eval dependencies if needed
pip install -e ".[eval]" --quiet 2>/dev/null || true

# Create output directories
mkdir -p evaluation/responses evaluation/scores evaluation/results

# ============================================================
# Step 1: Build Test Set
# ============================================================
if [ "$SKIP_BUILD" = "0" ]; then
    echo ""
    echo "========================================"
    echo "[1/4] Building test set..."
    echo "========================================"

    if [ -f "$TEST_SET" ]; then
        echo "Test set already exists at $TEST_SET"
        python3 -c "
import json
with open('$TEST_SET') as f:
    data = json.load(f)
print(f'  Samples: {data[\"metadata\"][\"total_samples\"]}')
"
    else
        python3 scripts/evaluation/build_test_set.py \
            --datasets-dir "$DATASETS_DIR" \
            --output "$TEST_SET"
    fi
    echo ""
else
    echo "[1/4] Skipping test set build (SKIP_BUILD=1)"
fi

# ============================================================
# Step 2: Generate Responses
# ============================================================
if [ "$SKIP_GENERATE" = "0" ]; then
    echo ""
    echo "========================================"
    echo "[2/4] Generating model responses..."
    echo "========================================"

    python3 scripts/evaluation/generate_responses.py \
        --test-set "$TEST_SET" \
        --output-dir "$RESPONSES_DIR"

    echo ""
else
    echo "[2/4] Skipping response generation (SKIP_GENERATE=1)"
fi

# ============================================================
# Step 3: LLM Judge Scoring
# ============================================================
if [ "$SKIP_JUDGE" = "0" ]; then
    echo ""
    echo "========================================"
    echo "[3/4] Running LLM judge scoring..."
    echo "       Judges: $JUDGE_MODELS"
    echo "       Runs:   $JUDGE_RUNS"
    echo "========================================"

    for JUDGE in $JUDGE_MODELS; do
        echo ""
        echo "--- Judge: $JUDGE ---"

        # Check for API key per judge
        case "$JUDGE" in
            gpt-4o)
                if [ -z "${OPENAI_API_KEY:-}" ]; then
                    echo "ERROR: OPENAI_API_KEY not set for judge $JUDGE"
                    echo "  export OPENAI_API_KEY=sk-..."
                    exit 1
                fi
                ;;
            deepseek)
                if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
                    echo "ERROR: DEEPSEEK_API_KEY not set for judge $JUDGE"
                    echo "  export DEEPSEEK_API_KEY=sk-..."
                    exit 1
                fi
                ;;
            gemini)
                if [ -z "${GEMINI_API_KEY:-}" ]; then
                    echo "ERROR: GEMINI_API_KEY not set for judge $JUDGE"
                    echo "  export GEMINI_API_KEY=AIza..."
                    exit 1
                fi
                ;;
            claude)
                if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
                    echo "ERROR: ANTHROPIC_API_KEY not set for judge $JUDGE"
                    echo "  export ANTHROPIC_API_KEY=sk-ant-..."
                    exit 1
                fi
                ;;
            *)
                echo "ERROR: Unknown judge model: $JUDGE"
                echo "  Supported: gpt-4o, deepseek, gemini, claude"
                exit 1
                ;;
        esac

        python3 scripts/evaluation/run_llm_judge.py \
            --judge "$JUDGE" \
            --runs "$JUDGE_RUNS" \
            --resume \
            --responses-dir "$RESPONSES_DIR" \
            --test-set "$TEST_SET" \
            --scores-dir "$SCORES_DIR"
    done

    echo ""
else
    echo "[3/4] Skipping LLM judge scoring (SKIP_JUDGE=1)"
fi

# ============================================================
# Step 4: Aggregate Results
# ============================================================
if [ "$SKIP_AGGREGATE" = "0" ]; then
    echo ""
    echo "========================================"
    echo "[4/4] Aggregating results..."
    echo "========================================"

    AGGREGATE_ARGS="--scores-dir $SCORES_DIR --output-dir $RESULTS_DIR"

    # Include human scores if available
    if [ -f "evaluation/human_scores.json" ]; then
        echo "  Including human scores for correlation"
        AGGREGATE_ARGS="$AGGREGATE_ARGS --human-scores evaluation/human_scores.json"
    fi

    python3 scripts/evaluation/aggregate_results.py $AGGREGATE_ARGS

    echo ""
    echo "========================================"
    echo "  Results:"
    echo "========================================"
    echo ""
    cat "$RESULTS_DIR/comparison_tables.md"
else
    echo "[4/4] Skipping aggregation (SKIP_AGGREGATE=1)"
fi

echo ""
echo "============================================================"
echo "  Evaluation Pipeline Complete!"
echo "  End time: $(date)"
echo "============================================================"
echo ""
echo "Output files:"
echo "  Test set:     $TEST_SET"
echo "  Responses:    $RESPONSES_DIR/"
echo "  Scores:       $SCORES_DIR/"
echo "  Results:      $RESULTS_DIR/comparison_tables.md"
echo "  Full data:    $RESULTS_DIR/full_results.json"
