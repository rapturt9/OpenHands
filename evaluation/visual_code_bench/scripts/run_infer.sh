#!/bin/bash
set -eo pipefail

# Usage: ./evaluation/visual_code_bench/scripts/run_infer.sh <model_config> <commit_hash> [agent] [eval_limit] [num_workers]

MODEL_CONFIG=$1
COMMIT_HASH=$2
AGENT=${3:-CodeActAgent}
EVAL_LIMIT=${4:-}    # Optional limit on the number of tasks
NUM_WORKERS=${5:-1}  # Default to 1 worker

echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "COMMIT_HASH: $COMMIT_HASH"
echo "AGENT: $AGENT"
echo "EVAL_LIMIT: $EVAL_LIMIT"
echo "NUM_WORKERS: $NUM_WORKERS"

# Optional: Checkout the specific commit/version if needed
# git checkout $COMMIT_HASH

# Execute the inference with an optional task limit
poetry run python evaluation/visual_code_bench/run_infer.py \
  --agent-cls "$AGENT" \
  --llm-config "$MODEL_CONFIG" \
  --max-iterations 5 \
  --eval-num-workers "$NUM_WORKERS" \
  --eval-note "commit_$COMMIT_HASH" \
  ${EVAL_LIMIT:+--eval-n-limit "$EVAL_LIMIT"}
