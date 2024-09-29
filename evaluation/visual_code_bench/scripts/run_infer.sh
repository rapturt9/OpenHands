#!/bin/bash
set -eo pipefail

# Variables
MODEL_CONFIG=$1
COMMIT_HASH=$2
EVAL_LIMIT=$3
NUM_WORKERS=${4:-1}  # default is 1 if not provided

echo "Running evaluation with config: $MODEL_CONFIG, commit: $COMMIT_HASH, limit: $EVAL_LIMIT, workers: $NUM_WORKERS"

# Run the inference
poetry run python evaluation/visual_code_bench/run_infer.py \
  --model-config $MODEL_CONFIG \
  --commit-hash $COMMIT_HASH \
  --eval-n-limit $EVAL_LIMIT \
  --eval-num-workers $NUM_WORKERS
