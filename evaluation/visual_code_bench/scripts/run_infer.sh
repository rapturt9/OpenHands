#!/bin/bash
set -eo pipefail

# Variables from command-line arguments
MODEL_CONFIG=$1
COMMIT_HASH=$2
AGENT=$3
EVAL_LIMIT=$4
MAX_ITER=$5
NUM_WORKERS=$6
DATASET=${7:-"rvmalhot/VisualCodeBench"}  # default dataset if not provided
DATASET_SPLIT=${8:-"train"}  # default dataset split if not provided

# Default values if not provided
if [ -z "$NUM_WORKERS" ]; then
  NUM_WORKERS=1
  echo "Number of workers not specified, using default: $NUM_WORKERS"
fi

if [ -z "$AGENT" ]; then
  AGENT="CodeActAgent"
  echo "Agent not specified, using default: $AGENT"
fi

if [ -z "$MAX_ITER" ]; then
  MAX_ITER=30
  echo "MAX_ITER not specified, using default: $MAX_ITER"
fi

if [ -z "$USE_INSTANCE_IMAGE" ]; then
  USE_INSTANCE_IMAGE=true
  echo "USE_INSTANCE_IMAGE not specified, using default: $USE_INSTANCE_IMAGE"
fi

# Export the environment variables
export USE_INSTANCE_IMAGE=$USE_INSTANCE_IMAGE
echo "USE_INSTANCE_IMAGE: $USE_INSTANCE_IMAGE"

# Echo input details for debugging
echo "Running evaluation with config: $MODEL_CONFIG, commit: $COMMIT_HASH, limit: $EVAL_LIMIT, workers: $NUM_WORKERS, agent: $AGENT"
echo "Dataset: $DATASET, Dataset Split: $DATASET_SPLIT"

# Run the inference command
COMMAND="poetry run python evaluation/visual_code_bench/run_infer.py \
  --model-config $MODEL_CONFIG \
  --commit-hash $COMMIT_HASH \
  --agent-cls $AGENT \
  --eval-n-limit $EVAL_LIMIT \
  --max-iterations $MAX_ITER \
  --eval-num-workers $NUM_WORKERS \
  --dataset $DATASET \
  --dataset-split $DATASET_SPLIT"

# Check if EVAL_LIMIT is specified
if [ -n "$EVAL_LIMIT" ]; then
  echo "EVAL_LIMIT: $EVAL_LIMIT"
  COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
fi

# Execute the command
eval $COMMAND
