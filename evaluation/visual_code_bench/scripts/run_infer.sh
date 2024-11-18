#!/bin/bash

# Check if required arguments are provided
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 [model_config] [commit_hash] [agent_cls] [eval_limit] [num_workers]"
    echo "Example: $0 llm.eval_gpt4o_mini HEAD CodeActAgent 5 1"
    exit 1
fi

MODEL_CONFIG=$1
COMMIT_HASH=$2
AGENT_CLS=$3
EVAL_LIMIT=$4
NUM_WORKERS=${5:-1}  # Default to 1 worker if not specified

# Navigate to the repository root
cd "$(git rev-parse --show-toplevel)"

# Checkout the specified commit
git checkout $COMMIT_HASH

# Run the evaluation
python3 -m evaluation.visual_code_bench.run_infer \
    --llm-config $MODEL_CONFIG \
    --agent-cls $AGENT_CLS \
    --eval-n-limit $EVAL_LIMIT \
    --eval-num-workers $NUM_WORKERS
