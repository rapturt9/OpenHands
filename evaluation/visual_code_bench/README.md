mamba activate openhands

Main code: https://github.com/malhotra5/VisualCodeBench

Huggingface: https://huggingface.co/datasets/rvmalhot/VisualCodeBench

# VisualCodeBench Evaluation

This folder contains the evaluation harness for evaluating changes in the VisualCodeBench dataset.

## Setup Environment and LLM Configuration

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.

## Running the Evaluation

To start an evaluation run, use the following command:

```bash
./evaluation/visual_code_bench/scripts/run_infer.sh [model_config] [commit_hash] [eval_limit] [num_workers]

# example
./evaluation/visual_code_bench/scripts/run_infer.sh llm.eval_gpt4o_mini HEAD CodeActAgent 5 1

```
