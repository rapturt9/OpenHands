mamba activate openhands

Main code: https://github.com/malhotra5/VisualCodeBench

Huggingface: https://huggingface.co/datasets/rvmalhot/VisualCodeBench

# VisualCodeBench Evaluation

This folder contains the evaluation harness for evaluating changes in the VisualCodeBench dataset.

## Setup Environment and LLM Configuration

Please follow instructions `/Development.md` to setup your local development environment

Evaluation instructions: `/Users/ram/Documents/GitHub/OpenDevin/docs/modules/usage/how-to/evaluation-harness.md`

## Running the Evaluation

To start an evaluation run, use the following command:

```bash
./evaluation/visual_code_bench/scripts/run_infer.sh [model_config] [commit_hash] [eval_limit] [num_workers]

# example
./evaluation/visual_code_bench/scripts/run_infer.sh llm.eval_gpt4o_mini HEAD CodeActAgent 5 1

```

Huggingface DB Schema:

---

dataset_info:
features:

- name: id
  dtype: int32
- name: prev_image
  dtype: image
- name: post_image
  dtype: image
- name: changes
  dtype: string
- name: prev_code_files
  dtype: string
- name: post_code_files
  dtype: string
  splits:
- name: train
  num_bytes: 332523506.0
  num_examples: 15
  download_size: 321589914
  dataset_size: 332523506.0
  configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-\*

---

Tasks

1. Layout edits: reorganizing existing content

- Removing elements
- Move location for one
- Swap two locations
- Adding elements
- Replacing elements
- Text change

2. Stylistic edits: changing styles of existing elements

- Color change
- Style change
- Interaction change
- Changing element sizes

Inputs:
Text description of changes to be made
Simple or compound descriptions
Virtual post it notes on webpage image to indicating where changes should be made

Evaluation
Clip score for image similarity and block level matching of html elements

Our goal is to create the evaluation to get the new html file by running codeact on the original html and instructions. We will then compare the new html file (and screenshot) with the original html file (and screenshot) to get the clip score.

Take inspiration from `evaluation/bird/README.md`
