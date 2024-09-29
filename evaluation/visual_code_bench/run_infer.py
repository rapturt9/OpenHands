import argparse

from datasets import load_dataset


# Argument parser for command-line inputs
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-config', required=True, help='Model configuration to use'
    )
    parser.add_argument(
        '--commit-hash',
        required=True,
        help='Git commit hash of the version being tested',
    )
    parser.add_argument(
        '--eval-n-limit', type=int, default=10, help='Limit the number of evaluations'
    )
    parser.add_argument(
        '--eval-num-workers',
        type=int,
        default=1,
        help='Number of workers for evaluation',
    )
    return parser.parse_args()


def load_visual_code_bench_data():
    """Load the VisualCodeBench dataset from Hugging Face"""
    dataset = load_dataset('rvmalhot/VisualCodeBench', split='train')
    return dataset


def run_evaluation(dataset, eval_limit):
    """Dummy function to evaluate changes in the dataset"""
    for i, data in enumerate(dataset):
        if i >= eval_limit:
            break
        print(f"Evaluating instance {i}: {data['data/1/prev/index2.html']}")
        # Simulate the evaluation logic here


if __name__ == '__main__':
    args = parse_arguments()
    print(
        f'Running evaluation with {args.model_config} and commit hash {args.commit_hash}'
    )
    dataset = load_visual_code_bench_data()
    run_evaluation(dataset, args.eval_n_limit)
