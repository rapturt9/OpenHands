import argparse

from datasets import load_dataset
from PIL import Image

# or
Image.MAX_IMAGE_PIXELS = 10**8  # Set to a large enough value to handle large images


# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config', required=True, help='Model configuration')
    parser.add_argument('--commit-hash', required=True, help='Git commit hash')
    parser.add_argument('--agent-cls', required=True, help='Agent class to use')
    parser.add_argument(
        '--eval-n-limit', type=int, default=10, help='Number of evaluations to run'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=30,
        help='Max iterations for the evaluation',
    )
    parser.add_argument(
        '--eval-num-workers',
        type=int,
        default=1,
        help='Number of workers for evaluation',
    )
    parser.add_argument(
        '--dataset', default='rvmalhot/VisualCodeBench', help='Dataset to use'
    )
    parser.add_argument(
        '--dataset-split',
        default='train',
        help='Dataset split to use (e.g., train, test)',
    )
    return parser.parse_args()


def apply_changes(prev_code, changes):
    """Dummy function to apply changes to the code."""
    # In a real implementation, you would parse and apply changes from the changes file
    return f'{prev_code}\n<!-- Changes Applied: {changes} -->'


def main():
    args = parse_arguments()

    # Load the dataset from Hugging Face
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    print(f'Loaded dataset: {args.dataset} (split: {args.dataset_split})')

    # Iterate through each instance in the dataset
    for i, data in enumerate(dataset):
        if i >= args.eval_n_limit:
            break

        # Extract the previous code, changes, and the post-code for evaluation
        prev_code_files = data['prev_code_files']
        changes = data['changes']
        post_code_files = data['post_code_files']

        # Apply changes (dummy logic here, you need to replace this with actual application of changes)
        evaluated_code = apply_changes(prev_code_files, changes)

        # For this example, we simply compare the applied changes with the post code
        if evaluated_code.strip() == post_code_files.strip():
            print(f'Evaluation for instance {i}: Success')
        else:
            print(f'Evaluation for instance {i}: Failure')
            print(f'Expected:\n{post_code_files}\n')
            print(f'Got:\n{evaluated_code}\n')


if __name__ == '__main__':
    main()
