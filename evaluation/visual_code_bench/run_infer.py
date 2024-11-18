import argparse
import json
import os

from datasets import load_dataset
from PIL import Image, ImageChops

from openhands.core.config import (
    AppConfig,
    SandboxConfig,
    get_llm_config_arg,
)
from openhands.events.action import CmdRunAction

# Define workspace and output directories
WORKSPACE_DIR = './workspace'
SCREENSHOT_DIR = './evaluation/visual_code_bench/tasks/screenshots'
HTML_FILES_DIR = './evaluation/visual_code_bench/tasks/html_files'

# Ensure directories exist
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(HTML_FILES_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Run VisualCodeBench with OpenHands')
    parser.add_argument(
        '--agent-cls', type=str, required=True, help='Agent class to use'
    )
    parser.add_argument(
        '--llm-config', type=str, required=True, help='LLM configuration'
    )
    parser.add_argument(
        '--max-iterations', type=int, default=1, help='Pass@1: Single pass for testing'
    )
    parser.add_argument(
        '--eval-note', type=str, default='v1', help='Evaluation note/version'
    )
    parser.add_argument(
        '--eval-num-workers',
        type=int,
        default=1,
        help='Number of workers for parallel processing',
    )
    parser.add_argument(
        '--eval-n-limit',
        type=int,
        default=None,
        help='Limit the number of tasks to evaluate',
    )
    return parser.parse_args()


def load_visualcodebench(limit=None):
    from huggingface_hub import hf_hub_download

    dataset = load_dataset('rvmalhot/VisualCodeBench', split='train')
    if limit:
        dataset = dataset.select(range(limit))

    # Convert PIL images to bytes for saving
    def process_example(example):
        # Save images as bytes
        prev_img_bytes = example['prev_image']
        post_img_bytes = example['post_image']

        # Load code files from the dataset
        task_id = example['id']
        try:
            prev_html_path = hf_hub_download(
                repo_id='rvmalhot/VisualCodeBench',
                filename=f'data/{task_id}/prev/index.html',
                repo_type='dataset',
            )
            with open(prev_html_path, 'r') as f:
                prev_code = f.read()
        except Exception as e:
            print(f'Error loading prev HTML for task {task_id}: {e}')
            prev_code = example['prev_code_files']

        try:
            post_html_path = hf_hub_download(
                repo_id='rvmalhot/VisualCodeBench',
                filename=f'data/{task_id}/post/index.html',
                repo_type='dataset',
            )
            with open(post_html_path, 'r') as f:
                post_code = f.read()
        except Exception as e:
            print(f'Error loading post HTML for task {task_id}: {e}')
            post_code = example['post_code_files']

        return {
            'id': example['id'],
            'prev_image': prev_img_bytes,
            'post_image': post_img_bytes,
            'changes': example['changes'],
            'prev_code_files': prev_code,
            'post_code_files': post_code,
        }

    return dataset.map(process_example)


def setup_workspace(task):
    """Prepare the workspace by setting up HTML and assets."""
    html_path = os.path.join(WORKSPACE_DIR, 'index.html')
    with open(html_path, 'w') as f:
        f.write(task['prev_code_files'])

    # Save the prev_image for reference
    prev_img_path = os.path.join(WORKSPACE_DIR, 'prev_image.png')
    task['prev_image'].save(prev_img_path)

    # Instruction to the agent
    instruction = (
        f"Modify the HTML/CSS according to the following instruction:\n\n"
        f"{task['changes']}\n\n"
        "When done, save the modified HTML to 'index.html' and ensure all changes are complete.\n"
        "Exit when complete: <execute_bash> exit </execute_bash>.\n"
    )
    return instruction


def capture_screenshot(runtime, task_id):
    """Capture a screenshot of the modified HTML for comparison."""
    screenshot_path = os.path.join(SCREENSHOT_DIR, f'{task_id}_screenshot.png')
    runtime.run_action(
        CmdRunAction(command='screenshot index.html', save_as=screenshot_path)
    )
    return screenshot_path


def evaluate(task, screenshot_path):
    """Simple pixel-by-pixel comparison of generated screenshot with post_image."""
    post_img = task['post_image']
    generated_img = Image.open(screenshot_path)

    # Calculate the difference and check if the image matches
    diff = ImageChops.difference(generated_img, post_img)
    return not diff.getbbox()  # Returns True if images match exactly


def run_task(task, config, agent_cls):
    # Create workspace directory if it doesn't exist
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    # Save the HTML file
    html_path = os.path.join(WORKSPACE_DIR, 'index.html')
    with open(html_path, 'w') as f:
        f.write(task['prev_code_files'])

    # Save the prev_image for reference
    prev_img_path = os.path.join(WORKSPACE_DIR, 'prev_image.png')
    task['prev_image'].save(prev_img_path)

    # Save the post_image for comparison
    post_img_path = os.path.join(WORKSPACE_DIR, 'post_image.png')
    task['post_image'].save(post_img_path)

    # Save the results
    result = {
        'task_id': task['id'],
        'passed': False,  # Can't evaluate without Docker
        'screenshot_path': None,
        'html_path': os.path.join(HTML_FILES_DIR, f"{task['id']}_index.html"),
    }
    save_results(result)

    print(f"\nTask {task['id']} changes:")
    print(task['changes'])


def save_results(result):
    """Save HTML and screenshot for each task."""
    html_save_path = result['html_path']
    try:
        with open(os.path.join(WORKSPACE_DIR, 'index.html'), 'r') as src:
            with open(html_save_path, 'w') as dst:
                dst.write(src.read())
    except Exception as e:
        print(f'Error saving HTML: {e}')

    results_file_path = os.path.join(HTML_FILES_DIR, 'results.json')
    try:
        # Load existing results
        results = []
        if os.path.exists(results_file_path):
            with open(results_file_path, 'r') as f:
                try:
                    results = json.load(f)
                    if not isinstance(results, list):
                        results = [results]
                except json.JSONDecodeError:
                    results = []

        # Add new result
        results.append(result)

        # Save updated results
        with open(results_file_path, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f'Error saving results: {e}')


def get_config(agent_cls, llm_config_arg, max_iterations):
    config = AppConfig(
        default_agent=agent_cls,
        run_as_openhands=False,
        runtime='eventstream',
        max_iterations=max_iterations,
        sandbox=SandboxConfig(
            base_container_image='python:3.12-bookworm',
            enable_auto_lint=True,
            use_host_network=False,
            platform='linux/amd64',  # Specify platform for M1 compatibility
        ),
        workspace_base=None,
        workspace_mount_path=None,
    )

    config.set_llm_config(llm_config_arg)
    return config


def main():
    args = parse_args()
    llm_config_arg = get_llm_config_arg(args.llm_config)

    if llm_config_arg is None:
        raise ValueError(f'Could not find LLM config: {args.llm_config}')

    # Create the AppConfig using the get_config function
    config = get_config(args.agent_cls, llm_config_arg, args.max_iterations)

    # Load the dataset from Hugging Face
    dataset = load_visualcodebench(limit=args.eval_n_limit)

    # Process each task in the dataset
    for task in dataset:
        run_task(task, config, args.agent_cls)


if __name__ == '__main__':
    main()
