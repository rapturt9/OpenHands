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
from openhands.core.main import create_runtime
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
    dataset = load_dataset('rvmalhot/VisualCodeBench', split='train')
    if limit:
        dataset = dataset.select(range(limit))
    return dataset


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
    # instruction = setup_workspace(task)

    runtime = create_runtime(config)
    try:
        # Copy setup files to workspace
        runtime.copy_to(WORKSPACE_DIR, '/workspace')

        # Run the agent
        # state: State = run_controller(
        #     config=config,
        #     initial_user_action=MessageAction(content=instruction),
        #     fake_user_response_fn=lambda state: 'Please continue working as per instructions.',
        #     runtime=runtime,
        # )

        # Capture the screenshot of the final HTML output
        screenshot_path = capture_screenshot(runtime, task['id'])

        # Evaluation: Compare the screenshot with the expected post_image
        passed = evaluate(task, screenshot_path)

        # Save the results
        result = {
            'task_id': task['id'],
            'passed': passed,
            'screenshot_path': screenshot_path,
            'html_path': os.path.join(HTML_FILES_DIR, f"{task['id']}_index.html"),
        }
        save_results(result)
    finally:
        runtime.close()


def save_results(result):
    """Save HTML and screenshot for each task."""
    html_save_path = result['html_path']
    with open(html_save_path, 'w') as f:
        f.write(open(os.path.join(WORKSPACE_DIR, 'index.html')).read())
    with open(os.path.join(HTML_FILES_DIR, 'results.json'), 'a') as results_file:
        json.dump(result, results_file, indent=4)


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
