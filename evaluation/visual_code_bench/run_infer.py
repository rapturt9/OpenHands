# FILE: run_infer.py

import argparse
import json
import os
import shutil
import socket
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from io import BytesIO

from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

from openhands.core.config import (
    AppConfig,
    SandboxConfig,
    get_llm_config_arg,
)
from openhands.core.logger import openhands_logger as logger  # Import OpenHands logger

# Define workspace and output directories
WORKSPACE_DIR = './workspace'
SCREENSHOT_DIR = './evaluation/visual_code_bench/tasks/screenshots'
HTML_FILES_DIR = './evaluation/visual_code_bench/tasks/html_files'
REPO_DOWNLOAD_DIR = './repo_downloaded'  # Directory to store the downloaded repository

# Ensure directories exist
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(HTML_FILES_DIR, exist_ok=True)
os.makedirs(REPO_DOWNLOAD_DIR, exist_ok=True)


def download_repository():
    """
    Download the entire repository from Hugging Face Hub.
    This function clones the repository into REPO_DOWNLOAD_DIR.
    """
    repo_id = 'rvmalhot/VisualCodeBench'
    try:
        logger.info(f"Downloading repository '{repo_id}'...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=REPO_DOWNLOAD_DIR,
            repo_type='dataset',
            ignore_patterns=None,  # Download all files
        )
        logger.info(f"Repository downloaded to '{REPO_DOWNLOAD_DIR}'.")
    except Exception as e:
        logger.error(f"Error downloading repository '{repo_id}': {e}")
        raise e


def process_example(example):
    """Process each example in the dataset by extracting necessary files from the downloaded repository."""
    # Extract task_id
    task_id = str(example.get('id', 'unknown'))

    logger.info(f'Processing task ID: {task_id}')

    # Define paths within the downloaded repository
    prev_remote_path = os.path.join(REPO_DOWNLOAD_DIR, f'data/{task_id}/prev')
    post_remote_path = os.path.join(REPO_DOWNLOAD_DIR, f'data/{task_id}/post')

    # Check if 'prev' and 'post' directories exist
    prev_exists = os.path.exists(prev_remote_path)
    post_exists = os.path.exists(post_remote_path)

    # Check for the existence of 'prev_image' and 'post_image' in the example
    has_prev_image = 'prev_image' in example and example['prev_image'] is not None
    has_post_image = 'post_image' in example and example['post_image'] is not None

    if not (prev_exists and post_exists and has_prev_image and has_post_image):
        logger.warning(
            f'Skipping task {task_id}: '
            f'prev_exists={prev_exists}, post_exists={post_exists}, '
            f'has_prev_image={has_prev_image}, has_post_image={has_post_image}'
        )
        return {'skip': True}  # Indicate that this example should be skipped

    # Convert 'prev_image' to bytes if necessary
    try:
        if isinstance(example['prev_image'], bytes):
            prev_image_bytes = example['prev_image']
        elif isinstance(example['prev_image'], str):
            with open(example['prev_image'], 'rb') as img_file:
                prev_image_bytes = img_file.read()
            logger.info(f"Loaded 'prev_image' from file path for task {task_id}.")
        elif isinstance(example['prev_image'], Image.Image):
            buffer = BytesIO()
            example['prev_image'].save(buffer, format='PNG')
            prev_image_bytes = buffer.getvalue()
            logger.info(f"Converted 'prev_image' from PIL Image for task {task_id}.")
        else:
            logger.warning(
                f"Skipping task {task_id}: 'prev_image' has unsupported type {type(example['prev_image'])}."
            )
            return {'skip': True}
    except Exception as e:
        logger.error(f"Error processing 'prev_image' for task {task_id}: {e}")
        return {'skip': True}

    # Convert 'post_image' to bytes if necessary
    try:
        if isinstance(example['post_image'], bytes):
            post_image_bytes = example['post_image']
        elif isinstance(example['post_image'], str):
            with open(example['post_image'], 'rb') as img_file:
                post_image_bytes = img_file.read()
            logger.info(f"Loaded 'post_image' from file path for task {task_id}.")
        elif isinstance(example['post_image'], Image.Image):
            buffer = BytesIO()
            example['post_image'].save(buffer, format='PNG')
            post_image_bytes = buffer.getvalue()
            logger.info(f"Converted 'post_image' from PIL Image for task {task_id}.")
        else:
            logger.warning(
                f"Skipping task {task_id}: 'post_image' has unsupported type {type(example['post_image'])}."
            )
            return {'skip': True}
    except Exception as e:
        logger.error(f"Error processing 'post_image' for task {task_id}: {e}")
        return {'skip': True}

    # Proceed to extract other fields
    try:
        # Read the main HTML file for 'prev'
        index_html_path = os.path.join(prev_remote_path, 'index.html')
        with open(index_html_path, 'r') as f:
            prev_code = f.read()
        logger.info(f'Successfully read {index_html_path}')
    except Exception as e:
        logger.error(f'Error reading {index_html_path}: {e}')
        prev_code = example.get('prev_code_files', '')
        logger.info("Using default 'prev_code_files' from example.")

    # Optionally, read 'changes.md' if it exists
    changes_md_path = os.path.join(prev_remote_path, 'changes.md')
    if os.path.exists(changes_md_path):
        try:
            with open(changes_md_path, 'r') as f:
                changes = f.read()
            logger.info(f'Successfully read {changes_md_path}')
        except Exception as e:
            logger.error(f'Error reading {changes_md_path}: {e}')
            changes = example.get('changes', '')
            logger.info("Using default 'changes' from example.")
    else:
        changes = example.get('changes', '')
        logger.info(
            f"'changes.md' not found for task {task_id}. Using default 'changes'."
        )

    try:
        # Read the main HTML file for 'post'
        post_html_path = os.path.join(post_remote_path, 'index.html')
        with open(post_html_path, 'r') as f:
            post_code = f.read()
        logger.info(f'Successfully read {post_html_path}')
    except Exception as e:
        logger.error(f'Error reading {post_html_path}: {e}')
        post_code = example.get('post_code_files', '')
        logger.info("Using default 'post_code_files' from example.")

    logger.info(f'Task {task_id} processed successfully.')
    return {
        'task_id': task_id,
        'prev_image': prev_image_bytes,
        'post_image': post_image_bytes,
        'changes': changes,
        'prev_code_files': prev_code,
        'post_code_files': post_code,
        'skip': False,  # Indicate that this example should be processed
    }


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

    # Map the dataset using the updated process_example
    dataset = dataset.map(
        process_example, remove_columns=dataset.column_names, load_from_cache_file=False
    )

    # Filter out skipped tasks
    dataset = dataset.filter(lambda example: not example.get('skip', False))

    # Remove the 'skip' column
    if 'skip' in dataset.column_names:
        dataset = dataset.remove_columns(['skip'])

    # Verify that all entries have 'prev_image'
    for example in dataset:
        if 'prev_image' not in example:
            logger.error(
                f"Dataset entry missing 'prev_image': {example.get('task_id', 'unknown')}"
            )
            raise KeyError(
                f"Dataset entry missing 'prev_image' for task {example.get('task_id', 'unknown')}"
            )

    return dataset


def start_http_server(server):
    """Start the HTTP server."""
    logger.info(f'Starting HTTP server on {server.server_address}...')
    try:
        server.serve_forever()
    except Exception as e:
        logger.error(f'HTTP server error: {e}')


def get_free_port():
    """Find a free port to run the HTTP server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def setup_workspace(task):
    """Prepare the workspace by setting up HTML and assets."""
    # Clear the workspace directory to avoid residual files
    if os.path.exists(WORKSPACE_DIR):
        shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    logger.info(f"Workspace directory '{WORKSPACE_DIR}' is ready.")

    # Write the previous code to 'index.html' in the workspace
    html_path = os.path.join(WORKSPACE_DIR, 'index.html')
    with open(html_path, 'w') as f:
        f.write(task['prev_code_files'])
    logger.info(f'Written previous code to {html_path}')

    # Copy all files from 'prev' folder to workspace
    prev_folder = os.path.join(REPO_DOWNLOAD_DIR, f'data/{task["task_id"]}/prev')
    if os.path.exists(prev_folder):
        for root, dirs, files in os.walk(prev_folder):
            for file in files:
                src_path = os.path.join(root, file)
                relative_path = os.path.relpath(src_path, prev_folder)
                dest_path = os.path.join(WORKSPACE_DIR, relative_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(src_path, dest_path)
                logger.info(f'Copied {src_path} to {dest_path}')
    else:
        logger.warning(
            f"Prev folder '{prev_folder}' does not exist for task {task['task_id']}. Skipping task."
        )
        return None

    # Ensure 'changes.md' is copied if it exists
    changes_md_path = os.path.join(prev_folder, 'changes.md')
    if os.path.exists(changes_md_path):
        dest_changes_md = os.path.join(WORKSPACE_DIR, 'changes.md')
        shutil.copy(changes_md_path, dest_changes_md)
        logger.info(f'Copied {changes_md_path} to {dest_changes_md}')
    else:
        logger.warning(
            f"'changes.md' not found for task {task['task_id']}. Skipping copy."
        )

    # Save the prev_image for reference
    prev_img_path = os.path.join(WORKSPACE_DIR, 'prev_image.png')
    try:
        # Ensure that prev_image is in bytes
        if isinstance(task['prev_image'], bytes):
            prev_image = Image.open(BytesIO(task['prev_image']))
            prev_image.save(prev_img_path)
            logger.info(f'Saved prev_image to {prev_img_path}')
        else:
            logger.warning(
                f"Invalid type for prev_image in task {task['task_id']}. Skipping image saving."
            )
            prev_image = None
    except Exception as e:
        logger.error(f"Error saving prev_image for task {task['task_id']}: {e}")
        prev_image = None

    # Instruction to the agent
    instruction = (
        f"Modify the HTML/CSS according to the following instruction:\n\n"
        f"{task['changes']}\n\n"
        "When done, save the modified HTML to 'index.html' and ensure all changes are complete.\n"
        "Exit when complete: <execute_bash> exit </execute_bash>.\n"
    )
    return instruction


def capture_screenshot_playwright(url, screenshot_path):
    """Capture a screenshot of the given URL using Playwright."""
    try:
        with sync_playwright() as p:
            logger.info('Launching browser...')
            browser = p.chromium.launch(timeout=10000)  # 10 seconds for browser launch

            logger.info('Creating a new page...')
            page = browser.new_page()

            logger.info(f'Navigating to URL: {url}')
            try:
                page.goto(url, timeout=5000)  # Set timeout to 5 seconds
                logger.info('Page navigation completed.')
            except Exception as e:
                logger.warning(f'Page navigation timed out. {e}. Continuing...')

            logger.info('Waiting for network to be idle...')
            try:
                page.wait_for_load_state(
                    'networkidle', timeout=5000
                )  # Set timeout to 5 seconds
                logger.info('Page load state reached.')
            except Exception as e:
                logger.warning(f'Page load state timed out. {e}. Continuing...')

            logger.info('Capturing screenshot...')
            page.screenshot(
                path=screenshot_path, full_page=True
            )  # Capture full page screenshot

            logger.info(f'Screenshot saved to {screenshot_path}')
            browser.close()
            return True
    except Exception as e:
        logger.error(f'Error capturing screenshot with Playwright: {e}')
        return False


def evaluate(task, screenshot_path):
    """Compare generated screenshot with post_image using CLIP score."""
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        # Load CLIP model and processor
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

        # Load images
        post_image = Image.open(BytesIO(task['post_image']))
        generated_img = Image.open(screenshot_path)

        # Process images
        inputs = processor(
            images=[post_image, generated_img], return_tensors='pt', padding=True
        )

        # Get image features
        image_features = model.get_image_features(**inputs)

        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            image_features[0].unsqueeze(0), image_features[1].unsqueeze(0)
        ).item()

        logger.info(f'CLIP similarity score: {similarity}')

        return similarity > 0.95  # Consider it a match if similarity > 95%
    except Exception as e:
        logger.error(f'Error in CLIP evaluation: {e}')
        # Fallback to pixel comparison if CLIP fails
        try:
            post_image = Image.open(BytesIO(task['post_image']))
            generated_img = Image.open(screenshot_path)

            # Compare images directly without converting to bytes
            diff = ImageChops.difference(generated_img, post_image)
            logger.info(
                f"Pixel difference analysis: {'No difference' if not diff.getbbox() else 'Differences found'}"
            )
            return not diff.getbbox()
        except Exception as ex:
            logger.error(f'Error in fallback evaluation: {ex}')
            return False


def run_task(task, config, agent_cls):
    """Run a single task using the sandbox environment."""
    task_id = task['task_id']
    logger.info(f"\n{'='*80}\nExecuting task {task_id}\n{'='*80}")

    try:
        # Setup workspace and get instruction
        instruction = setup_workspace(task)
        if instruction is None:
            logger.warning(f'Skipping task {task_id} due to workspace setup issues.')
            return {
                'task_id': task_id,
                'passed': False,
                'error': 'Workspace setup failed due to missing data.',
                'screenshot_path': None,
                'html_path': None,
            }

        # Define the workspace path directly
        task_workspace = WORKSPACE_DIR

        # Uncomment and implement the agent runtime if needed
        """
        # Create runtime and run the agent
        logger.info(f"Creating runtime for task {task_id}...")
        runtime = create_runtime(config)
        logger.info(f"Running controller for task {task_id}...")
        state = asyncio.run(
            run_controller(
                config=config,
                initial_user_action=MessageAction(content=instruction),
                runtime=runtime,
            )
        )

        if state is None:
            logger.warning(f"Task {task_id} did not complete successfully.")
            return {
                'task_id': task_id,
                'passed': False,
                'error': state.last_error if state else 'Task did not complete',
                'screenshot_path': None,
                'html_path': None,
            }
        """

        # Save modified HTML directly in HTML_FILES_DIR
        html_filename = 'index.html'  # Use consistent naming
        html_path = os.path.join(HTML_FILES_DIR, html_filename)
        modified_html_path = os.path.join(
            task_workspace, 'index.html'
        )  # Ensure correct path
        if os.path.exists(modified_html_path):
            shutil.copy(modified_html_path, html_path)
            logger.info(f'Saved modified HTML to {html_path}')
        else:
            logger.error(f"Modified 'index.html' not found for task {task_id}")
            return {
                'task_id': task_id,
                'passed': False,
                'error': "Modified 'index.html' not found.",
                'screenshot_path': None,
                'html_path': None,
            }

        # Start HTTP server serving the WORKSPACE_DIR
        port = get_free_port()
        server_address = ('localhost', port)
        handler = partial(SimpleHTTPRequestHandler, directory=WORKSPACE_DIR)
        httpd = HTTPServer(server_address, handler)
        server_thread = threading.Thread(
            target=start_http_server, args=(httpd,), daemon=True
        )
        server_thread.start()
        logger.info(
            f"HTTP server started on port {port}, serving directory '{WORKSPACE_DIR}'."
        )

        # Wait briefly to ensure the server starts
        import time

        time.sleep(1)

        # Define the URL to access 'index.html'
        url = f'http://localhost:{port}'

        # Capture screenshot using Playwright
        logger.info(
            f'Capturing screenshot for task {task_id} with Playwright from {url}...'
        )
        screenshot_path = os.path.join(SCREENSHOT_DIR, f'{task_id}_screenshot.png')
        """screenshot_captured = capture_screenshot_playwright(url, screenshot_path)"""
        screenshot_captured = True  # Skip Playwright for now

        # Shutdown the HTTP server
        httpd.shutdown()
        server_thread.join()
        logger.info(f'HTTP server on port {port} has been shut down.')

        if not screenshot_captured:
            logger.error(f'Failed to capture screenshot for task {task_id}.')
            passed = False
        else:
            # Evaluate results
            logger.info(f'Evaluating results for task {task_id}...')
            passed = evaluate(task, screenshot_path)

        result = {
            'task_id': task_id,
            'passed': passed,
            'screenshot_path': screenshot_path if screenshot_captured else None,
            'html_path': html_path,
        }
        save_results(result)
        logger.info(f"Task {task_id} completed. {'PASSED' if passed else 'FAILED'}.")
        return result

    except Exception as e:
        logger.error(f'Error running task {task_id}: {e}', exc_info=True)
        return {
            'task_id': task_id,
            'passed': False,
            'error': str(e),
            'screenshot_path': None,
            'html_path': None,
        }


def save_results(result):
    """Save HTML and screenshot for each task."""
    results_file_path = os.path.join(HTML_FILES_DIR, 'results.json')
    try:
        # Load existing results
        if os.path.exists(results_file_path):
            with open(results_file_path, 'r') as f:
                try:
                    data = json.load(f)
                    if 'results' in data:
                        existing_results = data['results']
                    elif isinstance(data, list):
                        existing_results = data
                        data = {'results': existing_results}
                    else:
                        existing_results = []
                        data = {'results': existing_results}
                except json.JSONDecodeError:
                    logger.warning(
                        f'JSON decode error in {results_file_path}. Initializing new results.'
                    )
                    existing_results = []
                    data = {'results': existing_results}
        else:
            existing_results = []
            data = {'results': existing_results}

        # Append the new result
        existing_results.append(result)
        logger.info(f"Appended result for task {result['task_id']}.")

        # Write back to the results file
        with open(results_file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f'Results saved successfully to {results_file_path}.')

    except Exception as e:
        logger.error(f'Error saving results: {e}', exc_info=True)


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
            platform='linux/amd64',  # Specify platform for compatibility
        ),
        workspace_base=WORKSPACE_DIR,  # Set workspace_base to './workspace/'
        workspace_mount_path='/workspace',  # Map to container's /workspace
    )

    config.set_llm_config(llm_config_arg)
    return config


def get_event_stream():
    from openhands.events import EventStream
    from openhands.storage import InMemoryFileStore

    file_store = InMemoryFileStore()
    return EventStream(sid='default', file_store=file_store)


def main():
    """Main function to run the evaluation."""
    args = parse_args()

    logger.info(f"\n{'='*80}\nStarting VisualCodeBench Evaluation\n{'='*80}")
    logger.info(f'Agent: {args.agent_cls}')
    logger.info(f'Model: {args.llm_config}')
    logger.info(f'Max iterations: {args.max_iterations}')
    logger.info(f'Eval limit: {args.eval_n_limit}')
    logger.info(f'Num workers: {args.eval_num_workers}\n')

    # Step 1: Download the entire repository once
    logger.info('Downloading repository...')
    download_repository()

    # Setup config
    llm_config_arg = get_llm_config_arg(args.llm_config)
    if llm_config_arg is None:
        logger.error(f'Could not find LLM config: {args.llm_config}')
        raise ValueError(f'Could not find LLM config: {args.llm_config}')

    config = get_config(args.agent_cls, llm_config_arg, args.max_iterations)
    logger.info('Configuration created.')

    # Load dataset
    logger.info('Loading dataset from HuggingFace...')
    dataset = load_visualcodebench(limit=args.eval_n_limit)
    logger.info(f'Dataset loaded with {len(dataset)} examples.')

    if len(dataset) == 0:
        logger.warning('No tasks to evaluate. Exiting.')
        return

    # Process tasks
    results = []
    if args.eval_num_workers > 1:
        logger.info(f'Running evaluation with {args.eval_num_workers} workers...')
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.eval_num_workers) as executor:
            futures = [
                executor.submit(run_task, task, config, args.agent_cls)
                for task in dataset
            ]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                logger.info(f'Progress: {i}/{len(dataset)} tasks completed')
    else:
        logger.info('Running evaluation sequentially...')
        for i, task in enumerate(dataset, 1):
            result = run_task(task, config, args.agent_cls)
            results.append(result)
            logger.info(f'Progress: {i}/{len(dataset)} tasks completed')

    # Calculate and display summary
    total = len(results)
    passed = sum(1 for r in results if r.get('passed'))
    failed = total - passed

    logger.info(f"\n{'='*80}\nEvaluation Summary\n{'='*80}")
    logger.info(f'Total tasks: {total}')
    logger.info(f'Passed: {passed} ({(passed/total*100) if total > 0 else 0:.1f}%)')
    logger.info(f'Failed: {failed} ({(failed/total*100) if total > 0 else 0:.1f}%)')

    # Save summary and results to results file
    summary = {
        'total': total,
        'passed': passed,
        'failed': failed,
        'pass_rate': (passed / total) if total > 0 else 0,
        'model_config': args.llm_config,
        'agent_cls': args.agent_cls,
    }

    results_file = os.path.join(HTML_FILES_DIR, 'results.json')

    # Ensure the HTML_FILES_DIR exists
    os.makedirs(HTML_FILES_DIR, exist_ok=True)

    try:
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                data = json.load(f)
                if 'results' not in data:
                    data['results'] = []
        else:
            data = {'results': []}

        # Append the new results
        data['results'].extend(results)
        logger.info(f'Appended {len(results)} results to the results file.')

        # Add or update the summary
        data['summary'] = summary

        # Write back to the results file
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=4)

        logger.info(f'Summary and results saved successfully to {results_file}.')

    except Exception as e:
        logger.error(f'Error saving summary: {e}', exc_info=True)

    logger.info(f"\n{'='*80}\nEvaluation Complete\n{'='*80}")


if __name__ == '__main__':
    main()
