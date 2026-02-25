import json
import threading
import time
import itertools
import pandas as pd
import requests
from requests.exceptions import ConnectionError
import os
import multiprocessing
import queue

import federated_server
import run_multiple_clients

# --- GLOBAL CONFIGURATIONS ---
NUM_PARALLEL_EXECUTIONS = 2  # Change this number based on available cores/RAM
GRID_SEARCH_CONFIG_PATH = 'grid_search_config.json'


def load_json(filename: str) -> dict:
    with open(filename) as f:
        return json.load(f)


def is_already_tested(config: dict, global_csv_path: str) -> bool:
    """
    Checks if a specific configuration already exists in the summary CSV file.
    If it finds an incomplete pruning configuration (only Pre-Pruning completed),
    it removes the orphan rows from the CSV and returns False to restart it from scratch.
    """
    if not os.path.exists(global_csv_path):
        return False

    try:
        df = pd.read_csv(global_csv_path)
    except Exception:
        return False

    if df.empty:
        return False

    # Key hyperparameters defining a unique configuration
    keys_to_match = [
        'model_name', 'global_epoch', 'local_epoch', 'num_clients', 'learning_rate', 'models_percentage',
        'aggregation_algorithm', 'enable_pruning', 'pruning_threshold', 'num_custom_layers'
    ]

    # Create a boolean mask to find rows matching the configuration exactly
    mask = pd.Series([True] * len(df))
    for k in keys_to_match:
        if k in config and k in df.columns:
            # Compare as strings to avoid float/int casting issues
            mask &= (df[k].astype(str) == str(config[k]))

    # Get the indices of the rows matching this configuration
    matched_indices = df[mask].index

    if len(matched_indices) == 0:
        return False

    matched_df = df.loc[matched_indices]

    if config.get('enable_pruning', False):
        # Pruning is enabled, check if the Post-Pruning phase was completed
        if 'execution_phase' in matched_df.columns:
            has_post_pruning = (matched_df['execution_phase'] == 'Post-Pruning').any()

            if has_post_pruning:
                # Execution completed successfully previously
                return True
            else:
                # CLEAN & RESTART: Interrupted halfway (only Pre-Pruning exists).
                # Remove orphan rows from the DataFrame and overwrite the CSV.
                print(
                    f"[CLEAN & RESTART] Incomplete execution found for {config['model_name']} (LR: {config['learning_rate']}). Cleaning CSV..."
                )
                df_cleaned = df.drop(index=matched_indices)
                df_cleaned.to_csv(global_csv_path, index=False)
                return False
        else:
            return False
    else:
        # If pruning is not enabled, the existence of the row (Standard) is enough
        return True


def generate_configurations(base_config: dict, search_space: dict):
    """Generates the Cartesian product of all hyperparameters."""
    if not search_space:
        yield base_config
        return
    keys, values = zip(*search_space.items())
    for combo in itertools.product(*values):
        config = base_config.copy()
        config.update(dict(zip(keys, combo)))
        yield config


def wait_for_server_ready(url: str, timeout: int = 60) -> bool:
    """Waits for the Flask server to be ready to receive connections."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except ConnectionError:
            time.sleep(1)
    return False


def start_server_thread(config: dict) -> None:
    """Starts the federated server in the current thread."""
    worker_id = config.get('worker_id', 'N/A')
    port = config['port']
    print(f"[Worker {worker_id}] Starting server on port {port}...")
    try:
        server = federated_server.FederatedServer(config)
        server.run()
    except Exception as e:
        print(f"!!!!!!!!!! [Worker {worker_id}] SERVER CRASH ON PORT {port} !!!!!!!!!!!")
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print(f"[Worker {worker_id}] Server terminated.")


def run_grid_search_worker(worker_id: int, task_queue: multiprocessing.JoinableQueue, base_port: int,
                           csv_lock: multiprocessing.Lock) -> None:
    """
    Worker process that fetches configurations from the queue and starts the entire FL cycle (Server + Client).
    """
    print(f"--- Starting Grid Search Worker {worker_id} ---")
    while True:
        try:
            config = task_queue.get()
            if config is None:
                print(f"--- Worker {worker_id} received stop signal. Shutting down. ---")
                task_queue.task_done()
                break

            dataset_name = config['dataset_name']
            model_name = config['model_name']

            print("\n" + "#" * 80)
            print(f"### [Worker {worker_id}] STARTING CONFIGURATION: {dataset_name} | {model_name} ###")

            # Assign unique ports and paths to isolate parallel executions
            config['worker_id'] = worker_id
            config['port'] = base_port + (worker_id * 2)
            config['ta_port'] = base_port + (worker_id * 2) + 1
            config['MIN_NUM_WORKERS'] = config['num_clients']

            # Inject the Lock into the config so the Aggregator can use it to write to the CSV
            config['csv_lock'] = csv_lock

            run_identifier = f"{dataset_name}_{model_name}_w{worker_id}"
            config['log_dir'] = os.path.join(config['base_log_path'], run_identifier)
            config['run_metrics_output_path'] = os.path.join(config['base_csv_path'], dataset_name)
            config['splitting_dir'] = os.path.join(config['base_split_data_path'], run_identifier)

            os.makedirs(config['log_dir'], exist_ok=True)
            os.makedirs(config['splitting_dir'], exist_ok=True)

            server_thread = threading.Thread(target=start_server_thread, args=(config,), daemon=True)
            server_thread.start()

            server_url = f"http://{config['ip_address']}:{config['port']}/"
            if wait_for_server_ready(server_url):
                run_multiple_clients.main(config)
            else:
                print(f"[Worker {worker_id}] Unable to contact server. Skipping configuration.")

            server_thread.join()
            time.sleep(1)

        except queue.Empty:
            continue
        finally:
            task_queue.task_done()


def main():
    base_grid_config = load_json(GRID_SEARCH_CONFIG_PATH)
    csv_lock = multiprocessing.Lock()
    task_queue = multiprocessing.JoinableQueue()

    base_csv_path = base_grid_config.get('base_csv_path', 'csv')
    os.makedirs(base_csv_path, exist_ok=True)
    os.makedirs(base_grid_config.get('base_log_path', 'logs'), exist_ok=True)
    os.makedirs(base_grid_config.get('base_split_data_path', 'run_dataset'), exist_ok=True)

    global_csv_path = os.path.join(base_csv_path, "global_grid_search_summary.csv")

    common_space = base_grid_config['common_search_space']
    model_space = base_grid_config['model_specific_search_space']

    fixed_params = {k: v for k, v in base_grid_config.items() if
                    k not in ['datasets', 'common_search_space', 'model_specific_search_space']}

    configs_to_run_count = 0
    total_generated_configs = 0

    # Populate the queue
    for dataset_info in base_grid_config['datasets']:
        for model_name, specific_params in model_space.items():
            model_base_config = fixed_params.copy()
            model_base_config.update({
                'dataset_name': dataset_info['name'],
                'dataset_path': dataset_info['path'],
                'num_classes': dataset_info['num_classes'],
                'model_name': model_name
            })

            current_search_space = {**common_space, **specific_params}

            for hyper_config in generate_configurations(model_base_config, current_search_space):
                total_generated_configs += 1

                # SKIP CONTROL: Check if this config is already present in the global csv
                if is_already_tested(hyper_config, global_csv_path):
                    continue

                task_queue.put(hyper_config)
                configs_to_run_count += 1

    print("=" * 80)
    print(f"Total configurations explored: {total_generated_configs}")
    print(f"Configurations skipped (already executed): {total_generated_configs - configs_to_run_count}")
    print(f"Configurations queued for execution: {configs_to_run_count}")
    print("=" * 80)

    if task_queue.empty():
        print("=== No new configurations to execute. Exiting. ===")
        return

    # Start Parallel Workers
    processes = []
    print(f"\n=== Starting {NUM_PARALLEL_EXECUTIONS} parallel workers ===")
    base_port = base_grid_config['port']

    for i in range(NUM_PARALLEL_EXECUTIONS):
        process = multiprocessing.Process(target=run_grid_search_worker, args=(i, task_queue, base_port, csv_lock))
        processes.append(process)
        process.start()

    task_queue.join()

    # Poison pills to shut down processes cleanly
    for _ in range(NUM_PARALLEL_EXECUTIONS):
        task_queue.put(None)

    for process in processes:
        process.join()

    print("\n=== Parallel execution completed. ===")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()