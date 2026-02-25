import json
import threading
import time
import itertools

import numpy as np
import requests
from requests.exceptions import ConnectionError
import socketio

import csv
import os
from typing import Dict, List, Any, Generator, Set, FrozenSet, Tuple
import multiprocessing
import queue

from trusted_authority import TrustedAuthority
import federated_server
import run_multiple_clients

NUM_PARALLEL_EXECUTIONS = 4
GRID_SEARCH_CONFIG_PATH = 'grid_search_config.json'
VERBOSE_DUPLICATE_CHECK = False

# List of edge/vision models that require a fixed input image size of 224x224
EDGE_NET_MODELS = [
    'ResNet', 'ShuffleNet', 'MobileNet', 'EfficientNet',
    'ConvNeXt', 'MobileViT', 'EdgeNeXt', 'EfficientFormer',
    'ViT', 'DeiT'
]


def save_best_model_if_needed(run_summary: Dict, best_weights: List[np.ndarray], lock: multiprocessing.Lock,
                              models_dir: str, config: Dict) -> None:
    """
    Checks whether the current model has a better F1 score.
    If so, saves the model and copies the private key used to encrypt it
    into a dedicated subfolder for that model family.
    """
    if best_weights is None or run_summary is None:
        return

    model_name = run_summary.get('model_name', 'unknown')
    current_f1 = run_summary.get('best_f1', 0.0)
    ta_port = config['ta_port']  # Retrieve the port used by the TA in this run

    # Create the base directory and define the global scores file
    os.makedirs(models_dir, exist_ok=True)
    scores_file = os.path.join(models_dir, 'best_scores.json')

    lock.acquire()
    try:
        if os.path.exists(scores_file):
            with open(scores_file, 'r') as f:
                best_scores = json.load(f)
        else:
            best_scores = {}

        historical_best_f1 = best_scores.get(model_name, -1.0)

        if current_f1 > historical_best_f1:
            # Update the record in the json file (in the root models folder)
            best_scores[model_name] = current_f1
            with open(scores_file, 'w') as f:
                json.dump(best_scores, f, indent=4)

            # Create the specific subfolder for the model (e.g., saved_models/ResNet18)
            specific_model_dir = os.path.join(models_dir, model_name)
            os.makedirs(specific_model_dir, exist_ok=True)

            # Save the encrypted model INSIDE THE SUBFOLDER
            model_filename = os.path.join(specific_model_dir, f"best_{model_name}.pkl")
            import pickle
            with open(model_filename, 'wb') as f:
                pickle.dump(best_weights, f)

            # Copy the private key INTO THE SUBFOLDER
            import shutil
            temp_key_path = f'temp_keys/ta_privkey_port_{ta_port}.pkl'
            final_key_path = os.path.join(specific_model_dir, f"best_{model_name}_key.pkl")

            if os.path.exists(temp_key_path):
                shutil.copy(temp_key_path, final_key_path)

            print(
                f"\n[*] NEW RECORD for {model_name}! F1: {current_f1:.4f}. Data saved in: {specific_model_dir}/\n")
    finally:
        lock.release()


def get_config_fingerprint(config: Dict, keys: Set[str]) -> FrozenSet[Tuple[str, str]]:
    """Builds a hashable fingerprint from a config dict using only the specified keys."""
    fingerprint_items = []
    for key in sorted(list(keys)):
        if key in config and config[key] is not None and config[key] != '':
            fingerprint_items.append((key, str(config[key])))
    return frozenset(fingerprint_items)


def wait_for_server_ready(url: str, timeout: int = 60) -> bool:
    """Polls the server URL until it responds with HTTP 200 or the timeout expires."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
        except ConnectionError:
            time.sleep(1)
    return False


def load_json(filename: str) -> Dict:
    with open(filename) as f:
        return json.load(f)


def append_results_to_csv(csv_path: str, run_summary: Dict, lock: multiprocessing.Lock) -> None:
    """
    Appends a run summary dict to a shared CSV file in a process-safe manner.
    Merges new keys with existing headers if the file already exists.
    """
    lock.acquire()
    try:
        all_keys = set(run_summary.keys())
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                all_keys.update(header)

        file_is_empty = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            fieldnames = sorted(list(all_keys))
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if file_is_empty:
                writer.writeheader()
            writer.writerow(run_summary)
    finally:
        lock.release()


def generate_configurations(base_config: Dict, search_space: Dict) -> Generator[Dict[str, Any], None, None]:
    """Yields all combinations of hyperparameters from the search space merged into the base config."""
    if not search_space:
        yield base_config
        return
    keys, values = zip(*search_space.items())
    for combo in itertools.product(*values):
        config = base_config.copy()
        config.update(dict(zip(keys, combo)))
        yield config


def start_server_thread(config: Dict, server_instance_ref: List) -> None:
    """Instantiates and runs the federated server; stores the instance for later access."""
    worker_id = config.get('worker_id', 'N/A')
    port = config['port']
    print(f"[Worker {worker_id}] Starting server thread on port {port}...")
    try:
        server = federated_server.FederatedServer(config)
        server_instance_ref.append(server)
        server.run()
    except Exception as e:
        print(f"!!!!!!!!!! [Worker {worker_id}] SERVER THREAD CRASHED on port {port} !!!!!!!!!!!")
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print(f"[Worker {worker_id}] Server thread has finished.")


def start_ta_thread(config: Dict, ta_instance_ref: List) -> None:
    """Instantiates and runs the Trusted Authority; stores the instance for later access."""
    print(f"[Worker {config.get('worker_id', 'N/A')}] Starting TA thread on port {config['ta_port']}...")
    ta = TrustedAuthority(host=config['ip_address'], port=config['ta_port'])
    ta_instance_ref.append(ta)
    ta.run()
    print(f"[Worker {config.get('worker_id', 'N/A')}] TA thread has finished.")


def _apply_config_defaults(config: Dict, model_name: str = None) -> None:
    """
    Applies default values to a config in-place:
    - Sets fedprox_mu to 0.0 for non-FedProx algorithms.
    - Sets image_size to 224 for models that require it.
    """
    if config.get('aggregation_algorithm') != "FedProx":
        config['fedprox_mu'] = 0.0

    name = model_name or config.get('model_name', '')
    if any(net in name for net in EDGE_NET_MODELS):
        config['image_size'] = 224


def run_grid_search_worker(
        worker_id: int,
        task_queue: multiprocessing.JoinableQueue,
        base_port: int,
        csv_lock: multiprocessing.Lock
) -> None:
    """
    Worker process that continuously dequeues configurations and runs a full
    federated learning experiment for each one, including TA, server, and clients.
    Terminates upon receiving a sentinel (None) value from the queue.
    """
    print(f"--- Starting Grid Search Worker {worker_id} ---")
    while True:
        try:
            config = task_queue.get()
            if config is None:
                print(f"--- Worker {worker_id} received sentinel. Shutting down. ---")
                break

            dataset_name = config['dataset_name']
            run_identifier = f"{dataset_name}_run_{worker_id}_{config['model_name']}"

            # Set up per-run directories for data splits, logs, and metrics
            worker_splitting_dir = os.path.join(config['base_split_data_path'], run_identifier)
            worker_log_dir = os.path.join(config['base_log_path'], run_identifier)
            worker_metrics_dir = os.path.join(config['base_csv_path'], "runs")

            os.makedirs(worker_splitting_dir, exist_ok=True)
            os.makedirs(worker_log_dir, exist_ok=True)
            os.makedirs(worker_metrics_dir, exist_ok=True)

            print("\n" + "#" * 80)
            print(f"### [Worker {worker_id}] DEQUEUED NEW CONFIG FOR: {dataset_name} | {config['model_name']} ###")

            # Assign worker-specific ports and paths to avoid conflicts between parallel workers
            config['worker_id'] = worker_id
            config['port'] = base_port + worker_id * 2
            config['ta_port'] = base_port + worker_id * 2 + 1
            config['splitting_dir'] = worker_splitting_dir
            config['log_dir'] = worker_log_dir
            config['run_metrics_output_path'] = worker_metrics_dir
            config["MIN_NUM_WORKERS"] = int(config['num_clients'] * config['models_percentage'])

            # Start the Trusted Authority first, then wait briefly before launching the server
            ta_instance_ref = []
            server_instance_ref = []

            # Check if no_encryption is enabled
            ta_thread = None
            if config.get('encryption_mode') != 'no_encryption':
                ta_thread = threading.Thread(target=start_ta_thread, args=(config, ta_instance_ref))
                ta_thread.start()
                time.sleep(3)
            else:
                print(f"[Worker {worker_id}] Encryption disabled. Skipping TA start.")


            server_thread = threading.Thread(target=start_server_thread, args=(config, server_instance_ref))
            server_thread.start()

            server_url = f"http://{config['ip_address']}:{config['port']}"
            if wait_for_server_ready(server_url):
                run_multiple_clients.main(config)
            else:
                print(f"[Worker {worker_id}] Server failed to start. Skipping.")

            server_thread.join()

            # Gracefully shut down the Trusted Authority via socket
            try:
                sio_client = socketio.Client(reconnection=False)
                sio_client.connect(f"http://{config['ip_address']}:{config['ta_port']}", transports=['websocket'])
                sio_client.emit('shutdown_ta')
                time.sleep(1)
                sio_client.disconnect()
            except Exception as e:
                print(f"[Worker {worker_id}] Could not shutdown TA: {e}")

            # Shutdown TA
            if ta_thread is not None:
                ta_thread.join(timeout=15)

            # Collect and persist run results if available
            if server_instance_ref:
                run_summary = server_instance_ref[0].aggregator.get_run_summary()
                if run_summary:
                    # Save metrics in shared csv
                    append_results_to_csv(config['shared_csv_path'], run_summary, csv_lock)

                    # Save model and private key
                    best_weights = server_instance_ref[0].aggregator.best_model_weights
                    models_dir = config.get('base_model_output_path', 'saved_models')
                    save_best_model_if_needed(run_summary, best_weights, csv_lock, models_dir, config)

            time.sleep(1)

        except queue.Empty:
            continue
        finally:
            task_queue.task_done()


def main():
    base_grid_config = load_json(GRID_SEARCH_CONFIG_PATH)
    csv_lock = multiprocessing.Lock()
    task_queue = multiprocessing.JoinableQueue()

    # Ensure all base output directories exist
    base_csv_path = base_grid_config['base_csv_path']
    os.makedirs(base_csv_path, exist_ok=True)
    os.makedirs(base_grid_config.get('base_log_path', 'logs'), exist_ok=True)
    os.makedirs(base_grid_config.get('base_split_data_path', 'run_dataset'), exist_ok=True)

    shared_csv_path = os.path.join(base_csv_path, 'federated_grid_search_results.csv')
    os.makedirs(os.path.dirname(shared_csv_path), exist_ok=True)

    # Collect all possible search keys across common and model-specific spaces
    executed_fingerprints: Set[FrozenSet[Tuple[str, str]]] = set()
    common_search_space = base_grid_config['common_search_space']
    model_specific_search_space = base_grid_config['model_specific_search_space']

    all_possible_search_keys = set(common_search_space.keys())
    for model_params in model_specific_search_space.values():
        all_possible_search_keys.update(model_params.keys())
    all_possible_search_keys.update(['dataset_name', 'model_name'])

    # Load fingerprints of previously completed runs to enable resume-from-checkpoint behavior
    if os.path.exists(shared_csv_path) and os.path.getsize(shared_csv_path) > 0:
        with open(shared_csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name_from_row = row.get('model_name', '')
                _apply_config_defaults(row, model_name=model_name_from_row)
                fingerprint = get_config_fingerprint(row, all_possible_search_keys)
                executed_fingerprints.add(fingerprint)

    print(f"Loaded {len(executed_fingerprints)} fingerprints of previously executed configurations.")

    fixed_params = {
        k: v for k, v in base_grid_config.items()
        if k not in ['datasets', 'common_search_space', 'model_specific_search_space']
    }
    fixed_params['shared_csv_path'] = shared_csv_path

    configs_to_run_count = 0
    total_generated_configs = 0

    # Enumerate all dataset/model/hyperparameter combinations and enqueue new ones
    for dataset_info in base_grid_config['datasets']:
        for model_name, specific_params in model_specific_search_space.items():
            model_base_config = fixed_params.copy()
            model_base_config.update({
                'dataset_name': dataset_info['name'],
                'dataset_path': dataset_info['path'],
                'num_classes': dataset_info['num_classes'],
                'model_name': model_name
            })
            current_search_space = {**common_search_space, **specific_params}

            for hyper_config in generate_configurations(model_base_config, current_search_space):
                total_generated_configs += 1
                _apply_config_defaults(hyper_config, model_name=model_name)

                fingerprint = get_config_fingerprint(hyper_config, all_possible_search_keys)
                if fingerprint in executed_fingerprints:
                    continue

                executed_fingerprints.add(fingerprint)
                task_queue.put(hyper_config)
                configs_to_run_count += 1

    print("=" * 80)
    print(f"Generated {total_generated_configs} total configurations.")
    print(f"Skipped {total_generated_configs - configs_to_run_count} already executed or redundant configurations.")
    print(f"Adding {configs_to_run_count} new unique configurations to the execution queue.")
    print("=" * 80)

    if task_queue.empty():
        print("=== No new configurations to run. Exiting. ===")
        return

    # Launch parallel worker processes
    processes = []
    print(f"\n=== Starting {NUM_PARALLEL_EXECUTIONS} Grid Search workers ===")
    base_port = base_grid_config['port']
    for i in range(NUM_PARALLEL_EXECUTIONS):
        process = multiprocessing.Process(target=run_grid_search_worker, args=(i, task_queue, base_port, csv_lock))
        processes.append(process)
        process.start()

    task_queue.join()

    # Send one sentinel per worker to signal graceful shutdown
    for _ in range(NUM_PARALLEL_EXECUTIONS):
        task_queue.put(None)
    for process in processes:
        process.join()

    print("\n=== All parallel executions have finished. ===")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()