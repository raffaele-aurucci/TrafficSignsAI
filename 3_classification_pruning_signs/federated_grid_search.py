#TODO: FIX MODELS_PERCENTAGE = 0.5

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
import pickle
import shutil
from typing import Dict, Set, FrozenSet, Tuple

import federated_server
import run_multiple_clients

# Configuration Constants
NUM_PARALLEL_EXECUTIONS = 2
GRID_SEARCH_CONFIG_PATH = 'grid_search_config.json'

# Keys used to uniquely identify a specific experiment configuration
MATCH_KEYS = [
    'model_name', 'global_epoch', 'local_epoch', 'num_clients', 'learning_rate',
    'models_percentage', 'aggregation_algorithm', 'pruning_threshold', 'num_custom_layers'
]


# ─────────────────────────────────────────────────────────────────────────────
# FINGERPRINT HELPERS (O(1) lookup instead of O(n) per config)
# ─────────────────────────────────────────────────────────────────────────────

def make_config_fingerprint(config: dict) -> FrozenSet[Tuple[str, str]]:
    """Creates a hashable fingerprint from the config using only MATCH_KEYS."""
    return frozenset((k, str(config[k])) for k in MATCH_KEYS if k in config)


def build_fingerprint_sets(global_csv_path: str) -> Tuple[Set, Set]:
    """
    Reads the CSV ONCE and builds two fingerprint sets:
      - completed : configs with at least one 'Post-Pruning' row -> skip
      - pre_only  : configs with only 'Pre-Pruning' -> incomplete, needs restart

    This replaces is_already_tested() in the main loop,
    reducing duplicate-check cost from O(n) to O(1) per config.
    """
    completed: Set[FrozenSet] = set()
    pre_only: Set[FrozenSet] = set()

    if not os.path.exists(global_csv_path):
        return completed, pre_only

    try:
        df = pd.read_csv(global_csv_path, dtype=str)
    except Exception:
        return completed, pre_only

    if df.empty or 'execution_phase' not in df.columns:
        return completed, pre_only

    for _, row in df.iterrows():
        fp = frozenset(
            (k, str(row[k])) for k in MATCH_KEYS
            if k in row and pd.notna(row[k]) and str(row[k]) != ''
        )
        phase = str(row.get('execution_phase', ''))
        if phase == 'Post-Pruning':
            completed.add(fp)
        elif phase == 'Pre-Pruning':
            pre_only.add(fp)

    # Incomplete = Pre-Pruning only (missing the Post-Pruning entry)
    incomplete = pre_only - completed
    return completed, incomplete


# ─────────────────────────────────────────────────────────────────────────────
# CLEANUP FOR INCOMPLETE RUNS
# ─────────────────────────────────────────────────────────────────────────────

def cleanup_incomplete_run(config: dict, global_csv_path: str,
                            models_dir: str, lock: multiprocessing.Lock) -> None:
    """
    Handles an incomplete run (only Pre-Pruning present in CSV):
      1. Removes orphan rows from the CSV.
      2. If the orphan run had a better F1 than the saved record,
         deletes weights/keys from disk and cleans best_scores.json.
    Called once per incomplete config before requeueing.
    """
    model_name = config.get('model_name', 'unknown')
    print(f"[CLEAN & RESTART] Incomplete run (Pre-Pruning only) for {model_name} "
          f"(LR: {config.get('learning_rate')}).")

    try:
        df = pd.read_csv(global_csv_path, dtype=str)
    except Exception:
        return

    # Identify orphan rows
    mask = pd.Series([True] * len(df))
    for k in MATCH_KEYS:
        if k in config and k in df.columns:
            mask &= (df[k].astype(str) == str(config[k]))
    matched_indices = df[mask].index

    if len(matched_indices) == 0:
        return

    orphan_rows = df.loc[matched_indices].copy()

    # 1. Remove from CSV
    df.drop(index=matched_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(global_csv_path, index=False)

    # 2. JSON and Disk Cleanup
    if not models_dir:
        return

    scores_file = os.path.join(models_dir, 'best_scores.json')

    def _do_cleanup():
        if not os.path.exists(scores_file):
            return
        try:
            with open(scores_file, 'r') as f:
                best_scores = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        if model_name not in best_scores:
            return

        phase = "Pre-Pruning"
        safe_phase = phase.replace(" ", "_")
        phase_record = best_scores[model_name].get(phase, {})
        if not isinstance(phase_record, dict):
            return

        orphan_f1 = 0.0
        if not orphan_rows.empty and 'best_f1' in orphan_rows.columns:
            orphan_f1 = pd.to_numeric(orphan_rows['best_f1'], errors='coerce').max()
            orphan_f1 = orphan_f1 if not pd.isna(orphan_f1) else 0.0

        saved_f1 = phase_record.get('metrics', {}).get('best_f1', -1.0)

        # If existing record is better, don't delete files
        if orphan_f1 <= saved_f1:
            print(f"[CLEAN & RESTART] Saved best F1 ({saved_f1:.4f}) >= orphan F1 ({orphan_f1:.4f}) "
                  f"for {model_name}/{phase}. Skipping file deletion.")
            return

        weights_path = phase_record.get('model_weights_path', '')
        if weights_path and os.path.exists(weights_path):
            os.remove(weights_path)
            print(f"[CLEAN & RESTART] Deleted orphaned weights: {weights_path}")

        key_path = os.path.join(
            os.path.dirname(weights_path),
            f"best_{model_name}_{safe_phase}_key.pkl"
        )
        if os.path.exists(key_path):
            os.remove(key_path)
            print(f"[CLEAN & RESTART] Deleted orphaned key: {key_path}")

        best_scores[model_name].pop(phase, None)
        if not best_scores[model_name]:
            del best_scores[model_name]

        with open(scores_file, 'w') as f:
            json.dump(best_scores, f, indent=4)
        print(f"[CLEAN & RESTART] Cleaned JSON entry for {model_name} / {phase}.")

    with lock:
        _do_cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# BEST MODEL SAVING
# ─────────────────────────────────────────────────────────────────────────────

def save_best_model_if_needed(run_summary: dict, best_weights: list, lock: multiprocessing.Lock,
                              models_dir: str, config: dict, phase: str) -> None:
    """
    Checks if current model has a better F1 for its phase.
    If yes, saves weights (.pkl), copies private key, and updates best_scores.json.
    """
    if best_weights is None or run_summary is None:
        return

    model_name = run_summary.get('model_name', 'unknown')
    current_f1 = run_summary.get('best_f1', run_summary.get('f1_score', 0.0))

    safe_phase = phase.replace(" ", "_")
    ta_port = config.get('ta_port')

    metric_keys = {'best_f1', 'best_acc', 'best_prec', 'best_recall', 'best_loss',
                   'best_round', 'total_duration_sec', 'execution_phase', 'round_dataframe_path'}

    config_keys_to_exclude = {'worker_id', 'port', 'ta_port', 'csv_lock', 'log_dir',
                               'run_metrics_output_path', 'splitting_dir', 'dataset_path',
                               'MIN_NUM_WORKERS', 'ip_address', 'is_post_pruning_run'}

    os.makedirs(models_dir, exist_ok=True)
    scores_file = os.path.join(models_dir, 'best_scores.json')

    lock.acquire()
    try:
        best_scores = {}
        if os.path.exists(scores_file):
            with open(scores_file, 'r') as f:
                try:
                    best_scores = json.load(f)
                except json.JSONDecodeError:
                    best_scores = {}

        if model_name not in best_scores or not isinstance(best_scores[model_name], dict):
            best_scores[model_name] = {}

        phase_record = best_scores[model_name].get(phase, {})
        historical_best_f1 = phase_record.get('metrics', {}).get('best_f1', -1.0) \
            if isinstance(phase_record, dict) else float(phase_record)

        if current_f1 > historical_best_f1:
            specific_model_dir = os.path.join(models_dir, model_name)
            os.makedirs(specific_model_dir, exist_ok=True)
            model_filename = os.path.join(specific_model_dir, f"best_{model_name}_{safe_phase}.pkl")

            metrics = {k: v for k, v in run_summary.items() if k in metric_keys}
            grid_cfg = {k: v for k, v in run_summary.items()
                        if k not in metric_keys and k not in config_keys_to_exclude}

            best_scores[model_name][phase] = {
                'metrics': metrics,
                'grid_search_config': grid_cfg,
                'model_weights_path': model_filename,
            }

            with open(scores_file, 'w') as f:
                json.dump(best_scores, f, indent=4)

            with open(model_filename, 'wb') as f:
                pickle.dump(best_weights, f)

            if ta_port:
                temp_key_path = f'temp_keys/ta_privkey_port_{ta_port}.pkl'
                if os.path.exists(temp_key_path):
                    final_key_path = os.path.join(specific_model_dir, f"best_{model_name}_{safe_phase}_key.pkl")
                    shutil.copy(temp_key_path, final_key_path)

            print(f"\n[*] NEW RECORD ({phase}) for {model_name}! F1: {current_f1:.4f}. "
                  f"Saved in: {specific_model_dir}/\n")
    finally:
        lock.release()


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_json(filename: str) -> dict:
    with open(filename) as f:
        return json.load(f)


def generate_configurations(base_config: dict, search_space: dict):
    """Generates Cartesian product of all hyperparameters."""
    if not search_space:
        yield base_config
        return
    keys, values = zip(*search_space.items())
    for combo in itertools.product(*values):
        config = base_config.copy()
        config.update(dict(zip(keys, combo)))
        yield config


def wait_for_server_ready(url: str, timeout: int = 60) -> bool:
    """Waits for the Flask server to be ready for connections."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except ConnectionError:
            time.sleep(1)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# SERVER THREAD
# ─────────────────────────────────────────────────────────────────────────────

def start_server_thread(config: dict, server_instance_ref: list) -> None:
    """
    Starts the federated server and captures the instance.
    Monkey-patches _reset_for_new_training to preserve the Pre-Pruning aggregator
    reference before it is replaced by the Post-Pruning one.
    """
    worker_id = config.get('worker_id', 'N/A')
    port = config['port']
    print(f"[Worker {worker_id}] Starting server on port {port}...")
    try:
        server = federated_server.FederatedServer(config)

        original_reset = server._reset_for_new_training
        def _patched_reset():
            server.pre_pruning_aggregator = server.aggregator
            original_reset()
        server._reset_for_new_training = _patched_reset

        server_instance_ref.append(server)
        server.run()
    except Exception as e:
        print(f"!!!!!!!!!! [Worker {worker_id}] SERVER CRASH ON PORT {port} !!!!!!!!!!!")
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print(f"[Worker {worker_id}] Server terminated.")


# ─────────────────────────────────────────────────────────────────────────────
# WORKER PROCESS
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_search_worker(worker_id: int, task_queue: multiprocessing.JoinableQueue, base_port: int,
                           csv_lock: multiprocessing.Lock) -> None:
    """
    Worker process that pulls configurations from the queue and starts
     the full FL cycle (Server + Clients) for each.
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

            config['worker_id'] = worker_id
            config['port'] = base_port + (worker_id * 2)
            config['ta_port'] = base_port + (worker_id * 2) + 1
            config["MIN_NUM_WORKERS"] = int(config['num_clients'] * config['models_percentage'])
            config['csv_lock'] = csv_lock

            run_identifier = f"{dataset_name}_{model_name}_w{worker_id}"
            config['log_dir'] = os.path.join(config['base_log_path'], run_identifier)
            config['run_metrics_output_path'] = os.path.join(config['base_csv_path'], dataset_name)
            config['splitting_dir'] = os.path.join(config['base_split_data_path'], run_identifier)

            os.makedirs(config['log_dir'], exist_ok=True)
            os.makedirs(config['splitting_dir'], exist_ok=True)

            server_instance_ref = []
            server_thread = threading.Thread(target=start_server_thread, args=(config, server_instance_ref),
                                             daemon=True)
            server_thread.start()

            server_url = f"http://{config['ip_address']}:{config['port']}/"
            if wait_for_server_ready(server_url):
                run_multiple_clients.main(config)
            else:
                print(f"[Worker {worker_id}] Unable to contact server. Skipping configuration.")

            server_thread.join()

            # ── WEIGHT SAVING (PRE & POST PRUNING) ──────────────────────────
            if server_instance_ref:
                server_obj = server_instance_ref[0]
                models_dir = config.get('base_model_output_path', 'saved_models')

                # Pre-Pruning: aggregator saved via monkey-patch before reset
                pre_agg = getattr(server_obj, 'pre_pruning_aggregator', None)
                if pre_agg and pre_agg.best_model_weights is not None:
                    save_best_model_if_needed(pre_agg.run_summary, pre_agg.best_model_weights,
                                              csv_lock, models_dir, config, phase="Pre-Pruning")

                # Post-Pruning (or Standard): aggregator active at end of training
                post_agg = server_obj.aggregator
                if post_agg and post_agg.best_model_weights is not None:
                    save_best_model_if_needed(post_agg.run_summary, post_agg.best_model_weights,
                                              csv_lock, models_dir, config, phase="Post-Pruning")
            # ───────────────────────────────────────────────────────────────

            time.sleep(1)

        except queue.Empty:
            continue
        finally:
            task_queue.task_done()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

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

    models_dir = base_grid_config.get('base_model_output_path', 'saved_models')
    os.makedirs(models_dir, exist_ok=True)

    # ── KEY OPTIMIZATION ────────────────────────────────────────────────────
    # Reads the CSV once to build fingerprint sets.
    print("Building fingerprint sets from existing CSV (one-time read)...")
    completed_fps, incomplete_fps = build_fingerprint_sets(global_csv_path)
    print(f"  Completed configs (Post-Pruning present): {len(completed_fps)}")
    print(f"  Incomplete configs (Pre-Pruning only):    {len(incomplete_fps)}")
    # ───────────────────────────────────────────────────────────────────────

    configs_to_run_count = 0
    total_generated_configs = 0

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
                fp = make_config_fingerprint(hyper_config)

                # 1) Already completed (Pre + Post) -> Skip O(1)
                if fp in completed_fps:
                    continue

                # 2) Pre-Pruning only -> Cleanup and Requeue
                if fp in incomplete_fps:
                    cleanup_incomplete_run(hyper_config, global_csv_path, models_dir, csv_lock)
                    incomplete_fps.discard(fp)   # Prevent double cleanup if duplicates exist

                task_queue.put(hyper_config)
                configs_to_run_count += 1

    print("=" * 80)
    print(f"Total configurations explored:             {total_generated_configs}")
    print(f"Configurations skipped (already executed): {total_generated_configs - configs_to_run_count}")
    print(f"Configurations queued for execution:       {configs_to_run_count}")
    print("=" * 80)

    if task_queue.empty():
        print("=== No new configurations to execute. Exiting. ===")
        return

    processes = []
    print(f"\n=== Starting {NUM_PARALLEL_EXECUTIONS} parallel workers ===")
    base_port = base_grid_config['port']

    for i in range(NUM_PARALLEL_EXECUTIONS):
        process = multiprocessing.Process(target=run_grid_search_worker, args=(i, task_queue, base_port, csv_lock))
        processes.append(process)
        process.start()

    task_queue.join()

    # Poison pill to stop workers
    for _ in range(NUM_PARALLEL_EXECUTIONS):
        task_queue.put(None)

    for process in processes:
        process.join()

    print("\n=== Parallel execution completed. ===")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()