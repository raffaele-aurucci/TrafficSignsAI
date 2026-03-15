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


NUM_PARALLEL_EXECUTIONS = 1
GRID_SEARCH_CONFIG_PATH = 'grid_search_config.json'

MATCH_KEYS = [
    'model_name', 'global_epoch', 'local_epoch', 'num_clients', 'learning_rate',
    'models_percentage', 'aggregation_algorithm', 'pruning_threshold', 'num_custom_layers'
]

# ----------------------------------------------------------------------------
# FINGERPRINT HELPERS
# ----------------------------------------------------------------------------

def _normalize_fp_value(v) -> str:
    """
    Normalize a configuration value to a canonical string representation.

    Converts numeric values that are whole numbers to their integer string form
    (e.g. ``1.0`` → ``'1'``) to avoid false mismatches between ``'1'`` and
    ``'1.0'`` when comparing fingerprints read from different sources (JSON
    config vs CSV row).

    Args:
        v: The value to normalize. Can be any type; non-numeric values are
           converted via ``str()``.

    Returns:
        A normalized string representation of ``v``.
    """
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError):
        return str(v)


def make_config_fingerprint(config: dict) -> FrozenSet[Tuple[str, str]]:
    """
    Produce an immutable fingerprint for a hyperparameter configuration.

    Builds a ``frozenset`` of ``(key, normalized_value)`` pairs from the keys
    listed in ``MATCH_KEYS`` that are present in ``config``.  Using a frozenset
    makes the fingerprint order-independent and directly hashable, so it can be
    stored in a set or used as a dict key for O(1) deduplication lookups.

    Args:
        config: Dictionary containing the hyperparameter configuration.

    Returns:
        A ``FrozenSet`` of ``(str, str)`` tuples uniquely identifying the
        configuration with respect to ``MATCH_KEYS``.
    """
    return frozenset(
        (k, _normalize_fp_value(config[k])) for k in MATCH_KEYS if k in config
    )


def build_fingerprint_sets(global_csv_path: str) -> Tuple[Set, Set]:
    """
    Parse the global results CSV and build two sets of configuration fingerprints.

    Reads the CSV once and classifies every row by its ``execution_phase`` column:

    * **completed** — configurations for which a ``'Post-Pruning'`` row exists.
      These are fully finished runs and can be skipped entirely.
    * **incomplete** — configurations for which only a ``'Pre-Pruning'`` row
      exists (i.e. the run crashed or was interrupted before the post-pruning
      phase).  These should be cleaned up and re-queued.

    Args:
        global_csv_path: Absolute or relative path to the global grid-search
            summary CSV file.

    Returns:
        A tuple ``(completed, incomplete)`` where both elements are sets of
        ``FrozenSet`` fingerprints as produced by :func:`make_config_fingerprint`.
        Both sets are empty if the file does not exist or cannot be read.
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
            (k, _normalize_fp_value(row[k]))
            for k in MATCH_KEYS
            if k in row and pd.notna(row[k]) and str(row[k]) != ''
        )
        phase = str(row.get('execution_phase', ''))
        if phase == 'Post-Pruning':
            completed.add(fp)
        elif phase == 'Pre-Pruning':
            pre_only.add(fp)

    incomplete = pre_only - completed
    return completed, incomplete


# ----------------------------------------------------------------------------
# CLEANUP AND DUPLICATE HANDLING
# ----------------------------------------------------------------------------

def remove_duplicate_configurations(global_csv_path: str) -> None:
    """
    Remove duplicate rows from the global results CSV, keeping the last occurrence.

    A row is considered a duplicate when it shares the same values for all
    ``MATCH_KEYS`` columns **and** ``execution_phase``.  Duplicates can arise if
    a worker crashes after writing results but before the task is marked as done,
    causing it to be re-executed on restart.

    The file is overwritten in-place only when at least one duplicate is found;
    if the file is absent or unreadable the function returns silently.

    Args:
        global_csv_path: Path to the global grid-search summary CSV file.
    """
    if not os.path.exists(global_csv_path):
        return
    try:
        df = pd.read_csv(global_csv_path, dtype=str)
    except Exception as e:
        print(f"[CLEANUP ERROR] Cannot read CSV: {e}")
        return

    if df.empty or 'execution_phase' not in df.columns:
        return

    original_rows = len(df)
    subset_keys = [k for k in MATCH_KEYS if k in df.columns] + ['execution_phase']
    df_cleaned = df.drop_duplicates(subset=subset_keys, keep='last')
    removed_count = original_rows - len(df_cleaned)

    if removed_count > 0:
        print(f"[CLEANUP] Removed {removed_count} duplicate rows from the CSV.")
        df_cleaned.to_csv(global_csv_path, index=False)
    else:
        print("[CLEANUP] No duplicates found in the CSV.")


def cleanup_incomplete_run(config: dict, global_csv_path: str,
                           models_dir: str, lock: multiprocessing.Lock) -> None:
    """
    Remove all traces of an incomplete run (Pre-Pruning phase only) so it can
    be safely re-executed from scratch.

    An *incomplete run* is one where the Pre-Pruning phase completed and was
    written to the CSV, but the process terminated before the Post-Pruning phase
    finished.  Leaving these orphan rows in place would cause
    :func:`build_fingerprint_sets` to flag the configuration as incomplete on
    every startup, so they must be purged before re-queuing.

    The cleanup procedure:

    1. Removes all matching rows from the global CSV.
    2. Under ``lock``, inspects ``best_scores.json``:

       * Compares the orphan run's best F1 against the score already stored.
       * If the orphan score is strictly better, deletes the corresponding
         model-weights ``.pkl`` file and the associated TA private-key file,
         then removes the entry from ``best_scores.json``.
       * If the stored score is equal or better, skips file deletion to avoid
         destroying a valid checkpoint.

    Args:
        config:           Hyperparameter configuration dict of the incomplete run.
        global_csv_path:  Path to the global grid-search summary CSV file.
        models_dir:       Root directory where model checkpoints are stored.
        lock:             Inter-process lock protecting ``best_scores.json``
                          and the model files from concurrent access.
    """
    model_name = config.get('model_name', 'unknown')
    print(f"[CLEAN & RESTART] Incomplete run (Pre-Pruning only) for {model_name} "
          f"(LR: {config.get('learning_rate')}).")

    try:
        df = pd.read_csv(global_csv_path, dtype=str)
    except Exception:
        return

    mask = pd.Series([True] * len(df))
    for k in MATCH_KEYS:
        if k in config and k in df.columns:
            mask &= (df[k].astype(str).apply(_normalize_fp_value) == _normalize_fp_value(config[k]))
    matched_indices = df[mask].index

    if len(matched_indices) == 0:
        return

    orphan_rows = df.loc[matched_indices].copy()
    df.drop(index=matched_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(global_csv_path, index=False)

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


# ----------------------------------------------------------------------------
# BEST MODEL SAVING
# ----------------------------------------------------------------------------

def save_best_model_if_needed(run_summary: dict, best_weights: list, lock: multiprocessing.Lock,
                              models_dir: str, config: dict, phase: str) -> None:
    """
    Persist model weights to disk if the current run sets a new F1 record for
    the given execution phase.

    The function reads ``best_scores.json`` under ``lock``, compares the current
    run's F1 score against the historical best for ``(model_name, phase)``, and —
    if the current run is strictly better — atomically writes:

    * ``best_scores.json`` — updated with the new metrics and config snapshot.
    * ``best_{model_name}_{phase}.pkl`` — serialized model weights (pickle).
    * ``best_{model_name}_{phase}_key.pkl`` — copy of the TA private key, if a
      Trusted Aggregator port was used during this run.

    The separation between ``metric_keys`` and ``config_keys_to_exclude`` ensures
    that runtime-only fields (ports, paths, locks) are never persisted to the JSON
    record, keeping it portable and human-readable.

    Args:
        run_summary:  Dictionary produced by the aggregator at the end of the
                      federated training round; must contain at least
                      ``'model_name'`` and ``'best_f1'`` (or ``'f1_score'``).
        best_weights: List of serializable weight tensors/arrays to pickle.
        lock:         Inter-process lock protecting ``best_scores.json`` and the
                      model files from concurrent writes across workers.
        models_dir:   Root directory under which per-model subdirectories are
                      created to store checkpoints.
        config:       Full hyperparameter configuration dict for this run; used
                      to retrieve ``ta_port`` and to build the grid-search config
                      snapshot stored in ``best_scores.json``.
        phase:        Execution phase label, either ``'Pre-Pruning'`` or
                      ``'Post-Pruning'``.
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
                    final_key_path = os.path.join(specific_model_dir,
                                                  f"best_{model_name}_{safe_phase}_key.pkl")
                    shutil.copy(temp_key_path, final_key_path)

            print(f"\n[*] NEW RECORD ({phase}) for {model_name}! F1: {current_f1:.4f}. "
                  f"Saved in: {specific_model_dir}/\n")
    finally:
        lock.release()


# ----------------------------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------------------------

def load_json(filename: str) -> dict:
    """
    Load and parse a JSON file into a Python dictionary.

    Args:
        filename: Path to the JSON file to read.

    Returns:
        The deserialized contents of the file as a ``dict``.

    Raises:
        FileNotFoundError: If ``filename`` does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    with open(filename) as f:
        return json.load(f)


def generate_configurations(base_config: dict, search_space: dict):
    """
    Generate all hyperparameter configurations from a Cartesian product of the
    search space, merged on top of a base configuration.

    Iterates over every combination of values in ``search_space`` using
    ``itertools.product``.  For each combination, a shallow copy of
    ``base_config`` is created and updated with the sampled hyperparameters,
    so the base config is never mutated.

    If ``search_space`` is empty, the function yields ``base_config`` unchanged
    (single-configuration grid search).

    Args:
        base_config:   Dictionary of fixed parameters shared by all
                       configurations (dataset path, number of classes, etc.).
        search_space:  Dictionary mapping hyperparameter names to lists of
                       candidate values, e.g.
                       ``{'learning_rate': [0.01, 0.001], 'pruning_threshold': [0.3, 0.5]}``.

    Yields:
        A new ``dict`` for each combination, containing all keys from
        ``base_config`` with the sampled hyperparameters overwritten.
    """
    if not search_space:
        yield base_config
        return
    keys, values = zip(*search_space.items())
    for combo in itertools.product(*values):
        config = base_config.copy()
        config.update(dict(zip(keys, combo)))
        yield config


def wait_for_server_ready(url: str, timeout: int = 60) -> bool:
    """
    Poll a URL until the server responds with HTTP 200 or the timeout expires.

    Used after spawning the federated server thread to ensure it is fully
    initialized and accepting connections before the clients are started.
    Retries every second on ``ConnectionError`` (server not yet listening).

    Args:
        url:     The endpoint to poll, typically the server root
                 (e.g. ``'http://127.0.0.1:5000/'``).
        timeout: Maximum number of seconds to wait before giving up.
                 Defaults to 60.

    Returns:
        ``True`` if the server returned HTTP 200 within the timeout window,
        ``False`` otherwise.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except ConnectionError:
            time.sleep(1)
    return False


# ----------------------------------------------------------------------------
# SERVER THREAD
# ----------------------------------------------------------------------------

def start_server_thread(config: dict, server_instance_ref: list) -> None:
    """
    Instantiate and run a :class:`federated_server.FederatedServer` inside the
    calling thread.

    The function also monkey-patches ``_reset_for_new_training`` on the server
    so that the aggregator active *before* the pruning reset is stored as
    ``server.pre_pruning_aggregator``.  This reference is later used by
    :func:`_execute_single_config` to retrieve and save the Pre-Pruning
    model weights independently of the Post-Pruning ones.

    The server instance is appended to ``server_instance_ref`` (a shared list)
    so that the caller can access it after ``server.run()`` returns.

    Args:
        config:              Full configuration dict for this run; forwarded
                             directly to ``FederatedServer.__init__``.
        server_instance_ref: A mutable list used to pass the created server
                             object back to the calling scope.  Expected to be
                             empty on entry; the server is appended as its first
                             element.
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


# ----------------------------------------------------------------------------
# ISOLATED SUBPROCESS — one per configuration
# ----------------------------------------------------------------------------

def _execute_single_config(config: dict, csv_lock: multiprocessing.Lock) -> None:
    """
    Runs a SINGLE FL configuration in a dedicated child process.

    WHY A SUBPROCESS:
    Python does not guarantee that RAM is returned to the OS after gc.collect()
    or del: the glibc/PyMalloc allocator keeps memory in its internal pool
    for future allocations.
    The only absolute guarantee is process termination: when this function
    returns, the kernel deallocates the entire address space — the PyTorch
    model, weight numpy arrays, dataset, Flask app, CUDA cache — without
    exception.
    """
    worker_id = config.get('worker_id', 'N/A')
    model_name = config.get('model_name', '?')
    server_instance_ref = []

    try:
        server_thread = threading.Thread(
            target=start_server_thread,
            args=(config, server_instance_ref),
            daemon=True
        )
        server_thread.start()

        server_url = f"http://{config['ip_address']}:{config['port']}/"
        if wait_for_server_ready(server_url):
            run_multiple_clients.main(config)
        else:
            print(f"[Worker {worker_id}] Unable to contact server. Skipping '{model_name}'.")

        server_thread.join()

        # -- WEIGHT SAVING (PRE & POST PRUNING) --------------------------------
        if server_instance_ref:
            server_obj = server_instance_ref[0]
            models_dir = config.get('base_model_output_path', 'saved_models')

            pre_agg = getattr(server_obj, 'pre_pruning_aggregator', None)
            if pre_agg and pre_agg.best_model_weights is not None:
                save_best_model_if_needed(
                    pre_agg.run_summary, pre_agg.best_model_weights,
                    csv_lock, models_dir, config, phase="Pre-Pruning"
                )

            post_agg = server_obj.aggregator
            if post_agg and post_agg.best_model_weights is not None:
                save_best_model_if_needed(
                    post_agg.run_summary, post_agg.best_model_weights,
                    csv_lock, models_dir, config, phase="Post-Pruning"
                )
        # -----------------------------------------------------------------------

    except Exception as e:
        print(f"[Worker {worker_id}] Error in isolated subprocess for '{model_name}': {e}")
        import traceback
        traceback.print_exc()


# ----------------------------------------------------------------------------
# WORKER PROCESS
# ----------------------------------------------------------------------------

def run_grid_search_worker(worker_id: int, task_queue: multiprocessing.JoinableQueue,
                           base_port: int, csv_lock: multiprocessing.Lock) -> None:
    """
    Long-running worker that pulls configurations from the queue.
    For each configuration it launches an isolated child subprocess and waits
    for its completion before moving on to the next one.

    Process hierarchy:
        Main process
          └─ Worker process  (this one — stable RAM for the entire grid search)
               └─ Config subprocess  (one per config — exits → OS frees everything)
                    ├─ Server thread
                    └─ N Client threads

    The worker never directly allocates models or datasets.
    """
    print(f"--- Starting Grid Search Worker {worker_id} ---")

    while True:
        try:
            config = task_queue.get(timeout=1)
        except queue.Empty:
            continue

        if config is None:
            print(f"--- Worker {worker_id} received stop signal. Shutting down. ---")
            task_queue.task_done()
            break

        splitting_dir = None
        try:
            dataset_name = config['dataset_name']
            model_name = config['model_name']

            print("\n" + "#" * 80)
            print(f"### [Worker {worker_id}] STARTING CONFIGURATION: {dataset_name} | {model_name} ###")

            config['worker_id'] = worker_id
            config['port'] = base_port + (worker_id * 2)
            config['ta_port'] = base_port + (worker_id * 2) + 1
            config['MIN_NUM_WORKERS'] = int(config['num_clients'] * config['models_percentage'])
            config['csv_lock'] = csv_lock

            run_identifier = f"{dataset_name}_{model_name}_w{worker_id}"
            config['log_dir'] = os.path.join(config['base_log_path'], run_identifier)
            config['run_metrics_output_path'] = os.path.join(config['base_csv_path'], dataset_name)
            splitting_dir = os.path.join(config['base_split_data_path'], run_identifier)
            config['splitting_dir'] = splitting_dir

            os.makedirs(config['log_dir'], exist_ok=True)
            os.makedirs(splitting_dir, exist_ok=True)

            # Isolated subprocess: all heavy memory lives inside here.
            # When config_proc.join() returns, the kernel has already freed everything.
            config_proc = multiprocessing.Process(
                target=_execute_single_config,
                args=(config, csv_lock),
                daemon=False  # Non-daemon: we wait for explicit completion
            )
            config_proc.start()
            config_proc.join()

            if config_proc.exitcode != 0:
                print(f"[Worker {worker_id}] WARNING: subprocess for '{model_name}' "
                      f"exited with code {config_proc.exitcode}.")
            else:
                print(f"[Worker {worker_id}] Subprocess for '{model_name}' completed. "
                      f"Memory fully released by OS.")

        except Exception as e:
            print(f"[Worker {worker_id}] Error preparing configuration: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Splitting dir cleanup is performed in the worker (not in the subprocess)
            # to guarantee cleanup even if the subprocess crashes.
            if splitting_dir and os.path.isdir(splitting_dir):
                try:
                    shutil.rmtree(splitting_dir)
                    print(f"[Worker {worker_id}] Removed splitting dir: {splitting_dir}")
                except Exception as e:
                    print(f"[Worker {worker_id}] Could not remove splitting dir: {e}")

            task_queue.task_done()


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    """
    Entry point for the distributed grid search.

    Orchestrates the full lifecycle of a federated grid search:

    1. **Load configuration** — reads ``GRID_SEARCH_CONFIG_PATH`` to obtain the
       base parameters, the common search space, and per-model search spaces.
    2. **Deduplication** — calls :func:`remove_duplicate_configurations` to
       sanitize the results CSV before inspecting it.
    3. **Fingerprint scan** — calls :func:`build_fingerprint_sets` to classify
       existing results as *completed* or *incomplete* in a single CSV pass.
    4. **Queue population** — iterates over the full Cartesian product of
       hyperparameters.  Completed configurations are skipped; incomplete ones
       are cleaned up via :func:`cleanup_incomplete_run` before being re-queued;
       new ones are enqueued directly.
    5. **Worker launch** — spawns ``NUM_PARALLEL_EXECUTIONS`` worker processes
       (:func:`run_grid_search_worker`), each consuming configurations from the
       shared ``JoinableQueue``.
    6. **Graceful shutdown** — waits for the queue to drain, then sends one
       ``None`` sentinel per worker to signal termination, and joins all
       processes.
    """
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

    print("Checking and removing any duplicate configurations in the CSV...")
    remove_duplicate_configurations(global_csv_path)

    print("Building fingerprint sets from existing CSV (one-time read)...")
    completed_fps, incomplete_fps = build_fingerprint_sets(global_csv_path)
    print(f"  Completed configs (Post-Pruning present): {len(completed_fps)}")
    print(f"  Incomplete configs (Pre-Pruning only):    {len(incomplete_fps)}")

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

                if fp in completed_fps:
                    continue

                if fp in incomplete_fps:
                    cleanup_incomplete_run(hyper_config, global_csv_path, models_dir, csv_lock)
                    incomplete_fps.discard(fp)

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
        process = multiprocessing.Process(
            target=run_grid_search_worker,
            args=(i, task_queue, base_port, csv_lock)
        )
        processes.append(process)
        process.start()

    task_queue.join()

    for _ in range(NUM_PARALLEL_EXECUTIONS):
        task_queue.put(None)

    for process in processes:
        process.join()

    print("\n=== Parallel execution completed. ===")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()