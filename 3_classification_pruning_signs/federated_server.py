import logging
import random
import time
import os
import numpy as np
import json
from typing import Dict, List, Set
from threading import Lock
from flask import Flask, request
from flask_socketio import SocketIO, emit

from utils import object_to_pickle_string, pickle_string_to_object, sum_encrypted_weights, \
    multiply_encrypted_weights_by_scalar
from aggregator import Aggregator


def load_json(filename: str) -> Dict:
    """Loads and returns the contents of a JSON file."""
    with open(filename) as f:
        return json.load(f)


class FederatedServer:
    """
    Implements the server-side logic for federated learning orchestration.

    Manages client connections, initiates training rounds, aggregates model
    updates (plaintext and homomorphically encrypted), evaluates the global
    model, and — when enable_pruning=True — orchestrates a local dataset
    pruning phase between two consecutive FL training runs.

    Pruning flow (when enable_pruning=True):
      1. First FL training completes (max rounds or early stopping).
      2. Server saves first-phase results via aggregator.save_results().
      3. Server emits 'start_pruning' to ALL clients (pruning_participation=1.0)
         or to a configurable fraction (pruning_participation<1.0).
      4. Each client prunes its local dataset in-place and emits 'complete_pruning'.
      5. When all pruning clients are done, server resets its state and emits
         'server_ready_for_new_training'.
      6. Clients reinitialize their ModelManager and emit 'client_ready'.
      7. A full second FL training run starts on the pruned data, using
         post_pruning_models_percentage to select clients per round.
      8. Second run completes → normal shutdown.

    Key parameters (in config):
      - models_percentage        : fraction of clients per round in PRE-pruning training.
      - pruning_participation    : fraction of clients that perform local pruning (default 1.0).
      - post_pruning_models_pct  : fraction of clients per round in POST-pruning training
                                   (defaults to models_percentage if not set).
    """

    def __init__(self, config: Dict):
        self.config: Dict = config
        self.logger: logging.Logger = self._setup_logger()
        self.logger.info("Server configuration: %s", self.config)
        self.lock = Lock()

        # Federated learning state
        self.min_num_workers: int = self.config['MIN_NUM_WORKERS']
        self.num_clients_per_round: int = 0
        self.registered_clients: Set[str] = set()
        self.current_round: int = -1
        self.is_training_finished: bool = False

        # Per-round buffers
        self.client_updates_this_round: List[Dict] = []
        self.client_evaluations_this_round: List[Dict] = []
        self.client_metrics_buffer: List[Dict] = []
        self.total_training_size_in_round: int = 0

        self.aggregator = Aggregator(self.config, self.logger)
        self.client_stats_buffer: List[Dict] = []
        self.static_calibration_term: np.ndarray = None

        # Encryption
        self.encryption_mode: str = self.config.get('encryption_mode', 'none')
        if self.encryption_mode != 'no_encryption':
            self.logger.info("Encryption mode: %s. Server operates key-agnostic.", self.encryption_mode)

        # [PRUNING] Pruning orchestration state
        self.enable_pruning: bool = self.config.get('enable_pruning', False)
        self.pruning_phase_done: bool = False
        self.pruning_complete_count: int = 0
        self.expected_pruning_clients: Set[str] = set()
        # Buffer to hold pruning stats across the aggregator reset
        self._pending_pruning_stats: List[Dict] = []

        self.logger.info(
            "Pruning: %s | clients per round (pre and post): %d/%d (models_percentage=%.2f)",
            "ENABLED" if self.enable_pruning else "DISABLED",
            self.min_num_workers, self.config['num_clients'],
            self.config.get('models_percentage', 1.0)
        )

        # Flask & SocketIO setup
        self.app = Flask(__name__)
        self.socketio = SocketIO(
            self.app,
            ping_timeout=3600,
            ping_interval=5,
            max_http_buffer_size=int(1e32)
        )
        self._register_routes_and_handlers()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"FederatedServer-W{self.config.get('worker_id', 'N/A')}")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()

        datestr = time.strftime('%d%m')
        timestr = time.strftime('%m%d%H%M')
        log_dir = os.path.join(self.config.get('log_dir', 'logs'), datestr, "FL-Server-LOG")
        os.makedirs(log_dir, exist_ok=True)

        fh = logging.FileHandler(os.path.join(log_dir, f'{timestr}.log'))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARN)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def run(self) -> None:
        """Starts the Flask-SocketIO server."""
        self.logger.info("Federated Server starting...")
        self.socketio.run(self.app, host=self.config['ip_address'], port=self.config['port'])

    def _register_routes_and_handlers(self) -> None:
        """Registers Flask routes and all SocketIO event handlers."""
        self.app.route('/')(self._health_check)

        self.socketio.on('connect')(self._on_connect)
        self.socketio.on('disconnect')(self._on_disconnect)
        self.socketio.on('reconnect')(self._on_reconnect)
        self.socketio.on('client_wake_up')(self._on_client_wake_up)
        self.socketio.on('client_ready')(self._on_client_ready)
        self.socketio.on('client_update')(self._on_client_update)
        self.socketio.on('client_eval')(self._on_client_eval)
        # [PRUNING] Pruning completion handler
        self.socketio.on('complete_pruning')(self._on_complete_pruning)

    def _health_check(self):
        """HTTP endpoint used by grid search runner to verify the server is up."""
        return "Server is running", 200

    # ------------------------------------------------------------------
    # Training orchestration
    # ------------------------------------------------------------------

    def _start_next_training_round(self) -> None:
        """
        Increments round counter, selects clients using min_num_workers
        (same ratio for both pre- and post-pruning training), and broadcasts
        the update request to selected clients only.
        """
        self.current_round += 1
        self.logger.info("--- Starting Round %d ---", self.current_round)

        self.client_updates_this_round.clear()
        self.client_metrics_buffer.clear()

        all_available_clients = list(self.registered_clients)
        selected_clients = random.sample(all_available_clients, self.min_num_workers)
        self.num_clients_per_round = len(selected_clients)

        current_weights_pickled = object_to_pickle_string(self.aggregator.current_weights)
        request_data = {
            'epochs': self.config['local_epoch'],
            'batch_size': self.config['batch_size'],
            'learning_rate': self.config['learning_rate'],
            'round_number': self.current_round,
            'current_weights': current_weights_pickled,
            'total_training_size': self.total_training_size_in_round,
        }

        self.logger.info(
            "Requesting updates from %d/%d clients: %s",
            self.min_num_workers, len(all_available_clients), selected_clients
        )
        for sid in selected_clients:
            emit('request_update', request_data, room=sid)

    def _trigger_global_evaluation(self) -> None:
        """Broadcasts the aggregated model to all clients for evaluation."""
        self.logger.info("Triggering global evaluation on %d clients.", len(self.registered_clients))
        self.client_evaluations_this_round.clear()

        data_to_send = {
            'batch_size': self.config['batch_size'],
            'current_weights': object_to_pickle_string(self.aggregator.current_weights),
            'total_training_size': self.total_training_size_in_round,
            'STOP': self.is_training_finished,
            'round_number': self.current_round
        }

        for sid in self.registered_clients:
            emit('stop_and_eval', data_to_send, room=sid)

    def _aggregate_updates(self) -> None:
        """Aggregates client weight updates (plaintext or encrypted)."""
        train_losses = [x['train_loss'] for x in self.client_updates_this_round]
        train_sizes = [x['train_size'] for x in self.client_updates_this_round]
        self.total_training_size_in_round = sum(train_sizes)

        if self.encryption_mode == 'no_encryption':
            self.aggregator.aggregate_weights(
                self.client_updates_this_round,
                self.config['aggregation_algorithm']
            )
        else:
            self.aggregator.aggregate_encrypted_updates(self.client_updates_this_round)

        if self.config.get('weighted_aggregation', True):
            self.aggregator.aggregate_train_loss_weighted(train_losses, train_sizes, self.current_round)
        else:
            self.aggregator.aggregate_train_loss(train_losses, self.current_round)

    def _shutdown_server(self) -> None:
        """Sends shutdown to clients and stops the SocketIO server."""
        ta_address = f"http://{self.config['ip_address']}:{self.config.get('ta_port', 5002)}"
        emit('shutdown', {'ta_address': ta_address})
        time.sleep(1)
        self.logger.info("Shutting down the server.")
        self.socketio.stop()

    # ------------------------------------------------------------------
    # [PRUNING] Pruning phase
    # ------------------------------------------------------------------

    def _start_pruning_phase(self) -> None:
        """
        [PRUNING] Seleziona casualmente un sottoinsieme di client usando la stessa
        logica dei round di training (min_num_workers client estratti da registered_clients).

        - I client selezionati ricevono 'start_pruning' e prunano il dataset locale.
        - I client NON selezionati ricevono 'skip_pruning' e rimangono idle,
          esattamente come nei round di training in cui non vengono scelti.
        - La valutazione post-pruning resta su TUTTI i client (identico al training).
        """
        all_available_clients = list(self.registered_clients)

        # Stessa estrazione casuale usata in _start_next_training_round
        selected_clients = random.sample(all_available_clients, self.min_num_workers)
        idle_clients = [sid for sid in all_available_clients if sid not in selected_clients]

        self.expected_pruning_clients = set(selected_clients)
        self.pruning_complete_count = 0

        self.logger.info(
            "[PRUNING] Pruning phase started: %d/%d clients selected "
            "(same ratio as training rounds via models_percentage).",
            len(selected_clients), len(all_available_clients)
        )

        data_to_send = {
            'current_weights': object_to_pickle_string(self.aggregator.current_weights),
            'total_training_size': self.total_training_size_in_round,
        }

        # Client selezionati: prunano il dataset
        for sid in selected_clients:
            emit('start_pruning', data_to_send, room=sid)

        # Client idle: notifica esplicita di restare in attesa, come nei round di training
        for sid in idle_clients:
            emit('skip_pruning', room=sid)
            self.logger.info("[PRUNING] Client %s set to idle (not selected for pruning).", sid)

    def _on_complete_pruning(self, data: Dict = None):
        """
        [PRUNING] Receives pruning completion (and stats) from selected clients.
        When all expected clients have finished, resets server state and
        notifies ALL clients to reinitialize for the post-pruning training run.
        """
        with self.lock:
            if request.sid not in self.expected_pruning_clients:
                return

            # Accumulate pruning stats on the server so they survive the aggregator reset
            # The client wraps stats in {'pruning_stats': {...}}, so unwrap it first.
            if data and isinstance(data, dict):
                stats = data.get('pruning_stats', data)
                if stats:
                    self._pending_pruning_stats.append(stats)
                    self.logger.info(
                        "[PRUNING] Stats received from %s: before=%d after=%d (%.1f%% reduction).",
                        request.sid,
                        stats.get('samples_before', 0),
                        stats.get('samples_after', 0),
                        stats.get('reduction_pct', 0.0),
                    )

            self.pruning_complete_count += 1
            self.logger.info(
                "[PRUNING] Client %s completed pruning (%d/%d).",
                request.sid, self.pruning_complete_count, len(self.expected_pruning_clients)
            )

            if self.pruning_complete_count >= len(self.expected_pruning_clients):
                self.logger.info(
                    "[PRUNING] All %d selected clients finished pruning. "
                    "Resetting server for post-pruning training phase.",
                    len(self.expected_pruning_clients)
                )
                self._reset_for_new_training()

                # Notify ALL clients (pruned and non-pruned) to reinitialize
                for sid in self.registered_clients:
                    emit('server_ready_for_new_training', room=sid)

    def _reset_for_new_training(self) -> None:
        """
        [PRUNING] Resets all training state so a fresh FL run can start
        on the pruned datasets. The post-pruning run uses the same
        min_num_workers per round as the pre-pruning run.
        """
        self.logger.info("[PRUNING] Resetting server state for post-pruning training phase...")

        self.current_round = -1
        self.is_training_finished = False
        self.client_updates_this_round.clear()
        self.client_evaluations_this_round.clear()
        self.client_metrics_buffer.clear()
        self.client_stats_buffer.clear()
        self.total_training_size_in_round = 0
        self.static_calibration_term = None
        self.expected_pruning_clients.clear()

        self.config['is_post_pruning_run'] = True

        # Fresh aggregator: clean metrics + re-initialized weights for fair comparison
        self.aggregator = Aggregator(self.config, self.logger)

        # Transfer pruning stats collected during the pruning phase to the new aggregator
        # so they are included in the Post-Pruning save_results() summary.
        if self._pending_pruning_stats:
            self.aggregator.collect_pruning_stats(self._pending_pruning_stats)
            self.logger.info(
                "[PRUNING] Transferred %d pruning stat records to the new aggregator.",
                len(self._pending_pruning_stats)
            )
            self._pending_pruning_stats.clear()

        self.logger.info("[PRUNING] Server reset complete. Waiting for client_ready signals...")

    # ------------------------------------------------------------------
    # SocketIO event handlers
    # ------------------------------------------------------------------

    def _on_connect(self):
        self.logger.info("Client connected: %s", request.sid)

    def _on_disconnect(self):
        self.logger.info("Client disconnected: %s", request.sid)
        if request.sid in self.registered_clients:
            self.registered_clients.remove(request.sid)

    def _on_reconnect(self):
        self.logger.info("Client reconnected: %s", request.sid)

    def _on_client_wake_up(self):
        """Responds to a waking client with the Trusted Authority address."""
        self.logger.info("Client %s waking up. Sending 'init'.", request.sid)
        ta_address = f"http://{self.config['ip_address']}:{self.config.get('ta_port', 5002)}"
        emit('init', {'ta_address': ta_address})

    def _on_client_ready(self, data: Dict):
        """
        Registers a client as ready. Once all expected clients are ready,
        triggers initialization and the first (or second) training run.
        """
        self.logger.info("Client %s is ready.", request.sid)
        self.registered_clients.add(request.sid)

        samples_per_class = pickle_string_to_object(data['samples_per_class'])
        self.client_stats_buffer.append(samples_per_class)

        if len(self.client_stats_buffer) >= self.config['num_clients'] and \
                self.static_calibration_term is None:
            self.logger.info("All clients ready. Starting initialization...")
            self._finalize_initialization_and_start_training()

    def _finalize_initialization_and_start_training(self) -> None:
        """Computes FedLC calibration term (if needed) and starts the first training round."""
        if self.config.get("aggregation_algorithm") == "FedLC":
            self.logger.info("Computing static calibration term for FedLC...")
            total_samples_per_class = np.sum(self.client_stats_buffer, axis=0)
            total_samples_per_class[total_samples_per_class == 0] = 1
            calibration_val = total_samples_per_class ** (-1 / 4)
            self.static_calibration_term = calibration_val

            data_to_send = {'calibration_val': object_to_pickle_string(self.static_calibration_term)}
            for sid in self.registered_clients:
                emit('distribute_calibration', data_to_send, room=sid)

        self.logger.info("Initialization complete. Starting federated training.")
        time.sleep(10)
        self._start_next_training_round()

    def _on_client_update(self, data: Dict):
        """
        Receives a local model update. When all round updates arrive,
        triggers aggregation and global evaluation.
        """
        with self.lock:
            self.logger.info("Received update from %s for round %d.",
                             request.sid, data['round_number'])

            if data['round_number'] != self.current_round:
                self.logger.warning(
                    "Stale update: round %d received, current is %d. Ignoring.",
                    data['round_number'], self.current_round
                )
                return

            try:
                data['weights'] = pickle_string_to_object(data['weights'])
            except Exception as e:
                self.logger.error("Error unpickling weights from %s: %s", request.sid, e)
                return

            self.client_updates_this_round.append(data)

            if len(self.client_updates_this_round) >= self.num_clients_per_round:
                self.logger.info(
                    "All %d updates for round %d received. Aggregating...",
                    self.num_clients_per_round, self.current_round
                )
                self._aggregate_updates()

                if self.current_round >= self.config['global_epoch'] - 1:
                    self.logger.info("Maximum global rounds reached. Finishing.")
                    self.is_training_finished = True

                self._trigger_global_evaluation()

    def _on_client_eval(self, data: Dict):
        """
        Receives an evaluation result. When all clients have reported,
        aggregates metrics then either:
          - starts the next round, or
          - enters the pruning phase (if enabled and first run), or
          - shuts down (second run done, or pruning disabled).
        """
        with self.lock:
            self.logger.info("Received evaluation from client %s.", request.sid)
            self.client_evaluations_this_round.append(data)

            if len(self.client_evaluations_this_round) < len(self.registered_clients):
                return  # Wait for remaining clients

            self.logger.info("All evaluations received for round %d.", self.current_round)

            is_early_stopping = self.aggregator.aggregate_evaluation_results(
                self.client_evaluations_this_round, self.current_round
            )

            if is_early_stopping:
                self.logger.info("Early stopping condition met.")
                self.is_training_finished = True

            if self.is_training_finished:
                self.logger.info("Training phase complete.")
                self.aggregator.log_best_model_stats()
                self.aggregator.save_results()

                # [PRUNING] Decide next action based on pruning config and phase
                if self.enable_pruning and not self.pruning_phase_done:
                    self.logger.info(
                        "[PRUNING] First training phase ended. "
                        "Launching pruning phase before post-pruning training run."
                    )
                    self.pruning_phase_done = True
                    self._start_pruning_phase()
                else:
                    self.logger.info("Federated process fully complete. Shutting down.")
                    for sid in self.registered_clients:
                        emit("shutdown", room=sid)
                    self._shutdown_server()
            else:
                self.logger.info("Proceeding to next round.")
                self._start_next_training_round()