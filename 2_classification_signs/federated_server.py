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

    Manages client connections, initiates training rounds, aggregates model updates
    (both plaintext and homomorphically encrypted), evaluates the global model,
    and serves a status endpoint.
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
        self.current_round: int = -1  # -1 indicates training has not started
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
            self.logger.info("Encryption mode enabled: %s. The server will operate key-agnostic.", self.encryption_mode)

        # Flask & SocketIO setup
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, ping_timeout=3600, ping_interval=5, max_http_buffer_size=int(1e32))
        self._register_routes_and_handlers()

    def _setup_logger(self) -> logging.Logger:
        """Configures and returns a file+stream logger for the server."""
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
        """Registers Flask routes and SocketIO event handlers."""
        self.app.route('/')( self._health_check)

        self.socketio.on('connect')(self._on_connect)
        self.socketio.on('disconnect')(self._on_disconnect)
        self.socketio.on('reconnect')(self._on_reconnect)
        self.socketio.on('client_wake_up')(self._on_client_wake_up)
        self.socketio.on('client_ready')(self._on_client_ready)
        self.socketio.on('client_update')(self._on_client_update)
        self.socketio.on('client_eval')(self._on_client_eval)

    def _health_check(self):
        """Simple HTTP endpoint used by the grid search runner to verify the server is up."""
        return "Server is running", 200

    def _start_next_training_round(self) -> None:
        """Increments the round counter, selects participating clients, and broadcasts the update request."""
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

        self.logger.info("Requesting updates from clients: %s", selected_clients)
        for sid in selected_clients:
            emit('request_update', request_data, room=sid)

    def _trigger_global_evaluation(self) -> None:
        """Broadcasts the current aggregated model to all clients for evaluation."""
        self.logger.info("Triggering global model evaluation on all %d clients.", len(self.registered_clients))
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
        """Aggregates client weight updates using either plaintext or encrypted aggregation."""
        train_losses = [x['train_loss'] for x in self.client_updates_this_round]
        train_sizes = [x['train_size'] for x in self.client_updates_this_round]
        self.total_training_size_in_round = sum(train_sizes)

        if self.encryption_mode == 'no_encryption':
            self.logger.info("Aggregating plaintext updates.")
            self.aggregator.aggregate_weights(
                self.client_updates_this_round,
                self.config['aggregation_algorithm']
            )
        else:
            self.logger.info("Aggregating encrypted updates.")
            self.aggregator.aggregate_encrypted_updates(self.client_updates_this_round)

        if self.config.get('weighted_aggregation', True):
            self.aggregator.aggregate_train_loss_weighted(train_losses, train_sizes, self.current_round)
        else:
            self.aggregator.aggregate_train_loss(train_losses, self.current_round)

    def _shutdown_server(self) -> None:
        """Emits a shutdown signal to all clients and stops the SocketIO server."""
        ta_address = f"http://{self.config['ip_address']}:{self.config['ta_port']}"
        emit('shutdown', {'ta_address': ta_address})
        time.sleep(1)
        self.logger.info("Shutting down the server.")
        self.socketio.stop()

    # --- SocketIO Event Handlers ---

    def _on_connect(self):
        self.logger.info("Client connected: %s", request.sid)

    def _on_disconnect(self):
        self.logger.info("Client disconnected: %s", request.sid)
        if request.sid in self.registered_clients:
            self.registered_clients.remove(request.sid)

    def _on_reconnect(self):
        self.logger.info("Client reconnected: %s", request.sid)

    def _on_client_wake_up(self):
        """Responds to a waking client with the Trusted Authority address for key retrieval."""
        self.logger.info("Client %s is waking up. Sending init signal with TA address.", request.sid)
        ta_address = f"http://{self.config['ip_address']}:{self.config['ta_port']}"
        emit('init', {'ta_address': ta_address})

    def _on_client_ready(self, data: Dict):
        """
        Registers a client as ready and stores its per-class sample counts.
        Once all expected clients are ready, triggers initialization and training.
        """
        self.logger.info("Client %s is ready and sent data stats.", request.sid)
        self.registered_clients.add(request.sid)

        samples_per_class = pickle_string_to_object(data['samples_per_class'])
        self.client_stats_buffer.append(samples_per_class)

        if len(self.client_stats_buffer) >= self.config['num_clients'] and self.static_calibration_term is None:
            self.logger.info("All clients are ready. Finalizing initialization...")
            self._finalize_initialization_and_start_training()

    def _finalize_initialization_and_start_training(self) -> None:
        """
        Computes and distributes the static calibration term (for FedLC),
        then kicks off the first federated training round.
        """
        if self.config.get("aggregation_algorithm") == "FedLC":
            self.logger.info("Calculating static calibration term for FedLC...")

            # Aggregate per-class sample counts across all clients, avoiding division by zero
            total_samples_per_class = np.sum(self.client_stats_buffer, axis=0)
            total_samples_per_class[total_samples_per_class == 0] = 1

            calibration_val = total_samples_per_class ** (-1 / 4)
            self.static_calibration_term = calibration_val

            self.logger.info("Distributing static calibration term to all clients.")
            data_to_send = {'calibration_val': object_to_pickle_string(self.static_calibration_term)}
            for sid in self.registered_clients:
                emit('distribute_calibration', data_to_send, room=sid)

        self.logger.info("Initialization complete. Starting federated training process.")
        # Brief delay to ensure clients have processed the calibration term before the first round
        time.sleep(10)
        self._start_next_training_round()

    def _on_client_update(self, data: Dict):
        """
        Receives a local model update from a client. Once all expected updates for the
        current round are collected, triggers aggregation and global evaluation.
        """
        with self.lock:
            self.logger.info("Received update from client %s for round %d.", request.sid, data['round_number'])

            if data['round_number'] != self.current_round:
                self.logger.warning("Received an update for an old round (%d). Current is %d. Ignoring.",
                                    data['round_number'], self.current_round)
                return

            try:
                data['weights'] = pickle_string_to_object(data['weights'])
            except Exception as e:
                self.logger.error("Error unpickling weights from client %s: %s", request.sid, e)
                return

            self.client_updates_this_round.append(data)

            if len(self.client_updates_this_round) >= self.num_clients_per_round:
                self.logger.info("All %d client updates for round %d received. Aggregating...",
                                 self.num_clients_per_round, self.current_round)
                self._aggregate_updates()

                if self.current_round >= self.config['global_epoch'] - 1:
                    self.logger.info("Maximum number of global rounds reached. Finishing process.")
                    self.is_training_finished = True

                self._trigger_global_evaluation()

    def _on_client_eval(self, data: Dict):
        """
        Receives an evaluation result from a client. Once all clients have reported,
        aggregates metrics and either starts the next round or shuts down if training is done.
        """
        with self.lock:
            self.logger.info("Received evaluation from client %s.", request.sid)
            self.client_evaluations_this_round.append(data)

            if len(self.client_evaluations_this_round) >= len(self.registered_clients):
                self.logger.info("All evaluations received for round %d. Aggregating evaluation metrics.",
                                 self.current_round)

                is_early_stopping = self.aggregator.aggregate_evaluation_results(
                    self.client_evaluations_this_round, self.current_round
                )

                if is_early_stopping:
                    self.logger.info("Early stopping condition met. Finishing process.")
                    self.is_training_finished = True

                if self.is_training_finished:
                    self.logger.info("Federated training is complete.")
                    self.aggregator.log_best_model_stats()
                    self.aggregator.save_results()

                    for sid in self.registered_clients:
                        emit("shutdown", room=sid)
                    self._shutdown_server()
                else:
                    self.logger.info("Proceeding to the next round.")
                    self._start_next_training_round()