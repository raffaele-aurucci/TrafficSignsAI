import logging
import time
import os
import socketio
import numpy as np
from typing import Dict
import threading

from model_manager import ModelManager
from utils import object_to_pickle_string, pickle_string_to_object, encrypt_weights, decrypt_weights
from data_pruning import prune_and_redistribute_client_dataset  # [PRUNING]


class ContextFilter(logging.Filter):
    """A logging filter that injects client_id into log records."""

    def __init__(self, client_id: str):
        super().__init__()
        self.client_id = client_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.client_id = self.client_id
        return True


class FederatedClient:
    """
    Implements the client-side logic for federated learning.

    Supports plaintext and homomorphically-encrypted weight exchange,
    FedAvg / FedProx / FedLC aggregation algorithms, and optional
    local dataset pruning between training phases.
    """

    def __init__(self, config: Dict, dataset_path: str):
        self.local_model: ModelManager = None
        self.config: Dict = config
        self.dataset_path: str = dataset_path
        self.client_id: str = os.path.basename(dataset_path)
        self.logger: logging.LoggerAdapter = self._setup_logger()

        # Encryption
        self.encryption_mode: str = self.config.get('encryption_mode', 'none')
        if self.encryption_mode != 'no_encryption':
            self.logger.info("Encryption enabled. Keys will be requested from the Trusted Authority.")
            self.paillier_pubkey = None
            self.paillier_privkey = None
            self.keys_received_event = threading.Event()

        # [PRUNING] Pruning configuration
        self.enable_pruning: bool = self.config.get('enable_pruning', False)
        self.pruning_threshold = self.config.get('pruning_threshold', 1.0)
        self.logger.info(
            "Pruning: %s (threshold=%s)",
            "ENABLED" if self.enable_pruning else "DISABLED",
            self.pruning_threshold
        )

        self.sio = socketio.Client(logger=True, request_timeout=10, reconnection=True)
        self._register_event_handlers()
        self.connect_to_server()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_logger(self) -> logging.LoggerAdapter:
        worker_id = self.config.get('worker_id', 'N/A')
        logger_name = f"FederatedClient-W{worker_id}"
        base_logger = logging.getLogger(logger_name)

        if not base_logger.hasHandlers():
            datestr = time.strftime('%d%m')
            timestr = time.strftime('%m%d%H%M')
            log_dir_base = self.config.get('log_dir', 'logs')
            log_dir = os.path.join(log_dir_base, datestr, "FL-Client-LOG")
            os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(os.path.join(log_dir, f'{timestr}_{self.client_id}.log'))
            file_handler.setLevel(logging.INFO)
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.WARN)

            formatter = logging.Formatter('%(asctime)s - %(client_id)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)

            base_logger.setLevel(logging.INFO)
            base_logger.addHandler(file_handler)
            base_logger.addHandler(stream_handler)

        adapter = logging.LoggerAdapter(base_logger, {'client_id': self.client_id})
        log_filter = ContextFilter(self.client_id)
        if not any(isinstance(f, ContextFilter) for f in adapter.logger.filters):
            adapter.logger.addFilter(log_filter)
        return adapter

    def connect_to_server(self) -> None:
        server_address = "http://" + self.config['ip_address'] + ":" + str(self.config['port'])
        self.logger.info("Connecting to server at %s", server_address)
        try:
            self.sio.connect(server_address, transports=['websocket'])
            self.logger.info("Sending wake up message to the server.")
            self.sio.emit('client_wake_up')
            self.sio.wait()
        except socketio.exceptions.ConnectionError as e:
            self.logger.error("Failed to connect to the server: %s", e)
            exit(1)

    def _register_event_handlers(self) -> None:
        self.sio.on('connect', self._on_connect)
        self.sio.on('disconnect', self._on_disconnect)
        self.sio.on('reconnect', self._on_reconnect)
        self.sio.on('shutdown', self._on_shutdown)
        self.sio.on('init', self._on_init)
        self.sio.on('request_update', self._on_request_update)
        self.sio.on('stop_and_eval', self._on_stop_and_eval)
        self.sio.on('distribute_calibration', self._on_distribute_calibration)
        # [PRUNING] New events
        self.sio.on('start_pruning', self._on_start_pruning)
        self.sio.on('skip_pruning', self._on_skip_pruning)
        self.sio.on('server_ready_for_new_training', self._on_server_ready_for_new_training)

    # ------------------------------------------------------------------
    # Connection events
    # ------------------------------------------------------------------

    def _on_connect(self):
        self.logger.info("Successfully connected with SID: %s", self.sio.sid)

    def _on_disconnect(self):
        self.logger.info("Disconnected from the server.")
        self.sio.disconnect()

    def _on_reconnect(self):
        self.logger.info("Reconnected to the server.")

    def _on_shutdown(self):
        self.logger.info("Received shutdown signal. Exiting.")
        self.sio.disconnect()
        exit(0)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _on_init(self, data: Dict):
        """
        Handles the initialization signal from the server.
        If encryption is enabled, connects to the Trusted Authority first.
        """
        self.logger.info("Received initialization signal from server.")
        if self.encryption_mode != 'no_encryption':
            ta_address = data['ta_address']
            self.logger.info("Connecting to Trusted Authority at %s.", ta_address)
            ta_sio = socketio.Client(reconnection=False)

            @ta_sio.on('distribute_keys')
            def on_receive_keys(key_data):
                self.logger.info("Keys received from Trusted Authority.")
                self.paillier_pubkey = pickle_string_to_object(key_data['pubkey'])
                self.paillier_privkey = pickle_string_to_object(key_data['privkey'])
                self.keys_received_event.set()
                ta_sio.disconnect()

            try:
                ta_sio.connect(ta_address, transports=['websocket'])
                ta_sio.emit('request_keys')
                if not self.keys_received_event.wait(timeout=30):
                    self.logger.error("CRITICAL: Did not receive keys from TA within timeout.")
                    exit(1)
            except Exception as e:
                self.logger.error("CRITICAL: Failed to get keys from TA: %s", e)
                exit(1)

        self._initialize_model_and_report_ready()

    def _initialize_model_and_report_ready(self):
        """Initializes the local ModelManager and notifies the server."""
        self.logger.info("Initializing local model on dataset: %s", self.dataset_path)
        self.local_model = ModelManager(
            config=self.config,
            dataset_path=self.dataset_path
        )
        samples_per_class = self.local_model.get_samples_per_class()
        self.logger.info("Model initialized. Sending 'client_ready'.")
        self.sio.emit('client_ready', {'samples_per_class': object_to_pickle_string(samples_per_class)})

    def _on_distribute_calibration(self, data: Dict):
        self.logger.info("Received static calibration term from the server.")
        calibration_val = pickle_string_to_object(data['calibration_val'])
        self.local_model.set_calibration_term(calibration_val)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _process_server_weights(self, data: Dict):
        """
        Decrypts (if needed) and averages the weights received from the server.
        Works for both the training rounds and the pruning phase.
        """
        weights_pickled = data['current_weights']
        total_size = data.get('total_training_size', 0)
        weights_data = pickle_string_to_object(weights_pickled)
        plaintext_sum = None

        first_element = weights_data[0]
        if isinstance(first_element, dict):
            if self.encryption_mode == 'no_encryption':
                self.logger.error("Received encrypted-format weights while in 'no_encryption' mode.")
                return None
            try:
                plaintext_sum = decrypt_weights(self.paillier_privkey, weights_data, logger=self.logger)
            except Exception as e:
                self.logger.error("Weight decryption failed: %s", e, exc_info=True)
                return None
        elif isinstance(first_element, np.ndarray):
            plaintext_sum = weights_data
        else:
            self.logger.error("Unknown weights format: %s", type(first_element))
            return None

        if total_size > 0:
            return [w / total_size for w in plaintext_sum]
        else:
            self.logger.warning("total_training_size is 0. Using weights as-is.")
            return plaintext_sum

    # ------------------------------------------------------------------
    # Training round
    # ------------------------------------------------------------------

    def _on_request_update(self, data: Dict):
        """Spawns a worker thread to handle the local update without blocking the event loop."""
        self.logger.info("Received update request for round %s.", data['round_number'])
        worker_thread = threading.Thread(target=self._update_worker, args=(data,))
        worker_thread.daemon = True
        worker_thread.start()

    def _update_worker(self, data: Dict):
        """Full federated learning update cycle for a single round."""
        try:
            self.logger.info("Worker thread started for round %s.", data['round_number'])

            averaged_weights = self._process_server_weights(data)
            if averaged_weights is None:
                self.logger.error("Worker: failed to obtain valid weights. Aborting round.")
                return

            self.local_model.set_weights(averaged_weights)

            _, train_map, train_loss, train_size = self.local_model.train(
                epochs=data['epochs'],
                lr=data['learning_rate'],
                batch_size=data['batch_size'],
                algorithm=self.config.get("aggregation_algorithm", "FedAvg"),
                global_weights=averaged_weights,
                mu=self.config.get("fedprox_mu", 0.0)
            )

            local_weights = self.local_model.get_weights()

            if self.encryption_mode != 'no_encryption':
                weights_to_send = encrypt_weights(
                    self.paillier_pubkey, local_weights,
                    encryption_mode=self.encryption_mode, logger=self.logger
                )
            else:
                weights_to_send = local_weights

            avg_f1 = (np.mean(list(train_map['f1_score']))
                      if isinstance(train_map['f1_score'], list)
                      else train_map['f1_score'])
            avg_acc = (np.mean(list(train_map['accuracy']))
                       if isinstance(train_map['accuracy'], list)
                       else train_map['accuracy'])

            response = {
                'round_number': data['round_number'],
                'train_loss': train_loss,
                'avg_f1': avg_f1,
                'avg_acc': avg_acc,
                'train_size': train_size,
                'weights': object_to_pickle_string(weights_to_send)
            }

            self.sio.emit('client_update', response)
            self.logger.info("Round %s — train_loss=%.4f, f1=%.4f",
                             data['round_number'], train_loss, avg_f1)

        except Exception as e:
            self.logger.error("Error in update worker: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _on_stop_and_eval(self, data: Dict):
        """
        Applies the aggregated model, evaluates on the local validation set,
        and reports results to the server.

        If STOP=True and pruning is enabled, the client does NOT exit —
        it waits for the 'start_pruning' event from the server.
        """
        self.logger.info("Received aggregated model for evaluation.")
        final_weights = self._process_server_weights(data)
        if final_weights is None:
            self.logger.error("Evaluation failed: could not process server weights.")
            return

        self.local_model.set_weights(final_weights)
        valid_loss, metric_score, _, test_size = self.local_model.validate(data['batch_size'])

        response = {
            'test_loss': valid_loss,
            'test_f1': metric_score['f1_score'],
            'test_acc': metric_score['accuracy'],
            'test_prec': metric_score['precision'],
            'test_recall': metric_score['recall'],
            'test_size': test_size,
        }
        self.sio.emit('client_eval', response)

        if data.get('STOP', False):
            if self.enable_pruning:
                # [PRUNING] Non uscire: aspetta 'start_pruning' (se selezionato)
                # oppure 'skip_pruning' (se idle), poi 'server_ready_for_new_training'
                self.logger.info(
                    "Training phase completed (STOP=True). "
                    "Pruning enabled — waiting for 'start_pruning' or 'skip_pruning' from server."
                )
            else:
                self.logger.info("Federated training finished. Shutting down client.")
                exit(0)

    # ------------------------------------------------------------------
    # [PRUNING] Pruning phase handlers
    # ------------------------------------------------------------------

    def _on_start_pruning(self, data: Dict):
        """
        [PRUNING] Receives the global model from the server and runs local
        dataset pruning in-place. Emits 'complete_pruning' when done.

        The server sends the same payload structure as 'stop_and_eval', so
        _process_server_weights handles decryption transparently.
        """
        self.logger.info("[PRUNING] Received 'start_pruning' from server.")

        try:
            # Ottieni i pesi mediati (con eventuale decryption)
            averaged_weights = self._process_server_weights(data)
            if averaged_weights is None:
                self.logger.error("[PRUNING] Failed to process server weights. Pruning aborted.")
                return

            # Carica i pesi nel modello locale
            self.local_model.set_weights(averaged_weights)

            # Accede al modello PyTorch sottostante (nn.Module)
            pytorch_model = self.local_model.model

            device = self.config.get('device', 'cpu')
            threshold = self.config.get('pruning_threshold', 1.0)

            self.logger.info(
                "[PRUNING] Starting prune_and_redistribute_client_dataset "
                "on '%s' (threshold=%s, device=%s).",
                self.dataset_path, threshold, device
            )

            pruning_stats = prune_and_redistribute_client_dataset(
                model=pytorch_model,
                client_root=self.dataset_path,
                thresholds=threshold,
                device=device
            )

            self.logger.info(
                "[PRUNING] Pruning complete. Reduction: %.1f%% (%d -> %d samples). "
                "Emitting 'complete_pruning'.",
                pruning_stats.get('reduction_pct', 0),
                pruning_stats.get('samples_before', 0),
                pruning_stats.get('samples_after', 0),
            )
            self.sio.emit('complete_pruning', {'pruning_stats': pruning_stats})

        except Exception as e:
            self.logger.error("[PRUNING] Critical error during pruning: %s", e, exc_info=True)
            # Emit anyway so the server doesn't hang waiting for this client
            self.sio.emit('complete_pruning', {'pruning_stats': {}})

    def _on_skip_pruning(self):
        """
        [PRUNING] Received when this client was NOT selected for the pruning phase.
        Mirrors the idle behaviour during training rounds: the client does nothing
        and simply waits for 'server_ready_for_new_training', exactly as it would
        wait for 'request_update' in a round where it is not picked.
        """
        self.logger.info(
            "[PRUNING] Not selected for pruning this phase (idle). "
            "Waiting for 'server_ready_for_new_training'."
        )

    def _on_server_ready_for_new_training(self):
        """
        [PRUNING] Called after all clients have completed pruning and the server
        has reset its state. Reinitializes the ModelManager on the (now pruned)
        dataset folder (same path, different file contents) and signals readiness.
        """
        self.logger.info(
            "[PRUNING] Server ready for new training. "
            "Reinitializing ModelManager on pruned dataset: %s", self.dataset_path
        )

        # Reinizializza ModelManager: rilegge la struttura delle cartelle aggiornata
        self.local_model = ModelManager(
            config=self.config,
            dataset_path=self.dataset_path
        )

        samples_per_class = self.local_model.get_samples_per_class()
        self.logger.info("[PRUNING] Model reinitialized. Sending 'client_ready' for new training.")
        self.sio.emit('client_ready', {'samples_per_class': object_to_pickle_string(samples_per_class)})