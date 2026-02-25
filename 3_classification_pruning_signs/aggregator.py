import time
import os
import numpy as np
import pandas as pd
from typing import List, Dict
import logging

from model_manager import ModelManager
from utils import sum_encrypted_weights, multiply_encrypted_weights_by_scalar


class Aggregator:
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.current_weights = self._initialize_weights()
        self.metrics_history: Dict[int, Dict] = {}

        # Best metrics tracking
        self.best_f1_score: float = 0.0
        self.best_accuracy: float = 0.0
        self.best_precision: float = 0.0
        self.best_recall: float = 0.0
        self.best_loss: float = float('inf')
        self.best_model_weights: List[np.ndarray] = None
        self.best_round: int = -1

        # Early stopping logic
        self.prev_test_loss: float = None
        self.early_stop_counter: int = 0

        self.training_start_time: float = time.time()
        self.run_summary: Dict = None

    def _initialize_weights(self) -> List[np.ndarray]:
        """Initializes model weights by instantiating a temporary ModelManager."""
        self.logger.info("Initializing model weights...")
        temp_model = ModelManager(config=self.config, dataset_path="")
        initial_weights = temp_model.get_weights()
        self.logger.info(f"Trainable parameters initialized for model {self.config['model_name']}.")
        del temp_model
        return initial_weights

    def aggregate_weights(self, client_updates: List[Dict], algorithm: str) -> bool:
        """
        Performs weighted aggregation of client weights.
        Note: The final division (averaging) is delegated to clients in this architecture.
        """
        self.logger.info(f"Aggregating weights using '{algorithm}'...")
        if not client_updates:
            self.logger.warning("No client updates received for aggregation.")
            return False

        client_weights = [u['weights'] for u in client_updates]
        client_sizes = [u['train_size'] for u in client_updates]

        if algorithm in ["FedAvg", "FedProx", "FedLC"]:
            total_size = sum(client_sizes)
            if total_size == 0:
                self.logger.warning("Total training size is 0. Skipping aggregation.")
                return False

            # Initialize summed weights with the first client's weighted contribution
            summed_weights = [w * client_sizes[0] for w in client_weights[0]]

            # Add contributions from the remaining clients
            for client_idx in range(1, len(client_weights)):
                for i in range(len(summed_weights)):
                    summed_weights[i] += client_weights[client_idx][i] * client_sizes[client_idx]

            self.current_weights = summed_weights
            self.logger.info(f"Weighted summation successful. Total size: {total_size}.")
            return True
        else:
            raise ValueError(f"Aggregation algorithm '{algorithm}' is not supported.")

    def aggregate_encrypted_updates(self, round_client_updates: List[Dict]):
        """Handles weight aggregation for encrypted updates using homomorphic operations."""
        client_sizes = [update['train_size'] for update in round_client_updates]

        # Start with the first weighted encrypted update
        summed_encrypted_weights = multiply_encrypted_weights_by_scalar(
            round_client_updates[0]['weights'], client_sizes[0]
        )

        # Add subsequent weighted encrypted updates
        for i in range(1, len(round_client_updates)):
            weighted_update = multiply_encrypted_weights_by_scalar(
                round_client_updates[i]['weights'], client_sizes[i]
            )
            summed_encrypted_weights = sum_encrypted_weights(summed_encrypted_weights, weighted_update)

        self.current_weights = summed_encrypted_weights
        self.logger.info("Encrypted aggregation (weighted sum) complete.")

    def aggregate_train_loss_weighted(self, client_losses: List[float], client_sizes: List[int], current_round: int):
        """Calculates and stores the weighted average of training losses from clients."""
        total_size = sum(client_sizes)
        if total_size == 0:
            return

        weighted_loss = np.average(client_losses, weights=client_sizes)
        self.logger.info(f"Round {current_round} - Weighted average training loss: {weighted_loss:.4f}")

        if current_round not in self.metrics_history:
            self.metrics_history[current_round] = {}
        self.metrics_history[current_round]['train_loss'] = weighted_loss

    def aggregate_evaluation_results(self, eval_updates: List[Dict], current_round: int) -> bool:
        """Aggregates test metrics and checks for early stopping criteria."""
        client_sizes = [u['test_size'] for u in eval_updates]
        total_test_size = sum(client_sizes)
        if total_test_size == 0:
            return False

        # Compute weighted averages for all metrics
        avg_loss = np.average([u['test_loss'] for u in eval_updates], weights=client_sizes)
        avg_f1 = np.average([u['test_f1'] for u in eval_updates], weights=client_sizes)
        avg_acc = np.average([u['test_acc'] for u in eval_updates], weights=client_sizes)
        avg_prec = np.average([u['test_prec'] for u in eval_updates], weights=client_sizes)
        avg_recall = np.average([u['test_recall'] for u in eval_updates], weights=client_sizes)

        self.logger.info(
            f"--- Round {current_round} Evaluation: Loss {avg_loss:.4f}, F1 {avg_f1:.4f}, Acc {avg_acc:.4f} ---")

        self.metrics_history[current_round].update({
            'test_loss': avg_loss, 'test_f1': avg_f1, 'test_acc': avg_acc,
            'test_prec': avg_prec, 'test_recall': avg_recall
        })

        # Track the best model based on F1 Score
        if avg_f1 > self.best_f1_score:
            self.best_f1_score, self.best_accuracy = avg_f1, avg_acc
            self.best_precision, self.best_recall = avg_prec, avg_recall
            self.best_loss = avg_loss
            self.best_model_weights = self.current_weights
            self.best_round = current_round

        # Early Stopping Logic: check if loss is increasing
        if self.prev_test_loss is not None and avg_loss > self.prev_test_loss:
            self.early_stop_counter += 1
        else:
            self.early_stop_counter = 0

        self.prev_test_loss = avg_loss
        return self.early_stop_counter >= self.config['early_stop_patience']

    def log_best_model_stats(self):
        """Logs a summary of the best performing round."""
        self.logger.info("=" * 30)
        self.logger.info("Federated Training Finished")
        self.logger.info(f"Best round: {self.best_round} | F1: {self.best_f1_score:.44f} | "
                         f"Acc: {self.best_accuracy:.4f} | Loss: {self.best_loss:.4f}")
        self.logger.info("=" * 30)

    def get_stats(self) -> pd.DataFrame:
        """Converts the metrics history dictionary to a Pandas DataFrame."""
        if not self.metrics_history:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(self.metrics_history, orient='index')

    def save_results(self) -> None:
        """Saves metrics to CSV and prepares the final run summary dictionary."""
        metrics_dir = self.config['run_metrics_output_path']
        base_csv_dir = self.config.get('base_csv_path', 'csv')
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(base_csv_dir, exist_ok=True)

        timestr = time.strftime('%Y%m%d-%H%M%S')
        metrics_df = self.get_stats()
        worker_id = self.config.get('worker_id', 'NA')
        dataset_name = self.config.get('dataset_name', 'unknown')

        # Save the standard round-by-round CSV
        filename = os.path.join(metrics_dir, f"{timestr}_{dataset_name}_worker_{worker_id}.csv")
        metrics_df.to_csv(filename)
        self.logger.info(f"Saved round metrics to {filename}")

        # Prepare the summary dictionary with best values and hyperparameters
        self.run_summary = self.config.copy()

        # Label the current execution phase
        is_post_pruning = self.config.get('is_post_pruning_run', False)
        if self.config.get('enable_pruning', False):
            phase = "Post-Pruning" if is_post_pruning else "Pre-Pruning"
        else:
            phase = "Standard"

        self.run_summary.update({
            'execution_phase': phase,
            'best_round': self.best_round,
            'best_f1': self.best_f1_score,
            'best_acc': self.best_accuracy,
            'best_prec': self.best_precision,
            'best_recall': self.best_recall,
            'best_loss': self.best_loss,
            'total_duration_sec': time.time() - self.training_start_time,
            'round_dataframe_path': filename,
        })

        # Cleanup: remove unnecessary paths or overly verbose dictionaries from the summary
        keys_to_remove = ['shared_csv_path', 'run_metrics_output_path', 'base_csv_path',
                          'base_log_path', 'base_plot_path', 'base_split_data_path',
                          'ip_address', 'port', 'ta_port', 'splitting_dir', 'early_stop_patience',
                          'dataset_path', 'csv_lock', 'log_dir', 'enable_pruning', 'weighted_aggregation',
                          'MIN_NUM_WORKERS', 'is_post_pruning_run']

        for key in keys_to_remove:
            self.run_summary.pop(key, None)

        # Append the dictionary to the Global CSV SAFELY for multiprocessing
        global_csv_path = os.path.join(base_csv_dir, "global_grid_search_summary.csv")
        summary_df = pd.DataFrame([self.run_summary])

        csv_lock = self.config.get('csv_lock')
        if csv_lock:
            csv_lock.acquire()

        try:
            if not os.path.exists(global_csv_path):
                # Create a new file if it does not exist
                summary_df.to_csv(global_csv_path, index=False)
            else:
                # Read the existing one, merge data, and overwrite to align columns correctly
                existing_df = pd.read_csv(global_csv_path)
                combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
                combined_df.to_csv(global_csv_path, index=False)

            self.logger.info(f"Saved global execution summary to {global_csv_path}")
        except Exception as e:
            self.logger.error(f"Failed to append to global summary CSV: {e}")
        finally:
            if csv_lock:
                csv_lock.release()

    def get_run_summary(self) -> Dict:
        return self.run_summary

    def get_parameter_plots(self) -> List[str]:
        """Returns a list of plot paths (Visualization logic currently disabled)."""
        return []