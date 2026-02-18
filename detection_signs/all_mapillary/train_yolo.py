from ultralytics import YOLO
import torch
import os
import json

# Models
model_list = ["yolo11n.pt", "yolo11s.pt", "yolo26n.pt", "yolo26s.pt"]

# Grid
grid_settings = [
    {'epochs': 100, 'patience': 20, 'lr0': 0.01,  'name': 'standard'},
    {'epochs': 150, 'patience': 30, 'lr0': 0.005, 'name': 'balanced'},
    {'epochs': 200, 'patience': 50, 'lr0': 0.003, 'name': 'precise'}
]

LOG_DIR = "logs"


def save_log(run_name: str, model_name: str, setting: dict, metrics: dict, status: str, error: str = ""):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{run_name}.json")

    log_data = {
        "run_name":    run_name,
        "model":       model_name,
        "setting":     setting['name'],
        "status":      status,
        "error":       error,
        "hyperparams": {
            "epochs":    setting['epochs'],
            "patience":  setting['patience'],
            "lr0":       setting['lr0'],
            "imgsz":     640,
            "batch":     16,
            "optimizer": "AdamW",
            "seed":      42,
        },
        "metrics": {
            "mAP50":     metrics.get("metrics/mAP50(B)",    "N/A"),
            "mAP50_95":  metrics.get("metrics/mAP50-95(B)", "N/A"),
            "precision": metrics.get("metrics/precision(B)", "N/A"),
            "recall":    metrics.get("metrics/recall(B)",    "N/A"),
            "box_loss":  metrics.get("train/box_loss",       "N/A"),
            "cls_loss":  metrics.get("train/cls_loss",       "N/A"),
        }
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)

    print(f"  → Log saved in: {log_path}")


def run_benchmarking():
    print("START COMPARATIVE TRAINING (GRID SEARCH)\n")

    for model_name in model_list:
        for setting in grid_settings:
            model_stem = model_name.split('.')[0]
            run_name   = f"{model_stem}_{setting['name']}"

            print(f"\n{'=' * 50}")
            print(f" TEST: {run_name}")
            print(f" MODEL: {model_name} | EPOCHS: {setting['epochs']} | PATIENCE: {setting['patience']}")
            print(f"{'=' * 50}\n")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            try:
                model = YOLO(model_name)

                results = model.train(
                    data="data.yaml",
                    epochs=setting['epochs'],
                    patience=setting['patience'],
                    lr0=setting['lr0'],
                    imgsz=640,
                    batch=16,
                    optimizer="AdamW",
                    cache=True,
                    device=0,
                    project="benchmark_tesi",
                    name=run_name,
                    exist_ok=True,
                    seed=42,
                )

                metrics = results.results_dict if hasattr(results, "results_dict") else {}
                save_log(run_name, model_name, setting, metrics, status="success")

            except Exception as e:
                print(f"Error training {model_name}: {e}")
                save_log(run_name, model_name, setting, metrics={}, status="error", error=str(e))
                continue

    print(f"\nBenchmark completed. Log saved in: {LOG_DIR}/")


if __name__ == "__main__":
    run_benchmarking()