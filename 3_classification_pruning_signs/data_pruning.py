import os
import shutil
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm


"""
================================================================================
LOCAL DATASET PRUNING MODULE
================================================================================
This module implements decentralized data pruning logic.
It runs entirely on the edge (client-side), ensuring data privacy.

  - 1 → compute_influence_scores
  - 2 → adaptive criterion with z-score normalization (z ∈ [-ε, +ε])
  - 3 → class safeguard
  - 4 → physical dataset reconstruction

Z-score normalization ensures that ε has a consistent, architecture-independent
meaning: "how many standard deviations from the center to keep".
  - ε = 1.0 → ~68% of samples kept  (normal distribution assumption)
  - ε = 2.0 → ~95% of samples kept
  - ε = 0.5 → ~38% of samples kept
"""


# ============================================================
#                  INFLUENCE SCORE CALCULATION
# ============================================================
def compute_influence_scores(model, dataset, device="cpu"):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Find the last trainable Linear module
    last_linear = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and any(p.requires_grad for p in module.parameters()):
            last_linear = module

    if last_linear is None:
        raise ValueError("[PRUNING] No trainable Linear layer found.")

    # Collect all parameters of the layer (weight + bias)
    last_layer_params = [p for p in last_linear.parameters() if p.requires_grad]

    scores = []
    for x, y in tqdm(dataloader, desc="Computing Influence Scores"):
        x, y = x.to(device), y.to(device)
        model.zero_grad()

        output = model(x)
        loss = criterion(output, y)

        grads = torch.autograd.grad(
            loss,
            last_layer_params,
            retain_graph=False,
            create_graph=False
        )

        # L2 norm concatenated over weight and bias gradients
        combined_norm = torch.cat([g.flatten() for g in grads]).norm().item()
        scores.append(combined_norm)

    return np.array(scores)


# ============================================================
# DATASET PRUNING AND IN-PLACE RECONSTRUCTION
# ============================================================
def prune_and_redistribute_client_dataset(model, client_root, thresholds=None, device="cpu"):
    """
    [2;3;4 - Adaptive Criterion with Z-score Normalization and Class Safeguard]

    Calculates influence scores, applies z-score normalization per class,
    filters samples within [-ε, +ε] z-score range, activates the safeguard
    if necessary, and physically rebuilds the train/ and valid/ folders in-place.

    Z-score normalization makes ε architecture-independent: the same ε value
    produces consistent pruning aggressiveness regardless of whether the model
    is a MobileNet, ViT, EfficientNet, etc.

    Expected structure of client_root:
        client_root/
            train/
                class_A/  img1.jpg  img2.jpg  ...
                class_B/  ...
            valid/
                class_A/  ...
                class_B/  ...

    Args:
        model:       PyTorch model for score calculation.
        client_root: Root path of the client's dataset.
        thresholds:  Float (global epsilon), dict {class_name: epsilon},
                     or use the special value "skip" to exclude a class
                     from pruning.
        device:      'cpu' or 'cuda'.
    """
    if thresholds is None:
        thresholds = 1.0

    print(f"[PRUNING] Starting on '{client_root}' with thresholds: {thresholds}")

    # Load train + valid keeping original paths for reconstruction
    dataset, all_samples, class_names = _load_datasets_keeping_structure(client_root)

    # Compute scores on the entire local dataset (train + valid together)
    scores = compute_influence_scores(model, dataset, device=device)

    all_paths = np.array([s[0] for s in all_samples])
    all_labels_idx = np.array([s[1] for s in all_samples])

    # Mask: True = keep sample, False = discard
    keep_mask = np.zeros(len(scores), dtype=bool)

    for cls_idx in np.unique(all_labels_idx):
        cls_name = class_names[cls_idx]

        # Determine epsilon for this class
        if isinstance(thresholds, dict):
            eps = thresholds.get(cls_name, 1.0)
        else:
            eps = thresholds

        indices = np.where(all_labels_idx == cls_idx)[0]
        cls_scores = scores[indices]

        # Bypass: the class is not pruned
        if eps == "skip" or eps is None:
            keep_mask[indices] = True
            print(f"  -> Class '{cls_name}': SKIPPED (kept all {len(indices)})")
            continue

        # Z-score normalization
        mu = cls_scores.mean()
        sigma = cls_scores.std()

        # Edge case: all scores are identical → zero variance, keep everything
        if sigma < 1e-8:
            keep_mask[indices] = True
            print(f"  -> Class '{cls_name}': zero variance, kept all {len(indices)} "
                  f"(mu={mu:.4f}, sigma≈0)")
            continue

        z_scores = (cls_scores - mu) / sigma

        # Criterion: keep samples within [-ε, +ε] in z-score space.
        # ε is now architecture-independent: same ε = same % of distribution
        # regardless of the model's gradient magnitude scale.
        in_range = (z_scores >= -eps) & (z_scores <= eps)
        kept_indices_local = indices[in_range]

        # CLASS SAFEGUARD: ensures minimum 10% of original or 15 samples
        min_samples = max(int(len(indices) * 0.1), 15)

        if len(kept_indices_local) < min_samples:
            print(f"[WARN] Class '{cls_name}': too reduced ({len(kept_indices_local)} < {min_samples}). "
                  f"Activating class safeguard.")
            needed = min_samples - len(kept_indices_local)

            # Recover discarded samples closest to the center (lowest |z-score|)
            # i.e. the most "prototypical" samples among those discarded
            rejected_mask = ~in_range
            rejected_indices_local = indices[rejected_mask]
            rejected_abs_z = np.abs(z_scores[rejected_mask])
            best_rejected_idx = np.argsort(rejected_abs_z)[:needed]
            recovered = rejected_indices_local[best_rejected_idx]
            kept_indices_local = np.concatenate([kept_indices_local, recovered])

        keep_mask[kept_indices_local] = True
        print(f"  -> Class '{cls_name}': {len(kept_indices_local)}/{len(indices)} kept "
              f"(eps={eps:.2f}, mu={mu:.4f}, sigma={sigma:.4f})")

    # Physical in-place reconstruction
    pruning_stats = _rebuild_dataset_in_place(client_root, all_paths, all_labels_idx, class_names, keep_mask)
    return pruning_stats


# ============================================================
# DATASET LOADING WITH PATH TRACKING
# ============================================================
def _load_datasets_keeping_structure(client_root):
    """
    Loads train and valid via ImageFolder, concatenates them for global
    score calculation, and returns the unified list of (path, class_idx).

    Note: ImageFolder assigns class indices in alphabetical order.
    The structure does not require CSV annotation files.

    Returns:
        ds_full (ConcatDataset): Complete train+valid dataset.
        all_samples (list):      List of tuples (absolute_path, class_idx).
        class_names (list):      Class names (ImageFolder order).
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(client_root, "train")
    valid_dir = os.path.join(client_root, "valid")

    if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
        raise FileNotFoundError(
            f"[PRUNING] 'train' or 'valid' folders not found in: {client_root}"
        )

    ds_train = datasets.ImageFolder(train_dir, transform=transform)
    ds_valid = datasets.ImageFolder(valid_dir, transform=transform)

    # Check consistency between train and valid classes
    if ds_train.classes != ds_valid.classes:
        raise ValueError(
            f"[PRUNING] Train and valid classes do not match:\n"
            f"  train: {ds_train.classes}\n  valid: {ds_valid.classes}"
        )

    ds_full = ConcatDataset([ds_train, ds_valid])
    # Samples ordered: first all train, then all valid
    all_samples = ds_train.samples + ds_valid.samples

    return ds_full, all_samples, ds_train.classes


# ============================================================
# PHYSICAL IN-PLACE RECONSTRUCTION
# ============================================================
def _rebuild_dataset_in_place(client_root, all_paths, all_labels_idx, class_names, keep_mask):
    """
    [4] Physically replaces train/ and valid/ with only the samples
    selected by keep_mask. Operates via a temporary folder to ensure
    atomicity and prevent data leakage between the two splits.
    """
    print("[PRUNING] Rebuilding folders (in-place operation)...")

    temp_root = client_root + "_pruning_temp"
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)

    os.makedirs(os.path.join(temp_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(temp_root, "valid"), exist_ok=True)

    count_train = 0
    count_valid = 0
    skipped = 0
    sep = os.path.sep

    for i, should_keep in enumerate(keep_mask):
        if not should_keep:
            continue

        original_path = all_paths[i]
        cls_name = class_names[all_labels_idx[i]]
        filename = os.path.basename(original_path)

        # Determine the target split from the original path
        if sep + "train" + sep in original_path:
            dest_split = "train"
            count_train += 1
        elif sep + "valid" + sep in original_path:
            dest_split = "valid"
            count_valid += 1
        else:
            print(f"[WARN] Unrecognized split for: {original_path}. Sample skipped.")
            skipped += 1
            continue

        dest_dir = os.path.join(temp_root, dest_split, cls_name)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(original_path, os.path.join(dest_dir, filename))

    # Atomic replacement: remove old folders and move the new ones
    shutil.rmtree(os.path.join(client_root, "train"))
    shutil.rmtree(os.path.join(client_root, "valid"))
    shutil.move(os.path.join(temp_root, "train"), os.path.join(client_root, "train"))
    shutil.move(os.path.join(temp_root, "valid"), os.path.join(client_root, "valid"))
    shutil.rmtree(temp_root)

    total_kept = count_train + count_valid
    total_original = len(keep_mask)
    reduction_pct = (1 - total_kept / max(total_original, 1)) * 100

    print(f"[PRUNING] Completed.")
    print(f"  Samples: {total_kept}/{total_original} kept (reduction: {reduction_pct:.1f}%)")
    print(f"  Train: {count_train} | Valid: {count_valid}"
          + (f" | Skipped: {skipped}" if skipped > 0 else ""))

    return {
        'samples_before': total_original,
        'samples_after': total_kept,
        'reduction_pct': round(reduction_pct, 2),
        'train_kept': count_train,
        'valid_kept': count_valid,
    }