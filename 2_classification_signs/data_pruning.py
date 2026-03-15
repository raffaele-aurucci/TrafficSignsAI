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

  - 1 → compute_influence_scores  (on train + valid for stable mu/sigma)
  - 2 → adaptive criterion with z-score normalization (z ∈ [-ε, +ε])
  - 3 → class safeguard
  - 4 → physical dataset reconstruction (pruning applied to train only)

Key design decision:
  Influence scores are computed on the full local dataset (train + valid)
  to obtain stable per-class mu and sigma estimates, especially important
  when each client holds few samples. However, the pruning mask is applied
  ONLY to the train split: the validation set is preserved intact to ensure
  unbiased evaluation on the original data distribution.

Z-score normalization ensures that ε has a consistent, architecture-independent
meaning: "how many standard deviations from the center to keep".
  - ε = 1.0 → ~68% of train samples kept  (normal distribution assumption)
  - ε = 2.0 → ~95% of train samples kept
  - ε = 0.5 → ~38% of train samples kept
"""

# ----------------------------------------------------------------------------
# INFLUENCE SCORE CALCULATION
# ----------------------------------------------------------------------------

def compute_influence_scores(model, dataset, device="cpu"):
    """
    Compute a per-sample influence score based on the gradient norm of the
    last trainable Linear layer.

    For each sample, a forward pass and a backward pass are performed.
    The influence score is the L2 norm of the concatenated gradients of the
    last Linear layer's weight and bias tensors with respect to the loss.
    A higher score indicates that the sample has a stronger effect on the
    model's last-layer parameters and is therefore considered more
    informative.

    Scores are computed on the full dataset passed in (typically train +
    valid concatenated) so that per-class statistics are estimated on as
    many samples as possible.

    Args:
        model:   A PyTorch ``nn.Module`` in which at least one ``nn.Linear``
                 layer has ``requires_grad=True``.  The last such layer is
                 used for gradient computation.
        dataset: A ``torch.utils.data.Dataset`` returning ``(image, label)``
                 pairs.  Processed one sample at a time (``batch_size=1``).
        device:  Device string passed to ``.to()``, e.g. ``'cpu'`` or
                 ``'cuda'``.  Defaults to ``'cpu'``.

    Returns:
        A ``numpy.ndarray`` of shape ``(N,)`` containing one float score per
        sample, in the same order as ``dataset``.

    Raises:
        ValueError: If no trainable ``nn.Linear`` layer is found in
                    ``model``.
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    last_linear = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and any(p.requires_grad for p in module.parameters()):
            last_linear = module

    if last_linear is None:
        raise ValueError("[PRUNING] No trainable Linear layer found.")

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

        combined_norm = torch.cat([g.flatten() for g in grads]).norm().item()
        scores.append(combined_norm)

    return np.array(scores)


# ----------------------------------------------------------------------------
# DATASET PRUNING AND IN-PLACE RECONSTRUCTION
# ----------------------------------------------------------------------------

def prune_and_redistribute_client_dataset(model, client_root, thresholds=None, device="cpu"):
    """
    Run the full pruning pipeline on a single client's local dataset.

    Implements steps 2–4 of the pruning algorithm:

    2. **Adaptive z-score criterion** — for each class, influence scores are
       z-score normalized using mu and sigma estimated on the combined
       train + valid split.  Only train samples whose z-score falls within
       ``[-ε, +ε]`` are retained.
    3. **Class safeguard** — if the retained set for any class would fall
       below ``max(10% of original train samples, 15)``, the discarded
       samples with the smallest ``|z-score|`` (i.e. closest to the class
       center) are recovered until the minimum is met.
    4. **Physical in-place reconstruction** — the ``train/`` and ``valid/``
       sub-folders are atomically replaced via a temporary directory; the
       validation split is always preserved in full.

    Expected directory layout of ``client_root``::

        client_root/
            train/
                class_A/  img1.jpg  img2.jpg  ...
                class_B/  ...
            valid/
                class_A/  ...
                class_B/  ...

    Args:
        model:       PyTorch ``nn.Module`` used to compute influence scores
                     via :func:`compute_influence_scores`.
        client_root: Absolute or relative path to the client's dataset root.
        thresholds:  Epsilon value(s) controlling the pruning aggressiveness:

                     * ``float`` — a single epsilon applied to every class.
                     * ``dict`` — mapping ``{class_name: epsilon}``; missing
                       classes default to ``1.0``.
                     * ``"skip"`` (per-class) — the class is excluded from
                       pruning and all its train samples are kept.

                     Defaults to ``1.0``.
        device:      Device string forwarded to
                     :func:`compute_influence_scores`.  Defaults to
                     ``'cpu'``.

    Returns:
        A ``dict`` with the following keys:

        * ``samples_before``    — total samples (train + valid) before pruning.
        * ``samples_after``     — total samples (train + valid) after pruning.
        * ``reduction_pct``     — overall reduction percentage.
        * ``train_kept``        — train samples kept.
        * ``valid_kept``        — valid samples kept (always equal to the
          original valid count).
        * ``train_before``      — train samples before pruning.
        * ``train_after``       — train samples after pruning.
        * ``valid_preserved``   — valid samples preserved (same as ``valid_kept``).
        * ``train_reduction_pct`` — percentage reduction on the train split only.
    """
    if thresholds is None:
        thresholds = 1.0

    print(f"[PRUNING] Starting on '{client_root}' with thresholds: {thresholds}")

    dataset, all_samples, class_names, train_count = _load_datasets_keeping_structure(client_root)
    scores = compute_influence_scores(model, dataset, device=device)

    all_paths = np.array([s[0] for s in all_samples])
    all_labels_idx = np.array([s[1] for s in all_samples])

    keep_mask = np.zeros(len(scores), dtype=bool)
    keep_mask[train_count:] = True  # validation set is always fully preserved

    for cls_idx in np.unique(all_labels_idx):
        cls_name = class_names[cls_idx]

        eps = thresholds.get(cls_name, 1.0) if isinstance(thresholds, dict) else thresholds

        indices = np.where((all_labels_idx == cls_idx) & (np.arange(len(scores)) < train_count))[0]
        cls_scores = scores[indices]

        if eps == "skip" or eps is None:
            keep_mask[indices] = True
            print(f"  -> Class '{cls_name}': SKIPPED (kept all {len(indices)})")
            continue

        all_cls_indices = np.where(all_labels_idx == cls_idx)[0]
        all_cls_scores = scores[all_cls_indices]
        mu = all_cls_scores.mean()
        sigma = all_cls_scores.std()

        if sigma < 1e-8:
            keep_mask[indices] = True
            print(f"  -> Class '{cls_name}': zero variance, kept all {len(indices)} train samples "
                  f"(mu={mu:.4f}, sigma≈0)")
            continue

        z_scores = (cls_scores - mu) / sigma
        in_range = (z_scores >= -eps) & (z_scores <= eps)
        kept_indices_local = indices[in_range]

        min_samples = max(int(len(indices) * 0.1), 15)

        if len(kept_indices_local) < min_samples:
            print(f"[WARN] Class '{cls_name}': too reduced ({len(kept_indices_local)} < {min_samples}). "
                  f"Activating class safeguard.")
            needed = min_samples - len(kept_indices_local)
            rejected_indices_local = indices[~in_range]
            rejected_abs_z = np.abs(z_scores[~in_range])
            recovered = rejected_indices_local[np.argsort(rejected_abs_z)[:needed]]
            kept_indices_local = np.concatenate([kept_indices_local, recovered])

        keep_mask[kept_indices_local] = True
        print(f"  -> Class '{cls_name}': {len(kept_indices_local)}/{len(indices)} train samples kept "
              f"(eps={eps:.2f}, mu={mu:.4f}, sigma={sigma:.4f})")

    train_before = int(train_count)
    train_after = int(keep_mask[:train_count].sum())
    valid_total = int(len(scores) - train_count)

    print(f"[PRUNING] Train: {train_after}/{train_before} kept | Valid: {valid_total}/{valid_total} (untouched)")

    pruning_stats = _rebuild_dataset_in_place(client_root, all_paths, all_labels_idx, class_names, keep_mask)

    pruning_stats['train_before'] = train_before
    pruning_stats['train_after'] = train_after
    pruning_stats['valid_preserved'] = valid_total
    pruning_stats['train_reduction_pct'] = round((1 - train_after / max(train_before, 1)) * 100, 2)
    return pruning_stats


# ----------------------------------------------------------------------------
# DATASET LOADING WITH PATH TRACKING
# ----------------------------------------------------------------------------

def _load_datasets_keeping_structure(client_root):
    """
    Load the ``train`` and ``valid`` splits as ``ImageFolder`` datasets,
    concatenate them into a single dataset for global score computation,
    and return the full ordered list of ``(path, class_idx)`` pairs.

    The concatenation order is always ``train`` first, then ``valid``.
    The returned ``train_count`` value marks the boundary between the two
    splits inside ``all_samples``: indices ``[0, train_count)`` belong to
    the train set and indices ``[train_count, ...)`` belong to the
    validation set.  This boundary is used by the caller to restrict
    pruning to train samples only.

    Note:
        ``ImageFolder`` assigns class indices in alphabetical order.
        No CSV annotation files are required.

    Args:
        client_root: Path to the client's dataset root, which must contain
                     ``train/`` and ``valid/`` sub-directories, each with
                     one sub-folder per class.

    Returns:
        A four-tuple ``(ds_full, all_samples, class_names, train_count)``:

        * ``ds_full`` (``ConcatDataset``) — the concatenated train + valid
          dataset ready for use with a ``DataLoader``.
        * ``all_samples`` (``list``) — ordered list of
          ``(absolute_path, class_idx)`` tuples covering both splits.
        * ``class_names`` (``list[str]``) — class names in ``ImageFolder``
          alphabetical order.
        * ``train_count`` (``int``) — number of samples in the train split;
          used as the split boundary index.

    Raises:
        FileNotFoundError: If ``train/`` or ``valid/`` is missing inside
                           ``client_root``.
        ValueError:        If the class lists of the two splits differ.
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

    if ds_train.classes != ds_valid.classes:
        raise ValueError(
            f"[PRUNING] Train and valid classes do not match:\n"
            f"  train: {ds_train.classes}\n  valid: {ds_valid.classes}"
        )

    ds_full = ConcatDataset([ds_train, ds_valid])
    all_samples = ds_train.samples + ds_valid.samples
    train_count = len(ds_train.samples)

    return ds_full, all_samples, ds_train.classes, train_count


# ----------------------------------------------------------------------------
# PHYSICAL IN-PLACE RECONSTRUCTION
# ----------------------------------------------------------------------------

def _rebuild_dataset_in_place(client_root, all_paths, all_labels_idx, class_names, keep_mask):
    """
    Physically replace the ``train/`` and ``valid/`` directories with only
    the samples selected by ``keep_mask``.

    The operation is performed via a temporary sibling directory
    (``<client_root>_pruning_temp``) to ensure atomicity: the original
    folders are removed only after all selected files have been
    successfully copied, preventing partial or corrupt states in the event
    of an unexpected interruption.

    Each sample's destination split (``train`` or ``valid``) is inferred
    from its original absolute path; samples whose path does not contain a
    recognizable split component are skipped and counted separately.

    Args:
        client_root:    Path to the client's dataset root.
        all_paths:      ``numpy.ndarray`` of absolute file paths, one per
                        sample, in the same order as ``keep_mask``.
        all_labels_idx: ``numpy.ndarray`` of integer class indices
                        corresponding to ``all_paths``.
        class_names:    List of class name strings indexed by
                        ``all_labels_idx``.
        keep_mask:      Boolean ``numpy.ndarray`` of shape ``(N,)``; ``True``
                        means the sample is kept, ``False`` means it is
                        discarded.

    Returns:
        A ``dict`` with the following keys:

        * ``samples_before``  — total number of samples before pruning
          (length of ``keep_mask``).
        * ``samples_after``   — total number of samples copied to the new
          directories.
        * ``reduction_pct``   — percentage of samples removed, rounded to
          two decimal places.
        * ``train_kept``      — number of train samples kept.
        * ``valid_kept``      — number of valid samples kept.
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