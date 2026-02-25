import os
import pandas as pd
import shutil
import re
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import List, Dict


class DatasetSplitter:
    """
    Splits an image dataset into multiple client-specific subsets for federated learning.

    Automatically detects classes from the source directory structure, builds an internal
    annotation mapping, performs a stratified split, and copies image files into a directory
    structure suitable for training.
    """

    def __init__(self, output_base_dir: str, source_images_dir: str, num_clients: int):
        """
        Args:
            output_base_dir (str): Root directory where client data folders will be created.
            source_images_dir (str): Directory containing source images, with one subdirectory per class.
            num_clients (int): Number of clients to split the data for.
        """
        self.output_base_dir = output_base_dir
        self.source_images_dir = source_images_dir
        self.num_clients = num_clients
        self.class_map: Dict[str, int] = {}
        self.index_to_dir_map: Dict[int, str] = {}

        os.makedirs(self.output_base_dir, exist_ok=True)
        self.dataframe = self._build_dataframe_from_folders()

        if self.dataframe.empty:
            raise ValueError(f"No images found in source directory: {self.source_images_dir}")

    def _build_dataframe_from_folders(self) -> pd.DataFrame:
        """
        Scans the source directory to build a dataframe of filenames and class labels.
        Class indices are assigned based on numeric values in directory names when possible,
        otherwise falling back to alphabetical order.
        """
        print(f"Scanning '{self.source_images_dir}' to infer classes from directories...")

        try:
            class_dirs = [
                d for d in os.listdir(self.source_images_dir)
                if os.path.isdir(os.path.join(self.source_images_dir, d))
            ]
        except FileNotFoundError:
            print(f"Error: Source image directory not found at {self.source_images_dir}")
            raise

        if not class_dirs:
            print(f"Error: No class subdirectories found in {self.source_images_dir}")
            return pd.DataFrame()

        # Attempt to map classes using numeric values found in directory names
        numeric_map = {}
        can_use_numeric_map = True
        for dir_name in class_dirs:
            match = re.search(r'\d+', dir_name)
            if match:
                numeric_map[dir_name] = int(match.group(0))
            else:
                can_use_numeric_map = False
                break

        if can_use_numeric_map and len(set(numeric_map.values())) == len(class_dirs):
            print("Using numeric values found in directory names for class mapping.")
            self.class_map = numeric_map
        else:
            print("Could not infer classes from numbers in directory names. Falling back to alphabetical order.")
            sorted_dirs = sorted(class_dirs)
            self.class_map = {dir_name: i for i, dir_name in enumerate(sorted_dirs)}

        print("--- Class Mapping Detected ---")
        for dir_name, class_idx in self.class_map.items():
            print(f"  '{dir_name}' -> Class {class_idx}")
        print("----------------------------")

        # Build reverse mapping from class index to directory name
        self.index_to_dir_map = {v: k for k, v in self.class_map.items()}

        # Populate the dataframe with filename and class label for each image
        records = []
        for dir_name, class_idx in self.class_map.items():
            class_path = os.path.join(self.source_images_dir, dir_name)
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    records.append({'filename': filename, 'class': class_idx})

        return pd.DataFrame(records)

    def _copy_images(self, indices: List[int], client_dir: str, split_name: str) -> None:
        """
        Copies image files into the appropriate train/validation directory for a client,
        preserving the original class subdirectory structure.
        """
        for index in indices:
            row = self.dataframe.loc[index]
            filename = row['filename']
            class_index = row['class']

            class_dir_name = self.index_to_dir_map.get(class_index)
            if not class_dir_name:
                print(f"Warning: Could not find directory name for class index {class_index}. Skipping.")
                continue

            destination_dir = os.path.join(client_dir, split_name, class_dir_name)
            os.makedirs(destination_dir, exist_ok=True)

            source_image_path = os.path.join(self.source_images_dir, class_dir_name, filename)
            destination_image_path = os.path.join(destination_dir, filename)

            if os.path.exists(source_image_path):
                shutil.copy(source_image_path, destination_image_path)
            else:
                print(f"Warning: Source image not found and will be skipped: {source_image_path}")

    def _clear_existing_split(self) -> None:
        """
        Deletes all contents of the output directory to ensure a clean split.
        Warning: This is a destructive operation.
        """
        print(f"Clearing existing data in '{self.output_base_dir}'...")
        if os.path.exists(self.output_base_dir):
            for item_name in os.listdir(self.output_base_dir):
                item_path = os.path.join(self.output_base_dir, item_name)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"  - Deleted directory: {item_path}")
        print("Clearing complete.")

    def split_dataset(self, validation_split_size: float = 0.1) -> None:
        """
        Performs the main dataset splitting operation.

        Clears any previous splits, uses StratifiedKFold to divide data among clients,
        then further splits each client's data into training and validation sets.
        """
        self._clear_existing_split()

        skf = StratifiedKFold(n_splits=self.num_clients, shuffle=True, random_state=42)
        X = self.dataframe.index
        y = self.dataframe['class']

        for i, (_, client_indices) in enumerate(skf.split(X, y)):
            client_dir = os.path.join(self.output_base_dir, f"client_{i}")
            os.makedirs(client_dir, exist_ok=True)
            print(f"Processing data for client_{i}...")

            train_indices, valid_indices = train_test_split(
                client_indices,
                test_size=validation_split_size,
                random_state=42,
                stratify=y.loc[client_indices]
            )

            self._copy_images(self.dataframe.index[train_indices], client_dir, "train")
            self._copy_images(self.dataframe.index[valid_indices], client_dir, "valid")

            print(f"  - Created train set with {len(train_indices)} samples.")
            print(f"  - Created valid set with {len(valid_indices)} samples.")