import cv2
import os
import glob
import shutil

from tqdm import tqdm

DATASET_PATH = "../../datasets/all_mapillary"
OUTPUT_DATASET = "../../datasets/filtered_mapillary"
LABELS_PATH = "../../datasets/filtered_mapillary/labels"

MIN_AREA_PIXELS = 1024  # 32x32 pixel

SPLITS = ["train", "val"]
EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]


def create_output_structure():
    # Output directory
    for split in SPLITS:
        os.makedirs(os.path.join(OUTPUT_DATASET, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DATASET, "labels", split), exist_ok=True)


def process_split(split):

    image_dir = os.path.join(DATASET_PATH, "images", split)
    label_dir = os.path.join(DATASET_PATH, "labels", split)

    out_image_dir = os.path.join(OUTPUT_DATASET, "images", split)
    out_label_dir = os.path.join(OUTPUT_DATASET, "labels", split)

    image_files = []
    for ext in EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))

    print(f"\n[{split}] Found {len(image_files)} images")

    for image_path in tqdm(image_files, desc="Filtering dataset", unit="img"):

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(label_dir, base_name + ".txt")

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        img_h, img_w = image.shape[:2]

        # Read labels
        with open(label_path, "r") as f:
            lines = f.readlines()

        filtered_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            w_n = float(parts[3])
            h_n = float(parts[4])

            # dimension pixel
            w_px = int(w_n * img_w)
            h_px = int(h_n * img_h)

            pixel_area = w_px * h_px

            if pixel_area >= MIN_AREA_PIXELS:
                filtered_lines.append(line.strip())

        # Save images and labels
        if len(filtered_lines) > 0:

            shutil.copy(image_path, os.path.join(out_image_dir, os.path.basename(image_path)))

            # filtered label
            out_label_path = os.path.join(out_label_dir, base_name + ".txt")
            with open(out_label_path, "w") as f:
                for l in filtered_lines:
                    f.write(l + "\n")

        # background images (without labels)
        else:
            shutil.copy(image_path, os.path.join(out_image_dir, os.path.basename(image_path)))
            open(os.path.join(out_label_dir, base_name + ".txt"), "w").close()

    print(f"[{split}] completed")


def count_empty_labels(split):

    label_dir = os.path.join(LABELS_PATH, split)
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))

    empty = 0
    total = len(txt_files)

    for txt in txt_files:
        with open(txt, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if len(lines) == 0:
            empty += 1

    print(f"[{split}] Empty labels: {empty}/{total} ({100*empty/total:.2f}%)")

    return empty, total


def main():

    create_output_structure()

    total_empty = 0
    total_files = 0

    for split in SPLITS:
        process_split(split)

        e, t = count_empty_labels(split)
        total_empty += e
        total_files += t

    print("\n========================")
    print(f"TOTAL: {total_empty}/{total_files} ({100 * total_empty / total_files:.2f}%)")


if __name__ == "__main__":
    main()


# [train] Empty Label: 1142/9823 (11.63%)
# [val] Empty Label: 268/2456 (10.91%)
#
# ========================
# TOTAL: 1410/12279 (11.48%)
