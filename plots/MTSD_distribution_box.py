import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE_DIR = './all_mapillary'
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
LABELS_DIR = os.path.join(BASE_DIR, 'labels')
OUTPUT_PLOT_FILE = './MTSD_distribution_box.png'

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def analyze_dataset():
    print(f"Analysis dataset in: {BASE_DIR}")
    print(f" -> Images: {IMAGES_DIR}")
    print(f" -> Labels:   {LABELS_DIR}")
    print("-" * 50)

    if not os.path.exists(IMAGES_DIR) or not os.path.exists(LABELS_DIR):
        print("Error: Directory 'images' or 'labels' doesn't exist.")
        return

    all_image_paths = []
    for root, dirs, files in os.walk(IMAGES_DIR):
        for file in files:
            if file.lower().endswith(IMG_EXTENSIONS):
                all_image_paths.append(os.path.join(root, file))
    print(f"Done! {len(all_image_paths)} images found.")
    print("-" * 50)

    all_areas = []

    min_area = float('inf')
    max_area = 0.0
    min_box_info = None
    max_box_info = None

    count_missing_labels = 0
    count_images_processed = 0


    for img_path in tqdm(all_image_paths, desc="Analyzing dimensions", unit="img"):

        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]
        root_dir = os.path.dirname(img_path)

        # Path
        try:
            rel_path = os.path.relpath(root_dir, IMAGES_DIR)
            label_folder = os.path.join(LABELS_DIR, rel_path)
            label_path = os.path.join(label_folder, base_name + '.txt')
        except ValueError:
            continue

        if not os.path.exists(label_path):
            count_missing_labels += 1
            continue

        # Read image
        img = cv2.imread(img_path)
        if img is None: continue
        h_img, w_img = img.shape[:2]

        # Read labels
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            has_boxes = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])

                    # Convert to pixels
                    w_px = w_norm * w_img
                    h_px = h_norm * h_img
                    area = w_px * h_px

                    # Save area
                    all_areas.append(area)

                    # Update min/max
                    if area < min_area:
                        min_area = area
                        min_box_info = (file_name, w_px, h_px)

                    if area > max_area:
                        max_area = area
                        max_box_info = (file_name, w_px, h_px)

                    has_boxes = True

            if has_boxes:
                count_images_processed += 1

        except Exception:
            pass


    total_boxes = len(all_areas)

    print("\n" + "=" * 50)
    print(" RESULTS ANALYSIS DATASET")
    print("=" * 50)
    print(f"Processed images: {count_images_processed}")
    print(f"Missed labels:     {count_missing_labels}")
    print(f"Total Boxes:        {total_boxes}")
    print("-" * 50)

    if total_boxes > 0:

        avg_area = np.mean(all_areas)
        median_area = np.median(all_areas)
        min_val = np.min(all_areas)
        max_val = np.max(all_areas)

        side_min = math.sqrt(min_val)
        side_avg = math.sqrt(avg_area)
        side_median = math.sqrt(median_area)
        side_max = math.sqrt(max_val)

        print(f"\n[1] SMALLEST BOX:")
        if min_box_info:
            print(f"    File: {min_box_info[0]}")
            print(f"    Dim:  {min_box_info[1]:.1f} x {min_box_info[2]:.1f} px")
            print(f"    Area: {min_val:.0f} px² (side {int(side_min)}px)")

        print(f"\n[2] AVERAGE VALUES:")
        print(f"    ARITHMETIC AVERAGE: {avg_area:.0f} px² (equiv. {int(side_avg)}px)")
        print(f"    MEDIAN: {median_area:.0f} px² (equiv. {int(side_median)}px)")

        print(f"\n[3] LARGEST BOX:")
        if max_box_info:
            print(f"    File: {max_box_info[0]}")
            print(f"    Dim:  {max_box_info[1]:.1f} x {max_box_info[2]:.1f} px")
            print(f"    Area: {max_val:.0f} px² (side {int(side_max)}px)")

            # Plot
            plt.figure(figsize=(12, 6))

            # Use a serif font family throughout the plot
            plt.rcParams['font.family'] = 'serif'

            safe_min = max(1, min_val)
            bins = np.logspace(np.log10(safe_min), np.log10(max_val), 100)

            weights = np.ones_like(all_areas) / len(all_areas) * 100

            plt.hist(
                all_areas,
                bins=bins,
                weights=weights,
                color='skyblue',
                edgecolor='black',
                linewidth=0.6,
                alpha=0.8,
                label='Bounding Box Distribution'
            )

            # Log scale on x-axis
            plt.xscale('log')

            # Mean & Median vertical lines
            plt.axvline(avg_area, color='red', linestyle='dashed', linewidth=2,
                        label=f'Mean: {avg_area:.0f} px²')
            plt.axvline(median_area, color='green', linestyle='solid', linewidth=2,
                        label=f'Median: {median_area:.0f} px²')

            # Bold serif title and axis labels with slightly larger fonts
            plt.title("Traffic Sign Bounding Box Size Distribution",
                      fontsize=16, pad=20, fontweight='bold', fontfamily='serif')
            plt.xlabel("Area (square pixels, log scale)", fontsize=14, fontfamily='serif')
            plt.ylabel("Bounding Boxes (%)", fontsize=14, fontfamily='serif')

            # Remove unnecessary spines
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Increase tick label size and apply serif font
            ax.tick_params(axis='both', which='major', labelsize=12)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily('serif')

            plt.legend(frameon=False, fontsize=12, prop={'family': 'serif'})

            plt.tight_layout()
            plt.savefig(OUTPUT_PLOT_FILE, dpi=300)

    else:
        print("No boxes found.")


if __name__ == "__main__":
    analyze_dataset()


# ==================================================
#  DATASET ANALYSIS RESULTS
# ==================================================
# Processed Images: 12160
# Missing Labels:   0
# Total Boxes:      59561
# --------------------------------------------------
#
# [1] SMALLEST BOX:
#     File: 4EQQkFcs0vn16rGSYYca9w.jpg
#     Dim:  3.9 x 2.1 px
#     Area: 8 px² (square equivalent 2 px)
#
# [2] AVERAGE VALUES (The "Norm"):
#     Arithmetic Mean: 9102 px² (square equivalent 95 px)
#     Median (Typical): 1604 px² (square equivalent 40 px)
#
# [3] LARGEST BOX:
#     File: 5wE5QZgu6YLJV8lFIzdp8w.jpg
#     Dim:  1229.4 x 1217.5 px
#     Area: 1,496,870 px² (square equivalent 1223 px)