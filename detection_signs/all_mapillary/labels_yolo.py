import json
from pathlib import Path

input_folder = Path('../../datasets/all_mapillary/annotations')  # Folder containing JSON files
output_folder = Path('../../datasets/all_mapillary/labels')  # Folder where YOLO TXT files will be created


def convert_to_yolo():
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Starting conversion from '{input_folder}' to '{output_folder}'...")

    converted_count = 0
    empty_count = 0

    files = list(input_folder.glob('*.json'))

    if not files:
        print("ERROR: No .json files found in the specified folder!")
        return

    for json_file in files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Image fundamental data
            img_width = data['width']
            img_height = data['height']

            yolo_lines = []

            # If "objects" list is empty, loop does not run and creates empty file (correct for YOLO)
            for obj in data.get('objects', []):

                # Here you can decide whether to include everything or only specific labels.
                # Currently we include everything and assign class 0.

                bbox = obj['bbox']
                xmin = bbox['xmin']
                ymin = bbox['ymin']
                xmax = bbox['xmax']
                ymax = bbox['ymax']

                # --- YOLO CALCULATIONS (0–1 normalization) ---
                # 1. Compute width and height
                box_w = xmax - xmin
                box_h = ymax - ymin

                # 2. Compute box center
                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0

                # 3. Normalize using image size
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                width_norm = box_w / img_width
                height_norm = box_h / img_height

                # 4. Format YOLO string: "0 x_center y_center width height"
                # We always use class "0" for generic traffic sign
                line = f"0 {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
                yolo_lines.append(line)

            # Write the .txt file
            # Filename must match JSON name (but with .txt extension)
            txt_filename = json_file.with_suffix('.txt').name
            output_path = output_folder / txt_filename

            with open(output_path, 'w') as out_f:
                if yolo_lines:
                    out_f.write('\n'.join(yolo_lines))
                else:
                    # If no objects exist, create empty file (YOLO background image)
                    pass

            if not yolo_lines:
                empty_count += 1

            converted_count += 1

        except Exception as e:
            print(f"Error processing file {json_file.name}: {e}")

    print("-" * 30)
    print("CONVERSION COMPLETED.")
    print(f"Processed files: {converted_count}")
    print(f"Empty files (no objects): {empty_count}")
    print(f"YOLO labels are ready in: {output_folder}")


if __name__ == "__main__":
    convert_to_yolo()
