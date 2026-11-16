import os
import argparse
import random
import shutil
from PIL import Image
import sys

# Get the root directory of the project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the project root to sys.path
sys.path.append(project_root)
from utils import adjust_cols_rows


def split_data(input_dir, train_perc, val_perc, test_perc, save_dir):
    """
    Splits the dataset into training, validation, and testing sets.
    Ensures images and labels correspond correctly by name.

    Args:
        input_dir (str): Directory containing 'images' and 'labels' subfolders.
        train_perc (float): Percentage of data to use for training.
        val_perc (float): Percentage of data to use for validation.
        test_perc (float): Percentage of data to use for testing.
        save_dir (str): Root directory where train/val/test splits are written.

    Raises:
        AssertionError: If the percentages do not sum up to 1.0.
        FileNotFoundError: If input_dir does not exist.
        ValueError: If no valid files are found in input_dir.
    """
    assert (
        train_perc + val_perc + test_perc == 1.0
    ), "The sum of percentages must be 1.0"

    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "labels")

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Directories {images_dir} or {labels_dir} do not exist.")

    images = sorted([f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))])
    labels = sorted([f for f in os.listdir(labels_dir) if f.endswith(".txt")])

    if not images or not labels:
        raise ValueError("No valid files found in input_dir.")

    # Ensure correspondence by matching file names without extensions
    common_files = set(os.path.splitext(f)[0] for f in images) & set(
        os.path.splitext(f)[0] for f in labels
    )
    images = sorted([f for f in images if os.path.splitext(f)[0] in common_files])
    labels = sorted([f for f in labels if os.path.splitext(f)[0] in common_files])

    data = list(zip(images, labels))
    random.shuffle(data)

    train_end = round(train_perc * len(data))
    val_end = train_end + round(val_perc * len(data))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    save_split("train", train_data, images_dir, labels_dir, save_dir)
    save_split("val", val_data, images_dir, labels_dir, save_dir)
    save_split("test", test_data, images_dir, labels_dir, save_dir)


def save_split(split_name, split_data, images_dir, labels_dir, save_dir):
    """
    Saves the split data into corresponding train, val, or test directories.

    Args:
        split_name (str): The name of the split ('train', 'val', or 'test').
        split_data (list): List of tuples containing image and label file names.
        images_dir (str): Source directory containing image files.
        labels_dir (str): Source directory containing label files.
        save_dir (str): Root directory where splits are written.
    """
    split_images_dir = os.path.join(save_dir, split_name, "images")
    split_labels_dir = os.path.join(save_dir, split_name, "labels")

    os.makedirs(split_images_dir, exist_ok=True)
    os.makedirs(split_labels_dir, exist_ok=True)

    for image, label in split_data:
        shutil.copy(
            os.path.join(images_dir, image),
            os.path.join(split_images_dir, image),
        )
        shutil.copy(
            os.path.join(labels_dir, label),
            os.path.join(split_labels_dir, label),
        )


def tile_image(
    image_path, label_path, tile_rows, tile_cols, output_images_dir, output_labels_dir
):
    """
    Tiles an image into smaller parts and adjusts the corresponding YOLO label annotations.

    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the label file.
        tile_rows (int): Number of rows to divide the image into.
        tile_cols (int): Number of columns to divide the image into.
        output_images_dir (str): Directory to save tiled images.
        output_labels_dir (str): Directory to save adjusted label files.
    """
    image = Image.open(image_path)
    width, height = image.size
    tile_cols, tile_rows = adjust_cols_rows(width, height, tile_cols, tile_rows)
    tile_width = width // tile_cols
    tile_height = height // tile_rows

    with open(label_path, "r") as f:
        annotations = [line.strip().split() for line in f.readlines()]

    for row in range(tile_rows):
        for col in range(tile_cols):
            tile_name = (
                f"{os.path.splitext(os.path.basename(image_path))[0]}_{row}_{col}"
            )
            tile_extension = os.path.splitext(image_path)[1]  # Keep original extension
            tile_image_path = os.path.join(
                output_images_dir, f"{tile_name}{tile_extension}"
            )
            tile_label_path = os.path.join(output_labels_dir, f"{tile_name}.txt")

            # Crop and save the tile image
            left = col * tile_width
            upper = row * tile_height
            right = left + tile_width
            lower = upper + tile_height
            tile = image.crop((left, upper, right, lower))
            tile.save(tile_image_path)

            # Adjust and save the annotations for the tile
            with open(tile_label_path, "w") as label_file:
                for annotation in annotations:
                    class_id, x, y, w, h = map(float, annotation)

                    # Convert normalized coordinates to absolute coordinates
                    abs_x = x * width
                    abs_y = y * height
                    abs_w = w * width
                    abs_h = h * height

                    # Calculate object bounds
                    obj_left = abs_x - abs_w / 2
                    obj_right = abs_x + abs_w / 2
                    obj_top = abs_y - abs_h / 2
                    obj_bottom = abs_y + abs_h / 2

                    # Check if the object is within the current tile
                    if (
                        obj_right <= left
                        or obj_left >= right
                        or obj_bottom <= upper
                        or obj_top >= lower
                    ):
                        continue  # Skip objects fully outside the tile

                    # Clip object bounds to the tile
                    obj_left = max(obj_left, left)
                    obj_right = min(obj_right, right)
                    obj_top = max(obj_top, upper)
                    obj_bottom = min(obj_bottom, lower)

                    # Adjust coordinates relative to the tile
                    tile_abs_x = (obj_left + obj_right) / 2 - left
                    tile_abs_y = (obj_top + obj_bottom) / 2 - upper
                    tile_abs_w = obj_right - obj_left
                    tile_abs_h = obj_bottom - obj_top

                    # Normalize coordinates
                    tile_x = tile_abs_x / tile_width
                    tile_y = tile_abs_y / tile_height
                    tile_w = tile_abs_w / tile_width
                    tile_h = tile_abs_h / tile_height

                    label_file.write(
                        f"{int(class_id)} {tile_x} {tile_y} {tile_w} {tile_h}\n"
                    )


def tile_split_images(save_dir, tile_rows, tile_cols):
    """
    Tiles all images and adjusts labels in train, val, and test splits.

    Args:
        save_dir (str): The root directory containing train, val, and test splits.
        tile_rows (int): Number of rows to divide the image into.
        tile_cols (int): Number of columns to divide the image into.
    """
    for split_name in ["train", "val", "test"]:
        split_images_dir = os.path.join(save_dir, split_name, "images")
        split_labels_dir = os.path.join(save_dir, split_name, "labels")

        if not os.path.exists(split_images_dir) or not os.path.exists(split_labels_dir):
            print(f"Skipping {split_name}: No data found.")
            continue

        output_images_dir = os.path.join(save_dir, f"{split_name}_tiled", "images")
        output_labels_dir = os.path.join(save_dir, f"{split_name}_tiled", "labels")
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        for image_file in os.listdir(split_images_dir):
            if image_file.endswith((".jpg", ".png")):
                label_file = os.path.splitext(image_file)[0] + ".txt"
                tile_image(
                    os.path.join(split_images_dir, image_file),
                    os.path.join(split_labels_dir, label_file),
                    tile_rows,
                    tile_cols,
                    output_images_dir,
                    output_labels_dir,
                )


def write_data_yaml(save_dir: str) -> None:
    yaml_path = os.path.join(save_dir, "data.yaml")
    content = (
        f"path: {save_dir}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "\n"
        "nc: 1\n"
        "names: ['human']\n"
    )
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"Saved: {yaml_path}")


def main():
    """
    Main function to parse command-line arguments and execute the dataset split, tiling, and validation.
    """
    parser = argparse.ArgumentParser(
        description="Split dataset into train, val, and test sets, and tile images."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=r"path/to/data",
        help="Directory containing 'images' and 'labels' subfolders.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=r"path/to/data",
        help="Root directory where train/val/test splits are written.",
    )
    parser.add_argument(
        "--train_perc",
        type=float,
        default=0.85,
        help="Percentage of data to use for training",
    )
    parser.add_argument(
        "--val_perc",
        type=float,
        default=0.15,
        help="Percentage of data to use for validation",
    )
    parser.add_argument(
        "--test_perc",
        type=float,
        default=0,
        help="Percentage of data to use for testing",
    )
    parser.add_argument(
        "--tile_rows",
        type=int,
        default=1,
        help="Number of rows to divide each image into",
    )
    parser.add_argument(
        "--tile_cols",
        type=int,
        default=1,
        help="Number of columns to divide each image into",
    )

    args = parser.parse_args()

    split_data(args.input_dir, args.train_perc, args.val_perc, args.test_perc, args.save_dir)
    if args.tile_cols > 1 or args.tile_rows > 1:
        tile_split_images(args.save_dir, args.tile_rows, args.tile_cols)
    write_data_yaml(args.save_dir)


if __name__ == "__main__":
    main()
