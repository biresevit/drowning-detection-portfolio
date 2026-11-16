import os
import shutil
import random
from dotenv import load_dotenv


def copy_folder(input_dir, save_dir):
    """
    Copies the entire folder from input_dir to save_dir.

    Parameters:
    - input_dir (str): Path to the source directory to copy.
    - save_dir (str): Path to the destination directory.

    Returns:
    - None
    """
    try:
        # Ensure the save directory does not exist; shutil.copytree requires this
        if os.path.exists(save_dir):
            print(
                f"The destination folder '{save_dir}' already exists. Removing it first."
            )
            shutil.rmtree(save_dir)  # Delete if already exists

        # Copy the folder
        shutil.copytree(input_dir, save_dir)
        print(f"Folder copied successfully from '{input_dir}' to '{save_dir}'.")
    except Exception as e:
        print(f"Error: {e}")


def split_dataset(input_dir, save_dir, perc_train, perc_val, perc_test):
    """
    Splits the dataset into train, validation, and test folders.

    Args:
        input_dir (str): Path to the input directory containing subfolders.
        save_dir (str): Path to the directory where the split datasets will be saved.
        perc_train (float): Percentage of data to be used for training.
        perc_val (float): Percentage of data to be used for validation.
        perc_test (float): Percentage of data to be used for testing.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    copy_folder(
        os.path.join(input_dir, "annotations"), os.path.join(save_dir, "annotations")
    )
    categories = [
        folder
        for folder in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, folder))
        and (folder == "drowning" or folder == "normal")
    ]
    for category_dir in categories:
        category = "drowning" if "drowning" in category_dir else "normal"
        category_path = os.path.join(input_dir, category_dir)
        videos = [
            f
            for f in os.listdir(category_path)
            if os.path.isfile(os.path.join(category_path, f))
        ]

        random.shuffle(videos)

        num_train = int(len(videos) * perc_train)
        num_val = int(len(videos) * perc_val)
        num_test = len(videos) - num_train - num_val

        datasets = {
            "train": videos[:num_train],
            "val": videos[num_train : num_train + num_val],
            "test": videos[num_train + num_val :],
        }

        for dataset, files in datasets.items():
            dataset_dir = os.path.join(save_dir, dataset, category)
            os.makedirs(dataset_dir, exist_ok=True)
            for file in files:
                src = os.path.join(category_path, file)
                dst = os.path.join(dataset_dir, file)
                shutil.copy(src, dst)

    print(f"Dataset split completed. Saved in: {save_dir}")


def parse_env():
    """
    Load environment variables from .env file and return them as a dictionary.
    """
    # Load environment variables from the .env file
    load_dotenv()

    # Fetch variables with defaults if not set in the .env file
    config = {
        "input_dir": os.getenv("INPUT_DIR"),
        "save_dir": os.getenv("SAVE_DIR"),
        "perc_train": float(os.getenv("PERC_TRAIN")),
        "perc_val": float(os.getenv("PERC_VAL")),
        "perc_test": float(os.getenv("PERC_TEST")),
    }
    return config


if __name__ == "__main__":
    args = parse_env()

    if args["perc_train"] + args["perc_val"] + args["perc_test"] != 1.0:
        raise ValueError(
            "The sum of perc_train, perc_val, and perc_test must equal 1.0"
        )

    input_dir = r"path/to/data"
    save_dir = r"path/to/data"
    split_dataset(
        input_dir, save_dir, args["perc_train"], args["perc_val"], args["perc_test"]
    )
