import numpy as np
import os


def merge_npy_files(inside_path, outside_path, output_path, shuffle=True):
    """
    Merge two .npy dataset files (inside + outside) into one.

    Args:
        inside_path (str): path to inside .npy file
        outside_path (str): path to outside .npy file
        output_path (str): path to save merged .npy file
        shuffle (bool): whether to shuffle combined dataset
    """

    print(f"[INFO] Loading inside file: {inside_path}")
    inside_data = np.load(inside_path)

    print(f"[INFO] Loading outside file: {outside_path}")
    outside_data = np.load(outside_path)

    print(f"[INFO] Inside shape:  {inside_data.shape}")
    print(f"[INFO] Outside shape: {outside_data.shape}")

    # sanity check
    assert inside_data.shape[1] == outside_data.shape[1], \
        "Feature dimension mismatch between inside and outside datasets!"

    # concatenate
    combined = np.vstack((inside_data, outside_data))
    print(f"[INFO] Combined shape: {combined.shape}")

    # shuffle (recommended for training)
    if shuffle:
        print("[INFO] Shuffling dataset...")
        np.random.shuffle(combined)

    # create directory if needed
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # save
    np.save(output_path, combined)
    print(f"[INFO] Saved merged dataset to: {output_path}")


if __name__ == "__main__":
    # CHANGE THESE PATHS
    INSIDE_PATH = "inside/Inside_0373_195303.npy"
    OUTSIDE_PATH = "outside/Outside_5708_0.0_0.1_350000.npy"
    OUTPUT_PATH = "dataset_ur5e.npy"

    merge_npy_files(INSIDE_PATH, OUTSIDE_PATH, OUTPUT_PATH, shuffle=True)