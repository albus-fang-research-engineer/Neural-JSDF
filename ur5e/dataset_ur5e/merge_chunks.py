import os
import numpy as np


def concat_chunks(input_dir, prefix="mixed_dataset_chunk_", output_path="ur5e_dataset.npy"):
    """
    Concatenate chunked .npy files into a single dataset.

    Args:
        input_dir (str): Directory containing chunk files
        prefix (str): Prefix of chunk files
        output_path (str): Path to save merged dataset
    """

    # get all matching files
    files = [
        f for f in os.listdir(input_dir)
        if f.startswith(prefix) and f.endswith(".npy")
    ]

    # sort by index (important!)
    files.sort()

    if len(files) == 0:
        raise ValueError("No chunk files found.")

    print(f"Found {len(files)} chunk files")

    all_data = []

    for i, f in enumerate(files):
        path = os.path.join(input_dir, f)
        data = np.load(path)

        print(f"[{i+1}/{len(files)}] Loaded {f} | shape={data.shape}")

        all_data.append(data)

    print("Concatenating...")
    merged = np.vstack(all_data).astype(np.float32)

    print(f"Final shape: {merged.shape}")

    np.save(output_path, merged)
    print(f"Saved merged dataset to: {output_path}")


if __name__ == "__main__":
    # ===== CHANGE THESE =====
    input_dir = "./"
    output_path = "./ur5e_mixed_dataset_full.npy"

    concat_chunks(input_dir, output_path=output_path)