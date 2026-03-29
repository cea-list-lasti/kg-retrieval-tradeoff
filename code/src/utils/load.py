from datasets import load_dataset, concatenate_datasets
import glob
import re

def load_parquet(path):

    # To make sure the parquet files are loaded in the right order
    files = sorted(
        glob.glob(path),
        key=lambda x: int(re.search(r"dataset_chunk_(\d+)\.parquet", x).group(1))
    )
    if len(files) == 0:
        raise FileNotFoundError(f"No parquet files found for pattern: {path}")
    # Concatenate into a single Hugging Face dataset
    dataset = load_dataset("parquet", data_files=files)
    dataset = concatenate_datasets([dataset["train"]])
    return dataset

def get_indices(path):
    with open(path) as f:
        indices = [int(line.strip()) for line in f]
    return sorted(indices)
