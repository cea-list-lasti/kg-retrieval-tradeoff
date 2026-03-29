import argparse
import os

from dotenv import load_dotenv
from pathlib import Path


load_dotenv()  # Loads variables from .env into os.environ

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data paths can be overridden through environment variables.
DATA_ROOT = Path(os.getenv("DATA_ROOT", str(PROJECT_ROOT)))
DATASETS_DIR = Path(os.getenv("DATASETS_DIR", str(DATA_ROOT / "datasets")))
DECOMP_DATASETS_DIR = Path(os.getenv("DECOMP_DATASETS_DIR", str(DATA_ROOT / "decomp_datasets")))
PREPROCESSED_DIR = Path(os.getenv("PREPROCESSED_DIR", str(DATA_ROOT / "preprocessed")))
LLM_MODELS_PATH = Path(os.getenv("LLM_MODELS_PATH", "/home/data/dataset/huggingface/LLMs"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "output")))
LOG_DIR = Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs")))

# Ensure directories exist
for directory in [DATASETS_DIR, DECOMP_DATASETS_DIR, PREPROCESSED_DIR, OUTPUT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def csv_list(string):
    return string.split(',')


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_hf_dataset(dataset_name, datasets_dir=None):
    base_dir = Path(datasets_dir) if datasets_dir else DATASETS_DIR
    local_path = base_dir / f"RoG-{dataset_name}"
    if local_path.exists():
        return str(local_path)
    return f"rmanluo/RoG-{dataset_name}"


def parse_args_llama():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default='graph_llm')

    parser.add_argument("--dataset", type=str, default='cwq')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--patience", type=float, default=2)

    # Model Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_steps", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))

    # Learning Rate Scheduler
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=float, default=1)

    # Inference
    parser.add_argument("--eval_batch_size", type=int, default=16)

    # LLM related
    parser.add_argument("--llm_model_name", type=str, default='7b')
    parser.add_argument("--llm_model_path", type=str, default='')
    parser.add_argument("--llm_frozen", type=str2bool, default=True)
    parser.add_argument("--llm_num_virtual_tokens", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--max_txt_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_memory", type=csv_list, default=[80, 80])

    # Data paths
    parser.add_argument("--datasets_dir", type=str, default=str(DATASETS_DIR))
    parser.add_argument("--decomp_datasets_dir", type=str, default=str(DECOMP_DATASETS_DIR))
    parser.add_argument("--preprocessed_dir", type=str, default=str(PREPROCESSED_DIR))

    # GNN related
    parser.add_argument("--gnn_model_name", type=str, default='gt')
    parser.add_argument("--gnn_num_layers", type=int, default=4)
    parser.add_argument("--gnn_in_dim", type=int, default=1024)
    parser.add_argument("--gnn_hidden_dim", type=int, default=1024)
    parser.add_argument("--gnn_num_heads", type=int, default=4)
    parser.add_argument("--gnn_dropout", type=float, default=0.0)

    # Pipeline / inference helpers
    parser.add_argument("--checkpoint_path", type=str, default='')
    parser.add_argument("--decomp_dataset_glob", type=str, default='')
    parser.add_argument("--decomp_run_name", type=str, default=os.getenv("DECOMP_RUN_NAME", "default"))
    parser.add_argument("--alpha", type=float, default=float(os.getenv("ALPHA", "0.5")))

    args = parser.parse_args()
    return args
