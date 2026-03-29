# The Structure-Content Trade-off in Knowledge Graph Retrieval: A Diagnostic Study of Question Decomposition

This repository contains code for the paper **"The Structure-Content Trade-off in Knowledge Graph Retrieval: A Diagnostic Study of Question Decomposition"**.

The RAG pipeline is inspired by G-Retriever ("G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering"), whose reference implementation is available [here](https://github.com/XiaoxinHe/G-Retriever) under MIT license.

## Environment Setup

Create a Python 3.9 environment:

```bash
conda create --name kg_rag python=3.9 -y
conda activate kg_rag
```

Install PyTorch (example for CUDA 11.8):

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data And Model Paths

The code now supports configurable paths through environment variables.

You can set them in a `.env` file at repo root:

```bash
DATA_ROOT=/absolute/path/to/your/data_root
DATASETS_DIR=/absolute/path/to/your/data_root/datasets
DECOMP_DATASETS_DIR=/absolute/path/to/your/data_root/decomp_datasets
PREPROCESSED_DIR=/absolute/path/to/your/data_root/preprocessed
OUTPUT_DIR=/absolute/path/to/your/data_root/output
LLM_MODELS_PATH=/absolute/path/to/local/llm/checkpoints
```

Defaults (if unset):
- `DATA_ROOT`: repo root
- `DATASETS_DIR`: `${DATA_ROOT}/datasets`
- `DECOMP_DATASETS_DIR`: `${DATA_ROOT}/decomp_datasets`
- `PREPROCESSED_DIR`: `${DATA_ROOT}/preprocessed`
- `OUTPUT_DIR`: `${PROJECT_ROOT}/output`

## Required Downloads

1. Download RoG datasets:
- https://huggingface.co/datasets/rmanluo/RoG-cwq
- https://huggingface.co/datasets/rmanluo/RoG-webqsp

Place them as:
- `${DATASETS_DIR}/RoG-cwq`
- `${DATASETS_DIR}/RoG-webqsp`

If local folders are missing, scripts fall back to Hugging Face dataset IDs.

2. Download LLaMA-2 checkpoints (or compatible LLM checkpoints), for example:
- https://huggingface.co/meta-llama/Llama-2-7b-hf

Place them under `LLM_MODELS_PATH` using the expected layout:
- `${LLM_MODELS_PATH}/meta-llama/Llama-2-7b-hf`
- `${LLM_MODELS_PATH}/meta-llama/Llama-2-13b-hf`

## End-To-End Usage

Run commands from the `code/` directory:

```bash
cd code
```

### 1. Generate Subquestions

```bash
python -m src.utils.subquestions --dataset webqsp
python -m src.utils.subquestions --dataset cwq
```

Outputs are written to:
- `${DECOMP_DATASETS_DIR}/webqsp`
- `${DECOMP_DATASETS_DIR}/cwq`

### 2. Preprocess Graph/Text Features

```bash
python -m src.dataset.preprocess.cwq_webqsp --dataset webqsp
python -m src.dataset.preprocess.cwq_webqsp --dataset cwq
```

This creates:
- node/edge CSVs
- encoded graphs
- encoded questions/subquestions
- train/val/test index files

under `${PREPROCESSED_DIR}/{dataset}`.

### 3. Build Retrieval Caches For Training/Eval

```bash
python -m src.dataset.webqsp
python -m src.dataset.cwq
```

This populates:
- `${PREPROCESSED_DIR}/{dataset}/cached_graphs`
- `${PREPROCESSED_DIR}/{dataset}/cached_desc`

### 4. Train GraphLLM

```bash
python train.py --dataset cwq --model_name graph_llm --llm_model_name 7b
python train.py --dataset webqsp --model_name graph_llm --llm_model_name 7b
```

### 5. Run Standard Inference

```bash
python inference.py --dataset cwq --model_name graph_llm --llm_model_name 7b
```

By default this reloads the best checkpoint using the training naming convention.

### 6. Run Decomposition Pipeline

```bash
python pipeline.py \
  --dataset cwq \
  --model_name graph_llm \
  --llm_model_name 7b \
  --decomp_run_name alpha_05 \
  --alpha 0.5
```

Optional:
- pass `--checkpoint_path /abs/path/to/checkpoint.pth` to force a specific checkpoint
- pass `--decomp_dataset_glob "/abs/path/dataset_chunk_*.parquet"` to override decomposition dataset input

## Notes

- CWQ split generation keeps the paper setting of evaluating the first 1000 test samples.
- If you use a different LLM backbone, adjust projector output size in `src/model/graph_llm.py`.
