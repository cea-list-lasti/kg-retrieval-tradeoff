import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch_geometric.data import Data
from tqdm import tqdm

from src.config import parse_args_llama, resolve_hf_dataset
from src.utils.lm_modeling import load_model, load_text2embedding
from src.utils.load import load_parquet


model_name = "sbert"


def build_paths(args):
    preprocessed_path = Path(args.preprocessed_dir) / args.dataset
    dataset_glob = args.decomp_dataset_glob or str(
        Path(args.decomp_datasets_dir) / args.dataset / "dataset_chunk_*.parquet"
    )
    return {
        "preprocessed": preprocessed_path,
        "dataset_glob": dataset_glob,
        "nodes": preprocessed_path / "nodes",
        "edges": preprocessed_path / "edges",
        "graphs": preprocessed_path / "graphs",
        "subquestion_embs": preprocessed_path / "embs",
    }


def step_one(args):
    paths = build_paths(args)
    os.makedirs(paths["preprocessed"], exist_ok=True)

    dataset = load_parquet(paths["dataset_glob"])

    os.makedirs(paths["nodes"], exist_ok=True)
    os.makedirs(paths["edges"], exist_ok=True)

    for i in tqdm(range(len(dataset))):
        nodes = {}
        edges = []
        for tri in dataset[i]["graph"]:
            h, r, t = tri
            h = h.lower()
            t = t.lower()
            if h not in nodes:
                nodes[h] = len(nodes)
            if t not in nodes:
                nodes[t] = len(nodes)
            edges.append({"src": nodes[h], "edge_attr": r, "dst": nodes[t]})
        nodes_df = pd.DataFrame(
            [{"node_id": v, "node_attr": k} for k, v in nodes.items()],
            columns=["node_id", "node_attr"],
        )
        edges_df = pd.DataFrame(edges, columns=["src", "edge_attr", "dst"])

        nodes_df.to_csv(paths["nodes"] / f"{i}.csv", index=False)
        edges_df.to_csv(paths["edges"] / f"{i}.csv", index=False)


def generate_split(args):
    paths = build_paths(args)
    dataset = load_dataset(resolve_hf_dataset(args.dataset, args.datasets_dir))

    train_indices = np.arange(len(dataset["train"]))
    val_indices = np.arange(len(dataset["validation"])) + len(dataset["train"])
    if args.dataset == "cwq":
        # Keep paper setting: first 1000 samples from test split.
        test_indices = np.arange(1000) + len(dataset["train"]) + len(dataset["validation"])
    else:
        test_indices = (
            np.arange(len(dataset["test"])) + len(dataset["train"]) + len(dataset["validation"])
        )

    print("# train samples: ", len(train_indices))
    print("# val samples: ", len(val_indices))
    print("# test samples: ", len(test_indices))

    split_dir = paths["preprocessed"] / "split"
    os.makedirs(split_dir, exist_ok=True)

    with open(split_dir / "train_indices.txt", "w") as file:
        file.write("\n".join(map(str, train_indices)))

    with open(split_dir / "val_indices.txt", "w") as file:
        file.write("\n".join(map(str, val_indices)))

    with open(split_dir / "test_indices.txt", "w") as file:
        file.write("\n".join(map(str, test_indices)))


def step_two(args):
    paths = build_paths(args)
    os.makedirs(paths["preprocessed"], exist_ok=True)

    print("Cleared Memory Cache")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading dataset...")
    dataset = load_parquet(paths["dataset_glob"])
    questions = [i["question"] for i in dataset]
    subquestions = [i["subquestions"] for i in dataset]

    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    print("Encoding questions...")
    q_embs = text2embedding(model, tokenizer, device, questions)
    torch.save(q_embs, paths["preprocessed"] / "q_embs.pt")

    print("Encoding subquestions...")
    os.makedirs(paths["subquestion_embs"], exist_ok=True)
    for i, sqs in enumerate(subquestions):
        sq_path = paths["subquestion_embs"] / f"sq_embs_{i}.pt"
        if sq_path.exists():
            continue
        sq_embs = text2embedding(model, tokenizer, device, sqs)
        torch.save(sq_embs, sq_path)
    print("Finished encoding all the subquestions")

    print("Encoding graphs...")
    os.makedirs(paths["graphs"], exist_ok=True)
    for index in tqdm(range(len(dataset))):
        nodes = pd.read_csv(paths["nodes"] / f"{index}.csv")
        edges = pd.read_csv(paths["edges"] / f"{index}.csv")
        nodes.node_attr.fillna("", inplace=True)
        if len(nodes) == 0:
            print(f"Empty graph at index {index}")
            continue
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())

        edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])

        pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
        torch.save(pyg_graph, paths["graphs"] / f"{index}.pt")


if __name__ == "__main__":
    cli_args = parse_args_llama()
    step_one(cli_args)
    step_two(cli_args)
    generate_split(cli_args)
