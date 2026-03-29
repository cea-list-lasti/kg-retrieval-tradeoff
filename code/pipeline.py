import gc
import logging
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Subset
from tqdm import tqdm

from src.config import parse_args_llama
from src.dataset.utils.retrieval import concatenate_subgraphs_2, retrieval_via_pcst_2
from src.model import llama_model_path, load_model
from src.utils.ckpt import _reload_best_model, _reload_model
from src.utils.evaluate import eval_funcs
from src.utils.lm_modeling import load_model as lm
from src.utils.lm_modeling import load_text2embedding
from src.utils.load import get_indices, load_parquet


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "sbert"


def final_prompt(question, subqa_pairs):
    context = ""
    for pair in subqa_pairs:
        context += f"{pair['question']}{pair['answer']}\n"
    return (
        "Use the given graph and question/answer pairs to answer the following question.\n"
        f"Question/answer pairs:\n{context}"
        f"Question: {question}\n"
        "Answer:"
    )


def build_paths(args):
    preprocessed = Path(args.preprocessed_dir) / args.dataset
    decomp_dataset_glob = args.decomp_dataset_glob or str(
        Path(args.decomp_datasets_dir) / args.dataset / "dataset_chunk_*.parquet"
    )
    run_name = args.decomp_run_name or f"alpha_{str(args.alpha).replace('.', '_')}"
    run_cache = preprocessed / "decomp_reasoning" / run_name
    output_dir = Path(args.output_dir) / args.dataset / run_name

    return {
        "preprocessed": preprocessed,
        "decomp_dataset_glob": decomp_dataset_glob,
        "output_dir": output_dir,
        "nodes": preprocessed / "nodes",
        "edges": preprocessed / "edges",
        "graphs": preprocessed / "graphs",
        "q_embs": preprocessed / "q_embs.pt",
        "split_indices": preprocessed / "split" / "test_indices.txt",
        "cached_graph_sub": run_cache / "cached_graphs_sub",
        "cached_desc_sub": run_cache / "cached_desc_sub",
        "cached_graph": preprocessed / "cached_graphs",
        "cached_desc": preprocessed / "cached_desc",
    }


def pipeline(args):
    paths = build_paths(args)

    os.makedirs(paths["cached_graph"], exist_ok=True)
    os.makedirs(paths["cached_desc"], exist_ok=True)
    os.makedirs(paths["cached_graph_sub"], exist_ok=True)
    os.makedirs(paths["cached_desc_sub"], exist_ok=True)
    os.makedirs(paths["output_dir"], exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Loading dataset from %s", paths["decomp_dataset_glob"])
    dataset = load_parquet(paths["decomp_dataset_glob"])
    if paths["split_indices"].exists():
        test_indices = get_indices(paths["split_indices"])
    else:
        logger.warning("Missing test split file at %s. Using sequential indices.", paths["split_indices"])
        test_indices = []

    if len(test_indices) > 0 and max(test_indices) < len(dataset):
        dataset = Subset(dataset, test_indices)
        dataset_indices = test_indices
    else:
        logger.warning(
            "Test indices do not fit loaded decomposition dataset; using sequential dataset indices."
        )
        dataset_indices = list(range(len(dataset)))
    logger.info("Pipeline input size: %s", len(dataset))

    model_emb, tokenizer, device = lm[model_name]()
    text2embedding = load_text2embedding[model_name]

    if not args.llm_model_path:
        args.llm_model_path = llama_model_path[args.llm_model_name]

    model = load_model[args.model_name](
        args=args,
        init_prompt="Your role is to answer a question using a graph.",
    )

    if args.checkpoint_path:
        model = _reload_model(model, args.checkpoint_path)
    else:
        model = _reload_best_model(model, args)
    model.eval()

    save_path = paths["output_dir"] / (
        f"model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_"
        f"llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_"
        f"max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_"
        f"patience_{args.patience}_num_epochs_{args.num_epochs}_alpha_{args.alpha}.csv"
    )
    logger.info("Saving predictions to %s", save_path)

    q_embs = torch.load(paths["q_embs"])
    all_results = []
    missing_parts = 0

    for ind, line in enumerate(tqdm(dataset)):
        index = dataset_indices[ind]
        nodes_path = paths["nodes"] / f"{index}.csv"
        edges_path = paths["edges"] / f"{index}.csv"
        graph_path = paths["graphs"] / f"{index}.pt"

        if not nodes_path.exists() or not edges_path.exists() or not graph_path.exists():
            logger.warning("Missing preprocessed files for index %s. Skipping.", index)
            missing_parts += 1
            continue

        subanswer_list = []
        q_emb = q_embs[index]
        nodes = pd.read_csv(nodes_path)
        edges = pd.read_csv(edges_path)
        graph = torch.load(graph_path)

        subquestions = line.get("subquestions", [])
        if len(subquestions) == 0:
            subquestions = [line["question"]]

        subgraphs = []
        answer = None
        question_cache_dir = paths["cached_graph_sub"] / str(index)
        desc_cache_dir = paths["cached_desc_sub"] / str(index)
        os.makedirs(question_cache_dir, exist_ok=True)
        os.makedirs(desc_cache_dir, exist_ok=True)

        for j, subquestion in enumerate(subquestions):
            text = subquestion if answer is None else answer["pred"][0] + subquestion

            sq_emb = text2embedding(model_emb, tokenizer, device, text)
            subg, desc = retrieval_via_pcst_2(
                graph,
                q_emb,
                sq_emb,
                nodes,
                edges,
                topk=3,
                topk_e=5,
                cost_e=0.5,
                alpha=float(args.alpha),
            )

            graph_cache_path = question_cache_dir / f"{j}.pt"
            desc_cache_path = desc_cache_dir / f"{j}.txt"
            torch.save(subg, graph_cache_path)
            with open(desc_cache_path, "w") as fp:
                fp.write(desc)
            subgraphs.append((subg, str(desc_cache_path)))

            sub_label = line.get("a_entity", line.get("answer", []))
            if isinstance(sub_label, list):
                sub_label = "|".join(sub_label).lower()
            else:
                sub_label = str(sub_label).lower()

            sample = {
                "id": line["id"],
                "label": sub_label,
                "subquestion": text,
                "desc": desc,
                "graph": subg,
            }
            with torch.no_grad():
                answer = model.inference_sub(sample)
            subanswer_list.append({"question": subquestion, "answer": answer["pred"][0]})

        try:
            merged_graph, merged_desc = concatenate_subgraphs_2(subgraphs)
        except Exception as exc:
            logger.warning("Failed to concatenate subgraphs at index %s: %s", index, exc)
            continue

        torch.save(merged_graph, paths["cached_graph"] / f"{index}.pt")
        with open(paths["cached_desc"] / f"{index}.txt", "w") as fp:
            fp.write(merged_desc)

        question = final_prompt(line["question"], subanswer_list)
        label = "|".join(line["answer"]).lower()
        final_sample = {
            "id": line["id"],
            "label": label,
            "subquestion": question,
            "desc": merged_desc,
            "graph": merged_graph,
        }
        with torch.no_grad():
            answer = model.inference_sub(final_sample)

        all_results.append(pd.DataFrame(answer))

    if missing_parts > 0:
        logger.info("Skipped %s entries due to missing files.", missing_parts)

    if len(all_results) == 0:
        logger.warning("No predictions were produced. Exiting without metrics.")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(save_path, index=False)
    logger.info("Done with pipeline.")

    acc, bad_calls = eval_funcs[args.dataset](save_path)
    with open(paths["output_dir"] / "bad_calls.txt", "w") as fp:
        fp.write(str(bad_calls))
    with open(paths["output_dir"] / "metrics.txt", "w") as fp:
        fp.write(str(acc))
    logger.info("Test Acc %s", acc)


if __name__ == "__main__":
    cli_args = parse_args_llama()
    pipeline(cli_args)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
    gc.collect()
