import os
import torch
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd

from src.config import parse_args_llama
from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.utils.collate import collate_fn
from src.utils.ckpt import _reload_best_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)


def inference(args):

    # Step 1: Load dataset

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()

    # Step 2: Build Node Classification Dataset
    test_dataset = [dataset[i] for i in idx_split['test']]
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    # Step 3: Build Model
    if not args.llm_model_path:
        args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](graph=dataset.graph, graph_type=dataset.graph_type, args=args)
    model = _reload_best_model(model, args)

    # Step 4. Evaluation
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}.csv'
    print(f'path: {path}')

    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    results = []
    for _, batch in enumerate(test_loader):
        with torch.no_grad():
            output = model.inference(batch)
            df = pd.DataFrame(output)
            results.append(df)
        progress_bar_test.update(1)

    if len(results) > 0:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv(path, index=False)
    else:
        print("No predictions were produced. Skipping metric computation.")
        return

    # Step 5. Compute Metrics
    acc = eval_funcs[args.dataset](path)
    print(f'Test Acc {acc}')


if __name__ == "__main__":

    args = parse_args_llama()

    inference(args)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
