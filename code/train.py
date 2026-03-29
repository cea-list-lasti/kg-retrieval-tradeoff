import os
import gc
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Subset
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.config import parse_args_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from src.utils.collate import collate_fn
from src.utils.lr_schedule import adjust_learning_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)

def train(args):

    path = Path(args.preprocessed_dir) / args.dataset

    # Step 1: Load dataset
    print(args)

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()

    # Handling missing graphs
    idx_train = []
    idx_val = []
    idx_test = []
    for index in idx_split["train"]:
        if (path / "cached_graphs" / f"{index}.pt").exists() and (path / "cached_desc" / f"{index}.txt").exists():
            idx_train.append(index)

    for index in idx_split["val"]:
        if (path / "cached_graphs" / f"{index}.pt").exists() and (path / "cached_desc" / f"{index}.txt").exists():
            idx_val.append(index)

    for index in idx_split["test"]:
        if (path / "cached_graphs" / f"{index}.pt").exists() and (path / "cached_desc" / f"{index}.txt").exists():
            idx_test.append(index)


    # Step 2: Build Node Classification Dataset
    train_dataset = Subset(dataset, idx_train)
    val_dataset = Subset(dataset, idx_val)
    test_dataset = Subset(dataset, idx_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    # Step 3: Build Model (LLM)
    if not args.llm_model_path:
        args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](graph_type=dataset.graph_type, args=args, init_prompt=dataset.prompt)

    # Step 4: Set Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
        betas=(0.9, 0.95)
    )
    trainable_params, all_param = model.print_trainable_params()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    # Step 5. Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float('inf')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for epoch in range(args.num_epochs):

        model.train()
        epoch_loss, accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):

            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)

            optimizer.step()
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

            if (step + 1) % args.grad_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                accum_loss = 0.

            progress_bar.update(1)

        print(f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")

        val_loss = 0.
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = model(batch)
                val_loss += loss.item()
            val_loss = val_loss/len(val_loader)
            print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        print(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

        if epoch - best_epoch >= args.patience:
            print(f'Early stop at epoch {epoch}')
            break

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()

    # Step 5: Model Evaluation
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}.csv'
    print(f'path: {path}')

    model = _reload_best_model(model, args)
    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    results = []
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            output = model.inference(batch)
            df = pd.DataFrame(output)
            results.append(df)
        progress_bar_test.update(1)

    if len(results) > 0:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv(path, index=False)
    else:
        print("No predictions were produced on the test set. Skipping metric computation.")
        return

    # Step 6: Compute Metrics
    acc = eval_funcs[args.dataset](path)
    print(f'Test Acc {acc}')


if __name__ == "__main__":

    args = parse_args_llama()

    train(args)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
    gc.collect()
