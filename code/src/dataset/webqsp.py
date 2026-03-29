import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from src.dataset.utils.retrieval import retrieval_via_pcst_2
from src.config import PREPROCESSED_DIR, DATASETS_DIR, resolve_hf_dataset

model_name = 'sbert'
path = PREPROCESSED_DIR / 'webqsp'
path_nodes = path / 'nodes'
path_edges = path / 'edges'
path_graphs = path / 'graphs'

cached_graph = path / 'cached_graphs'
cached_desc = path / 'cached_desc'


class WebQSPDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        dataset = datasets.load_dataset(resolve_hf_dataset("webqsp", DATASETS_DIR))
        self.dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        self.q_embs = torch.load(path / 'q_embs.pt')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        question = f'Question: {data["question"]}\nAnswer: '
        graph = torch.load(cached_graph / f'{index}.pt')
        with open(cached_desc / f'{index}.txt', 'r') as fp:
            desc = fp.read()
        label = ('|').join(data['answer']).lower()

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(path / 'split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(path / 'split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(path / 'split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


def preprocess():
    os.makedirs(str(cached_desc), exist_ok=True)
    os.makedirs(str(cached_graph), exist_ok=True)
    dataset = datasets.load_dataset(resolve_hf_dataset("webqsp", DATASETS_DIR))
    dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

    q_embs = torch.load(path / 'q_embs.pt')
    for index in tqdm(range(len(dataset))):
        if (cached_graph / f'{index}.pt').exists():
            continue

        nodes = pd.read_csv(path_nodes / f'{index}.csv')
        edges = pd.read_csv(path_edges / f'{index}.csv')
        if len(nodes) == 0:
            print(f'Empty graph at index {index}')
            continue
        graph = torch.load(path_graphs / f'{index}.pt')
        q_emb = q_embs[index]
        subg, desc = retrieval_via_pcst_2(graph, q_emb, nodes, edges, topk=3, topk_e=5, cost_e=0.5)
        torch.save(subg, cached_graph / f'{index}.pt')
        with open(cached_desc / f'{index}.txt', 'w') as fp:
            fp.write(desc)


if __name__ == '__main__':

    preprocess()
