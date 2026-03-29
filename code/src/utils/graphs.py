
import re
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Subset
import torch
import networkx as nx
from torch_geometric.utils import to_networkx

from src.config import parse_args_llama
from src.utils.load import load_parquet


def resolve_dataset_path(dataset_path=None):
    if dataset_path is not None:
        return dataset_path
    args = parse_args_llama()
    return f"{args.decomp_datasets_dir}/{args.dataset}/dataset_chunk_*.parquet"


# Collection of useful functions for evaluation, additional results...



def exact_matching(test_range, path, dataset_path=None):

    dataset = load_parquet(resolve_dataset_path(dataset_path))
    dataset = Subset(dataset, test_range)

    cached_desc = f'{path}/cached_desc'

    print(len(dataset))
    matches = 0

    for i in tqdm(test_range):
        graph_text_file = f"{cached_desc}/{i}.txt"
        node_attr_map = load_graph_text(graph_text_file)  # load node_id -> node_attr mapping

        a_entity_list = [entity.lower() for entity in dataset[i - test_range[0]]["answer"]]

        for a_entity in a_entity_list:
            matched_nodes = [node_id for node_id, node_attr in node_attr_map.items() if exact_match(a_entity,node_attr)]
            matched_nodes = [node_id for node_id, node_attr in node_attr_map.items() if a_entity in node_attr] # for flexible matching via a similarity threshold

            if matched_nodes:
                matches += 1
                print(f"Found match for '{a_entity}': Node IDs {matched_nodes}")

    print(f'Presence of answer entity for our method: {matches / len(test_range)}')


def load_graph_text(file_path):
    """Parses the textual graph file and returns a dictionary {node_id: node_attr}."""
    node_attr_map = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == "src,edge_attr,dst":
                break  # stop at the edges section
            
            if ',' in line:
                node_id, node_attr = line.strip().split(',', 1)  # split at first comma
                node_attr_map[node_id] = node_attr.lower()


    return node_attr_map

def exact_match(answer, node_attr):
    """Checks if the answer is a standalone word in node_attr."""
    words = re.split(r'\W+', node_attr)  # split by non-word characters
    return answer in words


# Check graph connectivity

def is_graph_connected(data):
    G = to_networkx(data, to_undirected=True) 
    return nx.is_connected(G)

def check_connectivity(path):
    cached_graph = f'{path}/cached_graphs'
    counter = 0
    for i in tqdm(range(31158,32157)):
        graph = torch.load(f'{cached_graph}/{i}.pt')
        if is_graph_connected(graph):
            counter += 1
        else:
            print("Graph is not connected")
    print(counter)
    print(counter/999)

def graph_density(data):
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)

    if num_nodes < 2:
        return 0  # A single node or empty graph has density 0

    return (2 * num_edges) / (num_nodes * (num_nodes - 1))

def check_density(path):
    cached_graph = f'{path}/cached_graphs'
    total_density = 0
    for i in tqdm(range(31158,32157)):
        graph = torch.load(f'{cached_graph}/{i}.pt')
        density = graph_density(graph)
        total_density += density
    print(total_density/999)



def check_size(path, test_range, dataset_path=None):

    dataset = load_parquet(resolve_dataset_path(dataset_path))
    dataset = Subset(dataset, test_range)

    cached_graphs  =f'{path}/cached_graphs'

    total_size = 0

    for i in tqdm(test_range):
        graph = torch.load(f'{cached_graphs}/{i}.pt')
        size = graph.num_nodes
        total_size += size
    print(total_size/999)
