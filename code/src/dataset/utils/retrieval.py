import torch
import numpy as np
from pcst_fast import pcst_fast
from torch_geometric.data.data import Data
import csv
import os


def retrieval_via_pcst_2(
    graph,
    q_emb,
    sq_emb=None,
    textual_nodes=None,
    textual_edges=None,
    topk=3,
    topk_e=5,
    cost_e=0.5,
    alpha=0.5,
):
    # Alpha interpolation used in this project:
    # alpha = 0.0 -> original-question focus
    # alpha = 1.0 -> subquestion focus

    # Backward compatibility with old call style:
    # retrieval_via_pcst_2(graph, q_emb, textual_nodes, textual_edges, ...)
    if textual_edges is None and textual_nodes is not None and not torch.is_tensor(sq_emb):
        textual_edges = textual_nodes
        textual_nodes = sq_emb
        sq_emb = None

    if sq_emb is None:
        sq_emb = q_emb

    if q_emb.dim() > 1:
        q_emb = q_emb.squeeze(0)
    if sq_emb.dim() > 1:
        sq_emb = sq_emb.squeeze(0)

    if textual_nodes is None or textual_edges is None:
        raise ValueError("textual_nodes and textual_edges must be provided.")

    c = 0.01
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    root = -1  # unrooted
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0
    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
        n_prizes_sub = torch.nn.CosineSimilarity(dim=-1)(sq_emb, graph.x)
        n_prizes_total = (1 - alpha) * n_prizes + alpha * n_prizes_sub
        topk = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes_total, topk, largest=True)

        n_prizes_total = torch.zeros_like(n_prizes_total)
        n_prizes_total[topk_n_indices] = torch.arange(topk, 0, -1).float()
    else:
        n_prizes_total = torch.zeros(graph.num_nodes)

    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.edge_attr)
        e_prizes_sub = torch.nn.CosineSimilarity(dim=-1)(sq_emb, graph.edge_attr)
        e_prizes_total = ((1 - alpha) * e_prizes + alpha * e_prizes_sub) / 2
        topk_e = min(topk_e, e_prizes_total.unique().size(0))

        topk_e_values, _ = torch.topk(e_prizes_total.unique(), topk_e, largest=True)
        e_prizes_total[e_prizes_total < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes_total == topk_e_values[k]
            value = min((topk_e-k)/sum(indices), last_topk_e_value)
            e_prizes_total[indices] = value
            last_topk_e_value = value*(1-c)
        # reduce the cost of the edges such that at least one edge is selected
        cost_e = min(cost_e, e_prizes_total.max().item()*(1-c/2))
    else:
        e_prizes_total = torch.zeros(graph.num_edges)

    costs = []
    edges = []
    vritual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.cpu().numpy()):
        prize_e = e_prizes_total[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = graph.num_nodes + len(vritual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            vritual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes_total, np.array(vritual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs+virtual_costs)
        edges = np.array(edges+virtual_edges)

    vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= graph.num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges+virtual_edges)

    if len(selected_edges) == 0:
        # Keep at least one node to avoid empty graph crashes downstream.
        selected_nodes = np.array([int(torch.argmax(n_prizes_total).item())])
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = graph.edge_attr[:0]
        n = textual_nodes.iloc[selected_nodes]
        e = textual_edges.iloc[:0]
        desc = n.to_csv(index=False) + '\n' + e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        x = graph.x[selected_nodes]
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))
        return data, desc

    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()]))

    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    desc = n.to_csv(index=False)+'\n'+e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    return data, desc


def merge_graph_files(file_paths):
    nodes = {} 
    edges = set()

    # Iterate over subgraph text descs
    for filepath in file_paths:
        if not os.path.isfile(filepath):
            continue
        
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            section = None
            
            for row in reader:
                if not row:
                    continue
                if row[0] == "node_id":  
                    section = "nodes"
                    continue
                elif row[0] == "src":
                    section = "edges"
                    continue
                
                if section == "nodes":
                    node_id, node_attr = row[0], row[1]
                    nodes[node_id] = node_attr
                
                elif section == "edges":
                    src, edge_attr, dst = row
                    edges.add((src, edge_attr, dst))

    sorted_nodes = sorted(nodes.items(), key=lambda x: int(x[0]))
    sorted_edges = sorted(edges, key=lambda x: (int(x[0]), x[1], int(x[2])))

    output_lines = []
    
    output_lines.append("node_id,node_attr")
    output_lines.extend([f"{node_id},{node_attr}" for node_id, node_attr in sorted_nodes])
    output_lines.append("")  # Blank line for separation
    output_lines.append("src,edge_attr,dst")
    output_lines.extend([f"{src},{edge_attr},{dst}" for src, edge_attr, dst in sorted_edges])

    return "\n".join(output_lines)

def concatenate_subgraphs_2(subgraphs):

    all_nodes = []
    all_edges = []
    all_edge_attrs = []
    all_desc_paths = []
    node_offset = 0
    mapping = {}  # Mapping for reindexing

    # Collect nodes + edges andd text description
    for subgraph, desc_path in subgraphs:

        all_desc_paths.append(desc_path)

        # Add Nodes
        for i in range(subgraph.num_nodes):
            mapping[node_offset + i] = len(all_nodes)
            all_nodes.append(subgraph.x[i].unsqueeze(0))

        # Add Edges
        src = subgraph.edge_index[0] + node_offset
        dst = subgraph.edge_index[1] + node_offset
        edges = torch.stack([src, dst], dim=0)
        all_edges.append(edges)
        all_edge_attrs.append(subgraph.edge_attr)

        node_offset += subgraph.num_nodes

    # Concatenate Nodes
    all_nodes = torch.cat(all_nodes, dim=0)
    unique_nodes, inverse_indices = torch.unique(all_nodes, dim=0, return_inverse=True) # remove the duplicate nodes 
    # print(f" {all_nodes.size(0) - unique_nodes.size(0)} duplicate nodes removed")

    # Remap edges for filtered nodes
    for i in range(len(all_edges)):
        all_edges[i] = inverse_indices[all_edges[i]]

    # Concatenate Edges
    all_edges = torch.cat(all_edges, dim=1)
    all_edge_attrs = torch.cat(all_edge_attrs, dim=0)

    # Remove Duplicate Edges
    edge_tuples = set()
    unique_edges = []
    unique_edge_attrs = []

    for i in range(all_edges.size(1)):
        edge = tuple(all_edges[:, i].tolist())
        if edge not in edge_tuples:
            edge_tuples.add(edge)
            unique_edges.append(edge)
            unique_edge_attrs.append(all_edge_attrs[i].unsqueeze(0))

    unique_edges = torch.tensor(unique_edges).T
    unique_edge_attrs = torch.cat(unique_edge_attrs, dim=0)

    # Create merged graph
    merged_graph = Data(
        x=unique_nodes,
        edge_index=unique_edges,
        edge_attr=unique_edge_attrs,
        num_nodes=unique_nodes.size(0),
    )

    # Create merged  desc
    merged_desc = merge_graph_files(all_desc_paths)

    return merged_graph, merged_desc
