import networkx as nx
import pickle
import os
import torch
from typing import Tuple, Dict
from torch_geometric.data import HeteroData


def load_graph(output_dir: str, prefix: str = '') -> Tuple[HeteroData, nx.Graph, Dict]:
    """
    Load saved graphs and mappings

    Args:
        output_dir: Directory where graphs were saved
        prefix: Prefix used when saving the graphs (e.g., 'train_' or 'val_')

    Returns:
        pyg_graph: Loaded PyG HeteroData graph
        nx_graph: Loaded NetworkX graph
        node_mappings: Dictionary of node mappings
    """
    pyg_path = os.path.join(output_dir, f'{prefix}pyg_graph.pt')
    if not os.path.exists(pyg_path):
        raise FileNotFoundError(f"PyG graph file not found at: {pyg_path}")
    pyg_graph = torch.load(pyg_path)

    nx_path = os.path.join(output_dir, f'{prefix}nx_graph.gpickle')
    if not os.path.exists(nx_path):
        raise FileNotFoundError(f"NetworkX graph file not found at: {nx_path}")
    with open(nx_path, 'rb') as f:
        nx_graph = pickle.load(f)

    mapping_path = os.path.join(output_dir, f'{prefix}node_mappings.pkl')
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Node mappings file not found at: {mapping_path}")
    with open(mapping_path, 'rb') as f:
        node_mappings = pickle.load(f)

    return pyg_graph, nx_graph, node_mappings