import os
import glob
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import networkx as nx
from torch_geometric.data import HeteroData
import torch
from typing import Dict, List, Tuple, Optional
import logging
import argparse

def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger.setLevel(logging.INFO)
    return logger

class GraphConstructor:
    def __init__(self, args):
        self.args = args
        self.logger = get_logger(__name__)
        self.node_mapping = {}

    def get_files(self, filename_pattern: str, root_dir: str, prefix: str = '') -> List[str]:
        """Get all files matching the pattern and prefix in root_dir"""
        files = glob.glob(os.path.join(root_dir, f'{prefix}{filename_pattern}'))
        self.logger.info(f"Found files matching pattern {prefix}{filename_pattern}: {files}")
        return files

    def process_edges(self,
                      edge_files: List[str],
                      target_node_type: str = 'transaction',
                      transaction_ids: Optional[np.ndarray] = None) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """
        Process edge files and build node mappings
        """
        self.logger.info("Processing edge files...")
        edge_dict = {}

        # Initialize target node mapping
        if target_node_type not in self.node_mapping:
            self.node_mapping[target_node_type] = {}

            # Create mapping for transaction nodes
            if transaction_ids is not None:
                for idx, trans_id in enumerate(transaction_ids):
                    self.node_mapping[target_node_type][trans_id] = idx

        for edge_file in edge_files:
            relation_name = os.path.basename(edge_file).replace('relation_', '').replace('_edgelist.csv', '')
            self.logger.info(f"Processing relation: {relation_name}")

            edges_df = pd.read_csv(edge_file)
            self.logger.info(f"Loaded {len(edges_df)} edges for relation {relation_name}")

            source_nodes = []
            target_nodes = []

            for _, row in edges_df.iterrows():
                src_id = row.iloc[0]  # transaction id
                dst_id = row.iloc[1]  # entity id

                if src_id not in self.node_mapping[target_node_type]:
                    if transaction_ids is not None:
                        continue
                    self.node_mapping[target_node_type][src_id] = len(self.node_mapping[target_node_type])
                src_idx = self.node_mapping[target_node_type][src_id]

                if relation_name not in self.node_mapping:
                    self.node_mapping[relation_name] = {}
                if dst_id not in self.node_mapping[relation_name]:
                    self.node_mapping[relation_name][dst_id] = len(self.node_mapping[relation_name])
                dst_idx = self.node_mapping[relation_name][dst_id]

                source_nodes.append(src_idx)
                target_nodes.append(dst_idx)

            if source_nodes:
                edge_dict[(target_node_type, f'has_{relation_name}', relation_name)] = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
                edge_dict[(relation_name, f'rev_has_{relation_name}', target_node_type)] = torch.tensor([target_nodes, source_nodes], dtype=torch.long)
                self.logger.info(f"Created edge tensors for relation {relation_name}")
            else:
                self.logger.warning(f"No valid edges found for relation {relation_name}")

        return edge_dict

    def create_splits(self,
                      num_nodes: int,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      stratify_labels: Optional[torch.Tensor] = None,
                      random_state: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create train/val/test masks for transductive setting with optional stratification.
        Also logs the class distribution and split sizes.
        """
        indices = np.arange(num_nodes)
        if stratify_labels is not None:
            labels_np = stratify_labels.cpu().numpy()
        else:
            labels_np = None

        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio < 0:
            raise ValueError("Train, validation, and test ratios must sum to 1 or less.")

        # splits
        temp_ratio = val_ratio + test_ratio
        train_indices, temp_indices, train_labels, temp_labels = train_test_split(
            indices, labels_np, test_size=temp_ratio, stratify=labels_np, random_state=random_state)

        temp_val_ratio = val_ratio / temp_ratio
        val_indices, test_indices, val_labels, test_labels = train_test_split(
            temp_indices, temp_labels, test_size=(test_ratio / temp_ratio), stratify=temp_labels, random_state=random_state)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        train_size = train_mask.sum().item()
        val_size = val_mask.sum().item()
        test_size = test_mask.sum().item()
        total_size = train_size + val_size + test_size
        self.logger.info(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # split checks
        assert not torch.any(train_mask & val_mask), "Train and validation masks overlap"
        assert not torch.any(train_mask & test_mask), "Train and test masks overlap"
        assert not torch.any(val_mask & test_mask), "Validation and test masks overlap"
        assert total_size == num_nodes, "Masks don't cover all nodes"

        def log_class_distribution(split_name, labels, indices):
            split_labels = labels[indices]
            unique, counts = np.unique(split_labels, return_counts=True)
            class_counts = dict(zip(unique, counts))
            total = counts.sum()
            percentages = {cls: (count / total) * 100 for cls, count in class_counts.items()}
            self.logger.info(f"{split_name} set class distribution:")
            for cls in unique:
                self.logger.info(f"  Class {cls}: {class_counts[cls]} samples, {percentages[cls]:.2f}%")

        if labels_np is not None:
            log_class_distribution("Train", labels_np, train_indices)
            log_class_distribution("Validation", labels_np, val_indices)
            log_class_distribution("Test", labels_np, test_indices)

        return train_mask, val_mask, test_mask

    def construct_pyg_graph(self,
                            features: torch.Tensor,
                            edge_dict: Dict[Tuple[str, str, str], torch.Tensor],
                            labels: Optional[torch.Tensor] = None,
                            train_mask: Optional[torch.Tensor] = None,
                            val_mask: Optional[torch.Tensor] = None,
                            test_mask: Optional[torch.Tensor] = None,
                            target_node_type: str = 'transaction') -> HeteroData:
        """
        Construct heterogeneous PyG graph
        """
        self.logger.info("Constructing PyG graph...")
        data = HeteroData()

        data[target_node_type].x = features
        if labels is not None:
            data[target_node_type].y = labels

        if train_mask is not None:
            data[target_node_type].train_mask = train_mask
        if val_mask is not None:
            data[target_node_type].val_mask = val_mask
        if test_mask is not None:
            data[target_node_type].test_mask = test_mask

        for node_type in self.node_mapping:
            if node_type == target_node_type:
                continue
            num_nodes = len(self.node_mapping[node_type])
            data[node_type].num_nodes = num_nodes
            # assign placeholder features
            # data[node_type].x = torch.zeros((num_nodes, 1))

        for (src_type, rel_type, dst_type), edge_index in edge_dict.items():
            data[(src_type, rel_type, dst_type)].edge_index = edge_index

        self.logger.info(f"Created PyG graph with node types: {data.node_types}")
        self.logger.info(f"And edge types: {data.edge_types}")
        return data

    def construct_nx_graph(self,
                           features: torch.Tensor,
                           edge_dict: Dict[Tuple[str, str, str], torch.Tensor],
                           labels: Optional[torch.Tensor] = None,
                           train_mask: Optional[torch.Tensor] = None,
                           val_mask: Optional[torch.Tensor] = None,
                           test_mask: Optional[torch.Tensor] = None,
                           target_node_type: str = 'transaction') -> nx.Graph:
        """
        Construct NetworkX graph
        """
        self.logger.info("Constructing NetworkX graph...")
        G = nx.Graph()

        for idx in range(len(features)):
            node_attrs = {
                'type': target_node_type,
                'features': features[idx].numpy()
            }
            if labels is not None:
                node_attrs['label'] = labels[idx].item()
            if train_mask is not None:
                node_attrs['train_mask'] = train_mask[idx].item()
            if val_mask is not None:
                node_attrs['val_mask'] = val_mask[idx].item()
            if test_mask is not None:
                node_attrs['test_mask'] = test_mask[idx].item()

            G.add_node(f'{target_node_type}_{idx}', **node_attrs)

        for node_type in self.node_mapping:
            if node_type == target_node_type:
                continue
            for node_id, idx in self.node_mapping[node_type].items():
                G.add_node(f'{node_type}_{idx}', type=node_type)

        for (src_type, rel_type, dst_type), edge_index in edge_dict.items():
            if rel_type.startswith('rev_'):
                continue

            edge_index_np = edge_index.numpy()
            for src_idx, dst_idx in zip(edge_index_np[0], edge_index_np[1]):
                src_node = f'{src_type}_{src_idx}'
                dst_node = f'{dst_type}_{dst_idx}'
                G.add_edge(src_node, dst_node, type=rel_type)

        self.logger.info(f"Created NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def save_graphs(self,
                    pyg_graph: HeteroData,
                    nx_graph: nx.Graph,
                    output_dir: str,
                    prefix: str = ''):
        """Save both PyG and NetworkX graphs"""
        os.makedirs(output_dir, exist_ok=True)

        torch.save(pyg_graph, os.path.join(output_dir, f'{prefix}pyg_graph.pt'))

        nx_path = os.path.join(output_dir, f'{prefix}nx_graph.gpickle')
        with open(nx_path, 'wb') as f:
            pickle.dump(nx_graph, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(output_dir, f'{prefix}node_mappings.pkl'), 'wb') as f:
            pickle.dump(self.node_mapping, f)

        self.logger.info(f"Saved graphs and mappings with prefix: {prefix}")

    def load_graphs(self, output_dir: str, prefix: str = '') -> Tuple[HeteroData, nx.Graph, Dict]:
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
        self.logger.info(f"Loading graphs with prefix: {prefix}")

        pyg_path = os.path.join(output_dir, f'{prefix}pyg_graph.pt')
        if not os.path.exists(pyg_path):
            raise FileNotFoundError(f"PyG graph file not found at: {pyg_path}")
        pyg_graph = torch.load(pyg_path)
        self.logger.info(f"Loaded PyG graph with {len(pyg_graph.node_types)} node types")

        nx_path = os.path.join(output_dir, f'{prefix}nx_graph.gpickle')
        if not os.path.exists(nx_path):
            raise FileNotFoundError(f"NetworkX graph file not found at: {nx_path}")
        with open(nx_path, 'rb') as f:
            nx_graph = pickle.load(f)
        self.logger.info(f"Loaded NetworkX graph with {nx_graph.number_of_nodes()} nodes")

        mapping_path = os.path.join(output_dir, f'{prefix}node_mappings.pkl')
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Node mappings file not found at: {mapping_path}")
        with open(mapping_path, 'rb') as f:
            node_mappings = pickle.load(f)
        self.node_mapping = node_mappings  # Update instance node_mapping
        self.logger.info(f"Loaded node mappings for {len(node_mappings)} node types")

        return pyg_graph, nx_graph, node_mappings

    def process_graph(self,
                      features: torch.Tensor,
                      edge_dict: Dict[Tuple[str, str, str], torch.Tensor],
                      labels: Optional[torch.Tensor] = None) -> Tuple[HeteroData, nx.Graph]:
        """Process and create graphs with appropriate masking"""
        if self.args.setting == 'transductive':
            train_mask, val_mask, test_mask = self.create_splits(
                num_nodes=len(features),
                train_ratio=self.args.train_ratio,
                val_ratio=self.args.val_ratio,
                stratify_labels=labels
            )

            pyg_graph = self.construct_pyg_graph(
                features=features,
                edge_dict=edge_dict,
                labels=labels,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask
            )

            nx_graph = self.construct_nx_graph(
                features=features,
                edge_dict=edge_dict,
                labels=labels,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask
            )

        else:
            pyg_graph = self.construct_pyg_graph(
                features=features,
                edge_dict=edge_dict,
                labels=labels
            )

            nx_graph = self.construct_nx_graph(
                features=features,
                edge_dict=edge_dict,
                labels=labels
            )

        return pyg_graph, nx_graph

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing preprocessed data')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save constructed graphs')
    parser.add_argument('--setting', type=str,
                        choices=['transductive', 'inductive'],
                        default='transductive',
                        help='Graph learning setting (transductive or inductive)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Training ratio for transductive setting')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation ratio for transductive setting')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--target-node-type', type=str,
                        default='transaction',
                        help='Type of target nodes')
    return parser.parse_args()

def main():
    args = parse_args()
    logger = get_logger(__name__)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    constructor = GraphConstructor(args)

    if args.setting == 'transductive':
        logger.info("Processing graph in transductive setting...")

        features_df = pd.read_csv(os.path.join(args.data_dir, 'features.csv'))
        labels_df = pd.read_csv(os.path.join(args.data_dir, 'labels.csv'))

        assert len(features_df) == len(labels_df), "Features and labels must have the same number of rows"

        if 'TransactionID' in labels_df.columns:
            features = torch.FloatTensor(features_df.values)
            labels = torch.tensor(labels_df['isFraud'].values, dtype=torch.long)
            transaction_ids = labels_df['TransactionID'].values
        else:
            raise ValueError("Labels file must contain TransactionID column")

        edge_files = constructor.get_files('relation_*_edgelist.csv', args.data_dir)
        logger.info(f"Found {len(edge_files)} relation files")

        edge_dict = constructor.process_edges(
            edge_files,
            args.target_node_type,
            transaction_ids
        )

        pyg_graph, nx_graph = constructor.process_graph(
            features=features,
            edge_dict=edge_dict,
            labels=labels
        )

        constructor.save_graphs(pyg_graph, nx_graph, args.output_dir)

        logger.info(f"PyG graph statistics:")
        for node_type in pyg_graph.node_types:
            logger.info(f"  {node_type} nodes: {pyg_graph[node_type].num_nodes}")
        for edge_type in pyg_graph.edge_types:
            logger.info(f"  {edge_type} edges: {pyg_graph[edge_type].num_edges}")

    else:
        logger.info("Processing graphs in inductive setting...")

        logger.info("Processing training graph...")
        train_features_df = pd.read_csv(os.path.join(args.data_dir, 'train_features.csv'))
        train_labels_df = pd.read_csv(os.path.join(args.data_dir, 'train_labels.csv'))

        assert len(train_features_df) == len(train_labels_df), "Training features and labels must have the same number of rows"

        if 'TransactionID' in train_labels_df.columns:
            train_features = torch.FloatTensor(train_features_df.values)
            train_labels = torch.tensor(train_labels_df['isFraud'].values, dtype=torch.long)
            train_transaction_ids = train_labels_df['TransactionID'].values
        else:
            raise ValueError("Training labels file must contain TransactionID column")

        train_edge_files = constructor.get_files('relation_*_edgelist.csv',
                                                 args.data_dir, 'train_')

        train_edge_dict = constructor.process_edges(
            train_edge_files,
            args.target_node_type,
            train_transaction_ids
        )

        train_pyg, train_nx = constructor.process_graph(
            features=train_features,
            edge_dict=train_edge_dict,
            labels=train_labels
        )

        constructor.save_graphs(train_pyg, train_nx, args.output_dir, 'train_')

        logger.info("Processing validation graph...")
        val_features_df = pd.read_csv(os.path.join(args.data_dir, 'val_features.csv'))
        val_labels_df = pd.read_csv(os.path.join(args.data_dir, 'val_labels.csv'))

        assert len(val_features_df) == len(val_labels_df), "Validation features and labels must have the same number of rows"

        if 'TransactionID' in val_labels_df.columns:
            val_features = torch.FloatTensor(val_features_df.values)
            val_labels = torch.tensor(val_labels_df['isFraud'].values, dtype=torch.long)
            val_transaction_ids = val_labels_df['TransactionID'].values
        else:
            raise ValueError("Validation labels file must contain TransactionID column")

        val_edge_files = constructor.get_files('relation_*_edgelist.csv',
                                               args.data_dir, 'val_')

        val_edge_dict = constructor.process_edges(
            val_edge_files,
            args.target_node_type,
            val_transaction_ids
        )

        val_pyg, val_nx = constructor.process_graph(
            features=val_features,
            edge_dict=val_edge_dict,
            labels=val_labels
        )

        constructor.save_graphs(val_pyg, val_nx, args.output_dir, 'val_')

        logger.info("Training graph statistics:")
        for node_type in train_pyg.node_types:
            logger.info(f"  {node_type} nodes: {train_pyg[node_type].num_nodes}")
        for edge_type in train_pyg.edge_types:
            logger.info(f"  {edge_type} edges: {train_pyg[edge_type].num_edges}")

        logger.info("Validation graph statistics:")
        for node_type in val_pyg.node_types:
            logger.info(f"  {node_type} nodes: {val_pyg[node_type].num_nodes}")
        for edge_type in val_pyg.edge_types:
            logger.info(f"  {edge_type} edges: {val_pyg[edge_type].num_edges}")

    logger.info("Graph construction completed successfully")

if __name__ == '__main__':
    main()