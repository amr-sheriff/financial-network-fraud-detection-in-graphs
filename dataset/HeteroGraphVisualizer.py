import matplotlib.colors as mcolors
import colorsys
import networkx as nx
import matplotlib.pyplot as plt
import random
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches
from typing import Optional, Dict, List, Set, Tuple
import torch
import os
import tempfile
# import pickle
import logging
from collections import defaultdict
from torch_geometric.loader import NeighborLoader


class HeteroGraphVisualizer:
    """
    A class for visualizing heterogeneous graphs with NetworkX.
    TODO: Tailored for bipartite graphs with transaction nodes and entity nodes, to be generalized for other use cases and experiments.
    """
    def __init__(self, pyg_graph, node_mappings, use_same_entity_color=False):
        self.pyg_graph = pyg_graph
        self.node_mappings = node_mappings
        self.logger = logging.getLogger('HeteroGraphVisualizer')
        self.use_same_entity_color = use_same_entity_color
        self.node_types = list(self.pyg_graph.node_types)

        self.color_scheme = self._generate_color_scheme()
        self.logger.info(f"Generated color scheme: {self.color_scheme}")

    def _generate_color_scheme(self) -> Dict[str, str]:
        """
        Generate a visually distinct color scheme for all node types in the graph.
        - Fixed color for transactions nodes.
        - Other entity types get evenly spaced colors in HSV space or the same color if specified.

        Returns:
            Dict[str, str]: A color scheme mapping node types to hex colors.
        """
        color_scheme = {'transaction': '#1f77b4'}

        other_types = [nt for nt in self.node_types if nt != 'transaction']
        n_colors = len(other_types)

        if self.use_same_entity_color:
            entity_color = '#5b806b'
            for node_type in other_types:
                color_scheme[node_type] = entity_color
        else:
            if n_colors > 0:
                HSV_tuples = [(x/n_colors, 0.8, 0.9) for x in range(n_colors)]
                RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
                hex_colors = [mcolors.rgb2hex(rgb) for rgb in RGB_tuples]

                for node_type, color in zip(other_types, hex_colors):
                    color_scheme[node_type] = color

        # Default color fallback
        color_scheme['default'] = '#7f7f7f'

        return color_scheme

    def _get_node_type_summary(self) -> str:
        """
        Get a summary of node types and their colors.

        Returns:
            str: A summary of node types and their colors.
        """
        summary = ["Node Type Color Scheme:"]
        for node_type, color in self.color_scheme.items():
            if node_type != 'default':
                count = (self.pyg_graph[node_type].num_nodes
                         if node_type in self.node_types else 0)
                summary.append(f"  • {node_type}: {color} ({count} nodes)")
        return '\n'.join(summary)

    def extract_subgraph(self,
                         n_transactions: int = 1,
                         n_hops: int = 2,
                         num_neighbors_per_hop: List[int] = [-1, 1],
                         start_transaction_indices: Optional[List[int]] = None,
                         seed: int = 42) -> Dict:
        """
        Extract an n-hop subgraph around a set of specified or random transactions sampled using NeighborLoader.

        Args:
            n_transactions (int): Number of transactions to sample.
            n_hops (int): Number of hops for the subgraph.
            num_neighbors_per_hop (List[int]): Number of neighbors to sample per hop.
            start_transaction_indices (Optional[List[int]]): List of transaction indices to start from.
            seed (int): Random seed for sampling transactions.

        Returns:
            Dict: A dictionary containing nodes and edges of the extracted subgraph.
        """
        if start_transaction_indices is not None:
            start_nodes = torch.tensor(start_transaction_indices[:n_transactions])
        else:
            torch.manual_seed(seed)
            all_trans_idx = torch.arange(self.pyg_graph['transaction'].num_nodes)
            perm = torch.randperm(len(all_trans_idx))
            start_nodes = all_trans_idx[perm[:n_transactions]]

        # num_neighbors_per_hop check
        if len(num_neighbors_per_hop) != n_hops:
            num_neighbors_per_hop = [num_neighbors_per_hop[0]] * n_hops

        loader = NeighborLoader(
            self.pyg_graph,
            num_neighbors=num_neighbors_per_hop,
            input_nodes=('transaction', start_nodes),
            batch_size=1024,
            shuffle=False
        )

        batch = next(iter(loader))

        nodes_by_type = {}
        for node_type in batch.node_types:
            # global indices
            nodes_by_type[node_type] = batch[node_type].n_id.tolist()

        edges_by_type = {}
        for edge_type in batch.edge_types:
            src_type, rel_type, dst_type = edge_type
            edge_index = batch[edge_type].edge_index

            src_local = edge_index[0]
            dst_local = edge_index[1]

            # Map local to global indices
            src_global = batch[src_type].n_id[src_local].tolist()
            dst_global = batch[dst_type].n_id[dst_local].tolist()

            edges = list(zip(src_global, dst_global))
            edges_by_type[edge_type] = set(edges)

        return {
            'nodes': nodes_by_type,
            'edges': edges_by_type
        }

    def plot_subgraph(self,
                      subgraph: Dict,
                      figsize: tuple = (23, 46),
                      with_labels: bool = True,
                      node_size: int = 1100,
                      save_path: Optional[str] = None,
                      with_legend: bool = True):
        """
        Visualize a bipartite subgraph using NetworkX with nodes colored by type, target nodes colored by class, and edges colored by source node type.
        - Transaction nodes are colored by class (0: normal, 1: fraud).
        - Entity nodes are colored by type.
        - Edges are colored by the source node type.
        - Optionally, the plot can be saved to a file.

        Args:
            subgraph (Dict): The subgraph to visualize.
            figsize (tuple): The figure size for the plot.
            with_labels (bool): Whether to display node labels.
            node_size (int): The size of nodes in the plot.
            save_path (Optional[str]): The path to save the plot.
            with_legend (bool): Whether to display the legend.

        Returns:
            None
        """
        plt.figure(figsize=figsize)
        G = nx.Graph()

        node_type_counts = defaultdict(int)
        node_labels = {}

        all_nodes = set()
        for edge_type, edges in subgraph['edges'].items():
            src_type, _, dst_type = edge_type
            for src, dst in edges:
                all_nodes.add((src_type, src))
                all_nodes.add((dst_type, dst))

        for node_type, node_idx in all_nodes:
            node_id = f"{node_type}_{node_idx}"
            bipartite_group = 0 if node_type == 'transaction' else 1
            G.add_node(node_id, node_type=node_type, bipartite=bipartite_group)
            node_type_counts[node_type] += 1

        edge_colors = []
        edge_type_counts = defaultdict(int)

        for edge_type, edges in subgraph['edges'].items():
            src_type, rel, dst_type = edge_type
            color = to_rgba(self.color_scheme.get(src_type, self.color_scheme['default']), 0.5)
            edge_type_counts[edge_type] = len(edges)

            for src, dst in edges:
                G.add_edge(
                    f"{src_type}_{src}",
                    f"{dst_type}_{dst}",
                    edge_type=rel
                )
                edge_colors.append(color)

        # bipartite sets
        top_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
        bottom_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 1]

        # bipartite layout positions
        pos = nx.bipartite_layout(G, top_nodes, align='vertical')

        node_colors = []
        for node_id, data in G.nodes(data=True):
            node_type = data.get('node_type', 'default')
            node_type_in_id, node_idx = node_id.split('_', 1)
            color = self.color_scheme.get(node_type, self.color_scheme['default'])

            label = node_id

            if node_type_in_id == 'transaction':
                idx = int(node_idx)
                try:
                    class_label = self.pyg_graph['transaction'].y[idx].item()
                except IndexError:
                    class_label = 'N/A'

                label += f"\nClass: {class_label}"

                if class_label == 1:
                    color = '#ff0000'

            node_labels[node_id] = label
            node_colors.append(color)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size)

        if with_labels:
            nx.draw_networkx_labels(G, pos, node_labels, font_size=12)

        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)

        plt.axis('off')
        plt.title("Transaction Network Subgraph", fontsize= 22)

        if with_legend:
            legend_handles = []

            if self.use_same_entity_color:
                entity_color = None

                for node_type, color in self.color_scheme.items():
                    if node_type != 'default' and node_type != 'transaction':
                        entity_color = color
                        break

                if entity_color is not None:
                    entity_patch = mpatches.Patch(color=entity_color, label='Entity Nodes')
                    legend_handles.append(entity_patch)
            else:
                for node_type, color in self.color_scheme.items():
                    if node_type != 'default' and node_type != 'transaction':
                        patch = mpatches.Patch(color=color, label=node_type)
                        legend_handles.append(patch)

            fraud_patch = mpatches.Patch(color='#ff0000', label='Fraudulent Transaction - Class 1')
            normal_patch = mpatches.Patch(color=self.color_scheme.get('transaction', '#1f77b4'), label='Benign Transaction - Class 0')
            legend_handles.extend([fraud_patch, normal_patch])

            plt.legend(handles=legend_handles, loc='lower right', fontsize=18)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_local_neighborhood(self,
                                transaction_idx: int,
                                n_hops: int = 1,
                                figsize: tuple = (23, 46),
                                save_path: Optional[str] = None,
                                with_legend: bool = True,
                                node_size: int = 1100,
                                with_labels: bool = True,
                                num_neighbors_per_hop: List[int] = [-1]):
        """
        Visualize the n-hops local neighborhood of a specific transaction node in the graph.

        Args:
            transaction_idx (int): The index of the transaction node to start from.
            n_hops (int): Number of hops for the subgraph.
            figsize (tuple): The figure size for the plot.
            save_path (Optional[str]): The path to save the plot.
            with_legend (bool): Whether to display the legend.
            node_size (int): The size of nodes in the plot.
            with_labels (bool): Whether to display node labels.
            num_neighbors_per_hop (List[int]): Number of neighbors to sample per hop.

        Returns:
            None
        """
        subgraph = self.extract_subgraph(
            n_transactions=1,
            n_hops=n_hops,
            start_transaction_indices=[transaction_idx],
            num_neighbors_per_hop=num_neighbors_per_hop
        )

        self.plot_subgraph(
            subgraph,
            figsize=figsize,
            with_labels=with_labels,
            node_size=node_size,
            save_path=save_path,
            with_legend=with_legend
        )

    def get_top_high_degree_entities(self, subgraph: Dict, top_n: int, class_label: int) -> Set[Tuple[str, int]]:
        """
        Get the top N highest-degree entity nodes in the subgraph, based on connections to transactions of a specific class.

        Parameters:
            subgraph (Dict): The subgraph from which to compute entity degrees.
            top_n (int): Number of top entities to select.
            class_label (int): The class label (0 for normal, 1 for fraud) to consider when counting degrees.

        Returns:
            Set[Tuple[str, int]]: A set of (node_type, node_idx) tuples representing the top entities.
        """
        entity_degrees = defaultdict(int)

        subgraph_transaction_labels = {}
        for idx in subgraph['nodes']['transaction']:
            subgraph_transaction_labels[idx] = self.pyg_graph['transaction'].y[idx].item()

        # count degrees of entities connected to transactions of the specified class
        for edge_type, edges in subgraph['edges'].items():
            src_type, _, dst_type = edge_type
            for src, dst in edges:
                if src_type == 'transaction' and subgraph_transaction_labels.get(src) == class_label:
                    if dst_type != 'transaction':
                        entity_degrees[(dst_type, dst)] += 1
                elif dst_type == 'transaction' and subgraph_transaction_labels.get(dst) == class_label:
                    if src_type != 'transaction':
                        entity_degrees[(src_type, src)] += 1

        # Sort by degree desc
        sorted_entities = sorted(entity_degrees.items(), key=lambda x: x[1], reverse=True)

        top_entities = set()
        for (node_type, node_idx), degree in sorted_entities[:top_n]:
            top_entities.add((node_type, node_idx))

        return top_entities

    def filter_subgraph_to_top_entities(self, subgraph: Dict, top_entities: Set[Tuple[str, int]]) -> Dict:
        """
        Filter the subgraph to include only the top entities and all transactions.

        Parameters:
            subgraph (Dict): The original subgraph.
            top_entities (Set[Tuple[str, int]]): Set of top entities to include.

        Returns:
            Dict: The filtered subgraph.
        """
        new_subgraph = {'nodes': {'transaction': set(subgraph['nodes']['transaction'])}, 'edges': {}}
        for node_type in self.node_types:
            if node_type != 'transaction':
                new_subgraph['nodes'][node_type] = set()

        for edge_type, edges in subgraph['edges'].items():
            src_type, _, dst_type = edge_type
            new_edges = set()
            for src, dst in edges:
                include_edge = False
                if src_type == 'transaction' and dst_type == 'transaction': # TODO no such edges in this bipartite graph will be used for other experiments
                    include_edge = True
                else:
                    if src_type != 'transaction' and (src_type, src) in top_entities:
                        include_edge = True
                    if dst_type != 'transaction' and (dst_type, dst) in top_entities:
                        include_edge = True
                if include_edge:
                    new_edges.add((src, dst))
                    new_subgraph['nodes'][src_type].add(src)
                    new_subgraph['nodes'][dst_type].add(dst)
            if new_edges:
                new_subgraph['edges'][edge_type] = new_edges

        # Convert node sets to lists
        for node_type in new_subgraph['nodes']:
            new_subgraph['nodes'][node_type] = list(new_subgraph['nodes'][node_type])

        return new_subgraph

    def plot_fraud_patterns(self,
                            n_fraud: int = 3,
                            n_normal: int = 3,
                            n_hops: int = 2,
                            figsize: tuple = (23, 46),
                            node_size: int = 1100,
                            with_labels: bool = True,
                            save_path: Optional[str] = None,
                            seed: int = 42,
                            num_neighbors_per_hop: List[int] = [-1, -1],
                            with_legend: bool = True,
                            top_entity_degree: Optional[int] = None):
        """
        Compare graph patterns between fraudulent and normal transactions by plotting their n-hop local neighborhoods with the highest-degree entities.

        Args:
            n_fraud (int): Number of fraudulent transactions to sample.
            n_normal (int): Number of normal transactions to sample.
            n_hops (int): Number of hops for the subgraph.
            figsize (tuple): The figure size for the plot.
            node_size (int): The size of nodes in the plot.
            with_labels (bool): Whether to display node labels.
            save_path (Optional[str]): The path to save the plot.
            seed (int): Random seed for sampling transactions.
            num_neighbors_per_hop (List[int]): Number of neighbors to sample per hop.
            with_legend (bool): Whether to display the legend.
            top_entity_degree (Optional[int]): If specified, only the top N highest-degree entity nodes
            connected to transactions of the specified class will be included in the plots.

        Returns:
            None
        """
        labels = self.pyg_graph['transaction'].y
        fraud_idx = (labels == 1).nonzero().flatten().tolist()
        normal_idx = (labels == 0).nonzero().flatten().tolist()

        random.seed(seed)
        sampled_fraud = random.sample(fraud_idx, min(n_fraud, len(fraud_idx)))
        sampled_normal = random.sample(normal_idx, min(n_normal, len(normal_idx)))

        fraud_subgraph = self.extract_subgraph(
            n_transactions=len(sampled_fraud),
            n_hops=n_hops,
            start_transaction_indices=sampled_fraud,
            num_neighbors_per_hop=num_neighbors_per_hop
        )

        normal_subgraph = self.extract_subgraph(
            n_transactions=len(sampled_normal),
            n_hops=n_hops,
            start_transaction_indices=sampled_normal,
            num_neighbors_per_hop=num_neighbors_per_hop
        )

        if top_entity_degree is not None:
            # Fraudulent transactions (class=1)
            top_fraud_entities = self.get_top_high_degree_entities(fraud_subgraph, top_entity_degree, class_label=1)
            fraud_subgraph = self.filter_subgraph_to_top_entities(fraud_subgraph, top_fraud_entities)

            # Benign transactions (class=0)
            top_normal_entities = self.get_top_high_degree_entities(normal_subgraph, top_entity_degree, class_label=0)
            normal_subgraph = self.filter_subgraph_to_top_entities(normal_subgraph, top_normal_entities)

        with tempfile.TemporaryDirectory() as tmpdir:
            fraud_path = os.path.join(tmpdir, 'fraud.png')
            normal_path = os.path.join(tmpdir, 'normal.png')

            self.plot_subgraph(
                fraud_subgraph,
                with_legend=with_legend,
                node_size=node_size,
                with_labels=with_labels,
                figsize=figsize,
                save_path=fraud_path
            )

            self.plot_subgraph(
                normal_subgraph,
                node_size=node_size,
                with_labels=with_labels,
                figsize=figsize,
                with_legend=with_legend,
                save_path=normal_path
            )

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            fraud_img = plt.imread(fraud_path)
            normal_img = plt.imread(normal_path)
            ax1.imshow(fraud_img)
            ax2.imshow(normal_img)
            ax1.set_title("Fraudulent Transaction Patterns", fontsize=22)
            ax2.set_title("Normal Transaction Patterns", fontsize=22)
            ax1.axis('off')
            ax2.axis('off')
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
                plt.close(fig)
            else:
                plt.show()
            plt.close(fig)

        return fig
