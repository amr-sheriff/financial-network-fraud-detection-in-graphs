import os
import pickle
import torch
from HeteroGraphVisualizer import HeteroGraphVisualizer

def visualize_graph(data_dir: str, prefix: str = ''):
    pyg_graph = torch.load(os.path.join(data_dir, f'{prefix}pyg_graph.pt'))
    with open(os.path.join(data_dir, f'{prefix}node_mappings.pkl'), 'rb') as f:
        node_mappings = pickle.load(f)

    viz = HeteroGraphVisualizer(pyg_graph, node_mappings, use_same_entity_color=True)

    viz_dir = os.path.join(data_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # random subgraph w\ 1 transaction and 2 hops
    subgraph = viz.extract_subgraph(n_transactions=1, n_hops=2)
    viz.plot_subgraph(
        subgraph,
        save_path=os.path.join(viz_dir, f'{prefix}random_subgraph.png'),
        with_labels=True
    )

    # local neighborhood of a specific transaction w\ 2 hops
    viz.plot_local_neighborhood(
        transaction_idx=2,
        n_hops=2,
        save_path=os.path.join(viz_dir, f'{prefix}transaction_2_neighborhood.png')
    )

    # Plot fraud vs normal patterns comparison
    viz.plot_fraud_patterns(
        n_fraud=3,
        n_normal=3,
        n_hops=2,
        num_neighbors_per_hop=[-1, 10],
        top_entity_degree=10,
        save_path=os.path.join(viz_dir, f'{prefix}fraud_patterns.png')
    )

    # random subgraph with 1 transaction and 1 hop
    subgraph = viz.extract_subgraph(n_transactions=1, n_hops=1, num_neighbors_per_hop=[-1])
    viz.plot_subgraph(
        subgraph,
        save_path=os.path.join(viz_dir, f'{prefix}random_subgraph_1_hop.png'),
        with_labels=True
    )

    # local neighborhood of a specific transaction w\ 1 hop
    viz.plot_local_neighborhood(
        transaction_idx=2,
        n_hops=1,
        num_neighbors_per_hop=[-1],
        save_path=os.path.join(viz_dir, f'{prefix}transaction_2_neighborhood_1_hop.png')
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='')
    args = parser.parse_args()

    visualize_graph(args.data_dir, args.prefix)
