import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import MessagePassing
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
import numpy as np
import copy
import pickle

from deepsnap.hetero_graph import HeteroGraph
from deepsnap.hetero_gnn import forward_op, HeteroConv

class HeteroGNNConv(MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, out_channels):
        super(HeteroGNNConv, self).__init__(aggr="mean")

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.out_channels = out_channels

        self.lin_src = nn.Linear(in_channels_src, out_channels)
        self.lin_dst = nn.Linear(in_channels_dst, out_channels)
        self.lin_update = nn.Linear(out_channels * 2, out_channels)

    def forward(self, node_feature_src, node_feature_dst, edge_index, size=None):
        return self.propagate(edge_index, node_feature_dst=node_feature_dst, node_feature_src=node_feature_src, size=size)

    def message_and_aggregate(self, edge_index, node_feature_src):
        out = edge_index.matmul(node_feature_src, reduce='mean')
        return out

    def update(self, aggr_out, node_feature_dst):
        hv = self.lin_dst(node_feature_dst)
        hu_aggr = self.lin_src(aggr_out)
        aggr_out = self.lin_update(torch.cat([hv, hu_aggr], dim=1))
        return aggr_out

class HeteroGNNWrapperConv(HeteroConv):
    def __init__(self, convs, args, aggr="mean"):
        super(HeteroGNNWrapperConv, self).__init__(convs, None)
        self.aggr = aggr
        self.mapping = {}
        self.alpha = None
        self.attn_proj = None

        if self.aggr == "attn":
            self.attn_proj = nn.Sequential(
                nn.Linear(args['hidden_size'], args['attn_size']),
                nn.Tanh(),
                nn.Linear(args['attn_size'], 1, bias=False)
            )

    def reset_parameters(self):
        super(HeteroGNNWrapperConv, self).reset_parameters()
        if self.aggr == "attn":
            for layer in self.attn_proj.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def forward(self, node_features, edge_indices):
        message_type_emb = {}
        for message_key, edge_index in edge_indices.items():
            src_type, edge_type, dst_type = message_key
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]
            message_type_emb[message_key] = self.convs[message_key](
                node_feature_src,
                node_feature_dst,
                edge_index,
            )

        node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
        mapping = {}
        for (src, edge_type, dst), item in message_type_emb.items():
            mapping[len(node_emb[dst])] = (src, edge_type, dst)
            node_emb[dst].append(item)
        self.mapping = mapping
        for node_type, embs in node_emb.items():
            if len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)
        return node_emb

    def aggregate(self, xs):
        if self.aggr == "mean":
            return torch.mean(torch.stack(xs, dim=0), dim=0)
        elif self.aggr == "attn":
            N = xs[0].shape[0]  # num nodes
            M = len(xs)
            x = torch.cat(xs, dim=0).view(M, N, -1)
            z = self.attn_proj(x).view(M, N)
            z = z.mean(1) # M * 1
            alpha = torch.softmax(z, dim=0)
            self.alpha = alpha.cpu().numpy()
            alpha = alpha.view(M, 1, 1)
            x = x * alpha
            return x.sum(dim=0)

def generate_convs(hetero_graph, conv, hidden_size, first_layer=False):
    convs = {}
    for message_type in hetero_graph.message_types:
        s, r, d = message_type
        if first_layer:
            in_src = hetero_graph.num_node_features(s)
            in_dst = hetero_graph.num_node_features(d)
        else:
            in_src = hidden_size
            in_dst = hidden_size
        convs[message_type] = conv(in_src, in_dst, hidden_size)
    return convs

class HeteroGNN(nn.Module):
    def __init__(self, hetero_graph, args, aggr="mean"):
        super(HeteroGNN, self).__init__()
        self.aggr = aggr
        self.hidden_size = args['hidden_size']

        self.convs1 = HeteroGNNWrapperConv(generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=True), args, aggr=self.aggr)
        self.convs2 = HeteroGNNWrapperConv(generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=False), args, aggr=self.aggr)

        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()
        self.post_mps = nn.ModuleDict()

        for node_type in hetero_graph.node_types:
            self.bns1[node_type] = nn.BatchNorm1d(self.hidden_size, eps=1)
            self.bns2[node_type] = nn.BatchNorm1d(self.hidden_size, eps=1)
            self.relus1[node_type] = nn.LeakyReLU()
            self.relus2[node_type] = nn.LeakyReLU()
            self.post_mps[node_type] = nn.Linear(self.hidden_size, hetero_graph.num_node_labels(node_type))

    def forward(self, node_feature, edge_index):
        x = node_feature
        x = self.convs1(x, edge_index)
        x = forward_op(x, self.bns1)
        x = forward_op(x, self.relus1)

        x = self.convs2(x, edge_index)
        x = forward_op(x, self.bns2)
        x = forward_op(x, self.relus2)

        x = forward_op(x, self.post_mps)
        return x

    def loss(self, preds, y, indices):
        loss = 0
        loss_func = F.cross_entropy
        for node_type in preds:
            loss += loss_func(preds[node_type][indices[node_type]], y[node_type][indices[node_type]])
        return loss

def train(model, optimizer, hetero_graph, train_idx):
    model.train()
    optimizer.zero_grad()
    preds = model(hetero_graph.node_feature, hetero_graph.edge_index)
    loss = model.loss(preds, hetero_graph.node_label, train_idx)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, graph, indices):
    model.eval()
    with torch.no_grad():
        preds = model(graph.node_feature, graph.edge_index)
        results = []

        # each split train, val, test
        for i, index in enumerate(indices):
            micro = 0
            macro = 0
            num_target_types = 0

            # evaluate target node only - transaction type
            for node_type in preds:
                if node_type == 'transaction' and node_type in index:
                    idx = index[node_type]
                    pred = preds[node_type][idx].max(1)[1]
                    label_np = graph.node_label[node_type][idx].cpu().numpy()
                    pred_np = pred.cpu().numpy()
                    micro += f1_score(label_np, pred_np, average='micro')
                    macro += f1_score(label_np, pred_np, average='macro')
                    num_target_types += 1

            if num_target_types > 0:
                micro /= num_target_types
                macro /= num_target_types
            results.append((micro, macro))

        return results

def load_and_prepare_data(args, subset_size=None):
    """
    Load and prepare the data with optional subset sampling
    Args:
        args: Dictionary containing model arguments
        subset_size: Optional float between 0 and 1 indicating what fraction of data to use
    """
    print("Device:", args['device'])

    with open("graphs/pyg_graph.pt", 'rb') as f:
        pyg_data = torch.load(f)

    target_node_type = 'transaction'

    train_mask = pyg_data[target_node_type].train_mask
    val_mask = pyg_data[target_node_type].val_mask
    test_mask = pyg_data[target_node_type].test_mask
    target_labels = pyg_data[target_node_type].y

    if subset_size is not None and 0 < subset_size < 1:
        train_indices = train_mask.nonzero().view(-1)
        num_train = int(len(train_indices) * subset_size)
        perm = torch.randperm(len(train_indices))
        sampled_train_indices = train_indices[perm[:num_train]]
        new_train_mask = torch.zeros_like(train_mask)
        new_train_mask[sampled_train_indices] = True
        train_mask = new_train_mask

        val_indices = val_mask.nonzero().view(-1)
        num_val = int(len(val_indices) * subset_size)
        perm = torch.randperm(len(val_indices))
        sampled_val_indices = val_indices[perm[:num_val]]
        new_val_mask = torch.zeros_like(val_mask)
        new_val_mask[sampled_val_indices] = True
        val_mask = new_val_mask

        test_indices = test_mask.nonzero().view(-1)
        num_test = int(len(test_indices) * subset_size)
        perm = torch.randperm(len(test_indices))
        sampled_test_indices = test_indices[perm[:num_test]]
        new_test_mask = torch.zeros_like(test_mask)
        new_test_mask[sampled_test_indices] = True
        test_mask = new_test_mask

        print(f"Using {subset_size*100}% of data:")
        print(f"Train samples: {train_mask.sum().item()}")
        print(f"Val samples: {val_mask.sum().item()}")
        print(f"Test samples: {test_mask.sum().item()}")

    y_train = target_labels[train_mask].cpu().numpy()
    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(
        [sum(class_counts)/c for c in class_counts],
        dtype=torch.float
    ).to(args['device'])

    train_idx = {target_node_type: train_mask.nonzero().view(-1)}
    val_idx = {target_node_type: val_mask.nonzero().view(-1)}
    test_idx = {target_node_type: test_mask.nonzero().view(-1)}

    edge_index = {}
    for edge_type in pyg_data.edge_types:
        src, rel, dst = edge_type
        edge_index[edge_type] = pyg_data[edge_type].edge_index

    node_feature = {}
    num_nodes_dict = {}

    for node_type in pyg_data.node_types:
        if hasattr(pyg_data[node_type], 'x'):
            node_feature[node_type] = pyg_data[node_type].x
            num_nodes_dict[node_type] = pyg_data[node_type].x.size(0)
        elif hasattr(pyg_data[node_type], 'num_nodes'):
            num_nodes = pyg_data[node_type].num_nodes
            num_nodes_dict[node_type] = num_nodes
            feature_dim = pyg_data[target_node_type].x.size(1)
            node_feature[node_type] = torch.zeros((num_nodes, feature_dim), dtype=torch.float)

    node_label = {}
    for node_type in pyg_data.node_types:
        if node_type == target_node_type:
            node_label[node_type] = target_labels
        else:
            num_nodes = num_nodes_dict[node_type]
            num_classes = len(class_counts)
            node_label[node_type] = torch.zeros(num_nodes, dtype=torch.long)

    hetero_graph = HeteroGraph(
        node_feature=node_feature,
        node_label=node_label,
        edge_index=edge_index,
        directed=True
    )

    print(f"Heterogeneous graph: {hetero_graph.num_nodes()} nodes, {hetero_graph.num_edges()} edges")

    for node_type in hetero_graph.node_types:
        print(f"Node type: {node_type}")
        print(f"  Features shape: {hetero_graph.node_feature[node_type].shape}")
        print(f"  Labels shape: {hetero_graph.node_label[node_type].shape}")
        if node_type == target_node_type:
            print(f"  Number of classes: {len(torch.unique(hetero_graph.node_label[node_type]))}")

    for key in hetero_graph.node_feature:
        hetero_graph.node_feature[key] = hetero_graph.node_feature[key].to(args['device'])
    for key in hetero_graph.node_label:
        hetero_graph.node_label[key] = hetero_graph.node_label[key].to(args['device'])

    for key in hetero_graph.edge_index:
        edge_index = hetero_graph.edge_index[key]
        src_type, _, dst_type = key
        adj = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            sparse_sizes=(
                hetero_graph.num_nodes(src_type),
                hetero_graph.num_nodes(dst_type)
            )
        )
        hetero_graph.edge_index[key] = adj.t().to(args['device'])

    return hetero_graph, train_idx, val_idx, test_idx, class_weights

def train_batch(model, optimizer, hetero_graph, train_idx, batch_size=6144):
    model.train()
    optimizer.zero_grad()

    total_loss = 0
    num_batches = 0

    transaction_nodes = train_idx['transaction']
    total_batches = (len(transaction_nodes) + batch_size - 1) // batch_size

    perm = torch.randperm(len(transaction_nodes))

    for i in range(0, len(transaction_nodes), batch_size):
        batch_nodes = transaction_nodes[perm[i:i + batch_size]]

        batch_train_idx = {}
        for node_type in hetero_graph.node_types:
            if node_type == 'transaction':
                batch_train_idx[node_type] = batch_nodes
            else:
                num_nodes = hetero_graph.node_label[node_type].size(0)
                batch_train_idx[node_type] = torch.arange(num_nodes, device=batch_nodes.device)

        preds = model(hetero_graph.node_feature, hetero_graph.edge_index)

        loss = model.loss(preds, hetero_graph.node_label, batch_train_idx)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

        current_batch = num_batches
        print(f"\rProcessing batch {current_batch}/{total_batches} "
              f"({(current_batch/total_batches)*100:.1f}%) - "
              f"Current loss: {loss.item():.4f}", end="")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()

    return total_loss / num_batches

def main(subset_size=None):
    args = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'hidden_size': 64,
        'epochs': 6,
        'weight_decay': 1e-5,
        'lr': 0.003,
        'attn_size': 32,
        'batch_size': 6144,
    }

    target_node_type = 'transaction'

    hetero_graph, train_idx, val_idx, test_idx, class_weights = load_and_prepare_data(args, subset_size=subset_size)

    model = HeteroGNN(hetero_graph, args, aggr="mean").to(args['device'])

    original_loss = model.loss
    def weighted_loss(preds, y, indices):
        loss = 0
        for node_type in preds:
            if node_type == 'transaction':  # class weights applied to target node type
                loss += F.cross_entropy(
                    preds[node_type][indices[node_type]],
                    y[node_type][indices[node_type]],
                    weight=class_weights
                )
            else:
                loss += F.cross_entropy(
                    preds[node_type][indices[node_type]],
                    y[node_type][indices[node_type]]
                )
        return loss
    model.loss = weighted_loss

    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    best_val = 0
    best_model = None

    for epoch in range(args['epochs']):
        print("Training started")
        # loss = train(model, optimizer, hetero_graph, train_idx)
        loss = train_batch(model, optimizer, hetero_graph, train_idx, batch_size=args['batch_size'])
        accs = evaluate(model, hetero_graph, [train_idx, val_idx, test_idx])

        train_acc = accs[0][0]
        val_acc = accs[1][0]
        test_acc = accs[2][0]

        if val_acc > best_val:
            best_val = val_acc
            best_model = copy.deepcopy(model)

        print(f"Epoch {epoch+1}/{args['epochs']}, Loss: {loss:.4f}, Train Micro F1: {train_acc*100:.2f}%, Val Micro F1: {val_acc*100:.2f}%, Test Micro F1: {test_acc*100:.2f}%")

    best_accs = evaluate(best_model, hetero_graph, [train_idx, val_idx, test_idx])
    print(f"Best Val Micro F1: {best_val*100:.2f}%, Test Micro F1 at Best: {best_accs[2][0]*100:.2f}%")

    best_model.eval()
    with torch.no_grad():
        preds = best_model(hetero_graph.node_feature, hetero_graph.edge_index)
    pred = preds[target_node_type][test_idx[target_node_type]].softmax(dim=1)
    y_true_test = hetero_graph.node_label[target_node_type][test_idx[target_node_type]].cpu().numpy()
    y_pred_test = pred.argmax(dim=1).cpu().numpy()

    if pred.size(1) == 2:
        y_score = pred[:,1].cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_true_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %.2f)' % roc_auc)
        plt.plot([0,1],[0,1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig('roc_curve.png')
        plt.show()

    cm = confusion_matrix(y_true_test, y_pred_test)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = [f"Class {i}" for i in range(cm.shape[0])]
    plt.xticks(np.arange(cm.shape[0]), classes, rotation=45)
    plt.yticks(np.arange(cm.shape[0]), classes)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    main()
