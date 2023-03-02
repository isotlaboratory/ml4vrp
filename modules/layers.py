import torch
import torch.nn as nn
import torch_geometric
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self):
        super().__init__("add")
        emb_size = 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output


class GCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)
        return output


class GAT(torch.nn.Module):
    """Graph Attention Network"""

    def __init__(self, nfeat, nhid, nclass, dropout, nheads, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.alpha = alpha

        self.gat1 = GATv2Conv(nfeat, nhid, heads=nheads, add_self_loops=False)
        self.gat2 = GATv2Conv(
            nhid * nheads, nhid * nheads, heads=4, add_self_loops=False
        )
        self.linear = nn.Linear(nhid * nheads * 4, nclass, bias=True)

    def forward(self, x, edge_index, edge_attr):
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gat2(h, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h


class GraphSAGE(torch.nn.Module):
    """GraphSAGE"""

    def __init__(self, nfeat, nhid, nclass, dropout, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.alpha = alpha

        self.gat1 = SAGEConv(
            nfeat,
            nhid,
        )
        self.gat2 = SAGEConv(
            nhid,
            nhid,
        )
        self.gat3 = SAGEConv(
            nhid,
            nhid,
        )
        self.gat4 = SAGEConv(
            nhid,
            nhid,
        )
        self.gat5 = SAGEConv(
            nhid,
            nhid,
        )
        self.lin = nn.Linear(nhid, nclass, bias=True)

    def forward(self, x, edge_index, edge_attr):
        h = self.gat1(x, edge_index)
        h = F.relu(h)
        h = self.gat2(h, edge_index)
        h = F.relu(h)
        h = self.gat3(h, edge_index)
        h = F.relu(h)
        h = self.gat4(h, edge_index)
        h = F.relu(h)
        h = self.gat5(h, edge_index)
        h = F.relu(h)
        h = self.lin(h)
        return h
