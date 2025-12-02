import math
import torch
import torch.nn as nn
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import Data

import rustworkx as rx


class Graph2QAOAParams(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, n_layers):
        """
        in_channels: Number of input features per node
        hidden_channels: Number of hidden units in GraphSAGE
        num_layers: Number of GraphSAGE layers
        n_layers: Number of QAOA layers (determines output size)
        """
        super().__init__()
        self.encoder = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
        )

        self.linear = nn.Sequential(
            nn.Linear(hidden_channels, 2 * hidden_channels),
            nn.ReLU(),
            nn.Linear(2 * hidden_channels, 2 * hidden_channels),
            nn.ReLU(),
            nn.Linear(2 * hidden_channels, 2 * n_layers)
        )

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        # Graph Neural Network
        node_embeddings: torch.Tensor = self.encoder(
            x, edge_index)     # (N, hidden_dim)
        graph_embeddings = node_embeddings.mean(
            dim=0, keepdim=True)    # (1, hidden_dim)

        # Linear Regression
        thetas = self.linear(graph_embeddings).squeeze(
            0)               # (2 * n_layers)
        thetas = (thetas + torch.pi) % (2 * torch.pi) - torch.pi
        return thetas
        # return thetas.tolist()


def rx2torch(graph: rx.PyGraph) -> Data:
    num_nodes = graph.num_nodes()

    # Node Features: log(1 + degree)
    degrees = []
    for v in range(num_nodes):
        deg_v = graph.degree(v)
        degrees.append([math.log1p(float(deg_v))])

    x = torch.tensor(degrees, dtype=torch.float32)

    # Edge Features
    edges = list(graph.edge_list())
    if len(edges) > 0:
        src = [u for (u, v) in edges] + [v for (u, v) in edges]
        dst = [v for (u, v) in edges] + [u for (u, v) in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)
