from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from failure_gnn.data import GraphData


class GraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, adjacency: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        propagated = torch.sparse.mm(adjacency, node_features)
        return self.linear(propagated)


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = GraphConvolution(in_dim, hidden_dim)
        self.conv2 = GraphConvolution(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: GraphData) -> torch.Tensor:
        hidden = torch.relu(self.conv1(graph.adjacency, graph.node_features))
        hidden = self.dropout(hidden)
        hidden = torch.relu(self.conv2(graph.adjacency, hidden))
        return hidden.mean(dim=0)


class BuildClassifier(nn.Module):
    def __init__(self, graph_dim: int, build_feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        combined_dim = graph_dim + build_feature_dim
        self.layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_embeddings: torch.Tensor, build_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([graph_embeddings, build_features], dim=1)
        return self.layers(combined).squeeze(1)


@dataclass
class ModelBundle:
    encoder: GraphEncoder
    classifier: BuildClassifier

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.classifier.parameters()
