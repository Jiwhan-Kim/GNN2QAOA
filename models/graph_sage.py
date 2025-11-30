import torch
import torch.nn as nn
import torch_geometric


def get_graph_embedding(graph):
    pass


def get_graph_embeddings(graphs: list):
    features = []

    for graph in graphs:
        get_graph_embedding(graph)

    return features
