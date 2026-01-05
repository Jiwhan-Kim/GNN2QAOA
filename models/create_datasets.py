import rustworkx as rx
import numpy as np
import torch
from torch_geometric.data import Data

import models.get_qaoa_optimized

from models import rx2torch, grad_train, get_qaoa_optimized
from utils import get_2pi

config = {
    "samples_per_graph:": 3,

    "environment": "simulator",
    "hw_noise": False,
    "shots": None,
    "num_layers": 3,

    "print": True,
    "draw": False
}


def convert_graphs(graphs: list[rx.PyGraph], name: str):
    tensors = [rx2torch(graph) for graph in graphs]
    torch.save(tensors, f"./datas/{name}_tensors.pt")

    return tensors


def create_datasets(graphs: list[rx.PyGraph], name: str, backend, session):
    samples_per_graph = config.get("samples_per_graph", 3)
    num_layers = config.get("num_layers", 3)

    datasets = []
    for idx, graph in enumerate(graphs):
        if idx == 0 or (idx + 1) % (len(graphs) // 10) == 0:
            print(f"[Create Datasets] Create Datasets {idx} / {len(graphs)}")

        sub_dataset = []
        if graph.num_edges() == 0:
            print(f"[Create Datasets] Zero Edge Training Sample: Index {idx}")
            sub_dataset = [np.zeros(num_layers)
                           for _ in range(samples_per_graph)]
            datasets.append(sub_dataset)
            continue

        models.get_qaoa_optimized.config["environment"] = config["environment"]
        models.get_qaoa_optimized.config["hw_noise"] = config["hw_noise"]
        models.get_qaoa_optimized.config["shots"] = config["shots"]
        models.get_qaoa_optimized.config["num_qubits"] = graph.num_nodes()
        models.get_qaoa_optimized.config["num_layers"] = num_layers

        models.get_qaoa_optimized.config["print"] = config["print"]
        models.get_qaoa_optimized.config["draw"] = config["draw"]

        for _ in range(samples_per_graph):
            thetas = grad_train(graph, np.random.randn(
                2 * num_layers), backend, session)
            thetas = get_2pi(thetas)
            sub_dataset.append(thetas)

        datasets.append(sub_dataset)

    torch.save(datasets, f"./datas/{name}_datasets.pt")
    return datasets
