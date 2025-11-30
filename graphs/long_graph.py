import numpy as np
import rustworkx as rx


def create_long_graph(n_qubits: int):
    graph = rx.PyGraph()

    graph.add_nodes_from(np.arange(0, n_qubits, 1))
    edge_list = []
    for i in range(n_qubits - 1):
        edge_list.append((i, i + 1, 1.0))

    graph.add_edges_from(edge_list)

    return graph, graph.weighted_edge_list()
