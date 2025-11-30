import rustworkx as rx


def create_random_graph(n_qubits: int):
    graph = rx.undirected_gnp_random_graph(n_qubits, 0.75, 1024)

    for edge_index in graph.edge_indices():
        graph.update_edge_by_index(edge_index, 1.0)

    return graph, graph.weighted_edge_list()
