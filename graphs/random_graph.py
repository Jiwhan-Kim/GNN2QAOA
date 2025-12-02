import rustworkx as rx
from rustworkx.visualization import mpl_draw
import matplotlib.pyplot as plt


def create_random_graph(n_qubits: int, edge_prob, seed):
    graph = rx.undirected_gnp_random_graph(n_qubits, edge_prob, seed)

    for edge_index in graph.edge_indices():
        graph.update_edge_by_index(edge_index, 1.0)

    return graph, graph.weighted_edge_list()


if __name__ == "__main__":
    graph, _ = create_random_graph(6, 0.3, 10)
    mpl_draw(graph, with_labels=True, node_color='lightblue', font_size=15)
    plt.show()
