import torch
import numpy as np
from qiskit_ibm_runtime import Session

from circuits import build_qaoa, plot_result, draw_plot_result
from models import Graph2QAOAParams
from learns_embedding import create_graphs
from stoch_graph_partition import get_thetas_stoch
from graphs import create_long_graph
from qpu_sampler import qpu_get_device, qpu_sampler

#
# n_layers = 2
# num_graphs = 550
#
# model = Graph2QAOAParams(1, 16, 2, n_layers)
# model.load_state_dict(torch.load(f'./datas/{num_graphs}_models.pt'))
#
# test_graphs, test_dataset = create_graphs(num_graphs // 10)
#
# with torch.no_grad():
#     for graph, data in zip(test_graphs, test_dataset):
#         model.eval()
#
#         if graph.num_edges() == 0:
#             print("[Eval] Zero Edge Test Sample")
#             continue
#
#         y = model(data)
#         print(y)


result = [
    [np.float64(2.6897584670383416), np.float64(2.2568321227503327),
     np.float64(2.0724588132609494), np.float64(1.7597998849551413)],
    [np.float64(-2.953666870806232), np.float64(0.5914986833117162),
     np.float64(0.9846313125305546), np.float64(2.70394855722106)],
    [np.float64(2.650515226647528), np.float64(2.570077217575262),
     np.float64(0.49639663609753093), np.float64(1.566008575610545)]
]
graph, edge_list = create_long_graph(15)

backend = qpu_get_device()

qcs = [build_qaoa(edge_list, thetas[:2], thetas[2:], 2, 15)
       for thetas in result]

results = qpu_sampler(backend, None, qcs, 4096)

print(f"Result: {results}")

plot_result = plot_result(results, edge_list)
print(f"Plot Result: {plot_result}")
