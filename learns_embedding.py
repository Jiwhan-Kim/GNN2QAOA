from models import Graph2QAOAParams, rx2torch
from graphs import create_random_graph, create_long_graph
from optims import MultiTargetMSELoss
from circuits import build_qaoa, plot_result, draw_plot_result
from sim_sampler import sim_sampler

import os
import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
from rustworkx.visualization import mpl_draw

from scipy_graph_partition import get_thetas
from stoch_graph_partition import get_thetas_stoch


def create_graphs(num_graphs):
    graphs = []
    dataset = []
    for i in range(num_graphs):
        # if i % (num_graphs // 5):
        #     print(f"[KimJW - Create] Create Graph {i / num_graphs * 100}%")
        g, _ = create_random_graph(
            5 + (i % 11), 0.3 + 0.05 * (i % 5), seed=2021189004 + i)
        graphs.append(g)
        dataset.append(rx2torch(g))

    torch.save(graphs, f'./datas/{num_graphs}_graphs.pt')
    torch.save(dataset, f'./datas/{num_graphs}_datasets.pt')

    return graphs, dataset


def get_graphs():
    graphs = torch.load(
        f'./datas/{num_graphs}_graphs.pt', weights_only=False)
    dataset = torch.load(
        f'./datas/{num_graphs}_datasets.pt', weights_only=False)

    return graphs, dataset


def get_params(graphs, _):
    samples_per_graph = 3

    list = []
    for i, graph in enumerate(graphs):
        if (i + 1) % 10 == 0:
            print(
                f"[KimJW - Create] Create Params (Cobyla) {(i + 1) / num_graphs * 100}%")

        consts = (graph.num_nodes(), n_layers, n_iteration, 'simulator')
        sub_list = []
        if graph.num_edges() == 0:
            print(
                f"[KimJW - Create] Note: Zero Edge Training Sample: graph {i}")
            sub_list = [
                [np.float64(0.0) for _ in range(2 * n_layers)]
                for _ in range(samples_per_graph)
            ]
            list.append(sub_list)
            continue

        for _ in range(samples_per_graph):
            new_thetas = get_thetas(consts, graph, 2 * np.pi *
                                    np.random.rand(2 * n_layers))

            # new_thetas = [((theta + np.pi) % (2 * np.pi) - np.pi) for theta in new_thetas]
            sub_list.append(new_thetas)

        list.append(sub_list)

    array = np.array(list)
    tensor = torch.tensor(array)
    torch.save(tensor, f'./datas/{num_graphs}_params.pt')
    return tensor


def train_model(graphs, dataset, params, test_graph, test_dataset):
    model = Graph2QAOAParams(1, 16, 3, n_layers)
    model.train()
    optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = MultiTargetMSELoss()

    for epoch in range(1, n_epochs + 1):
        if epoch % 10 == 0:
            print(f"[KimJW - Training] Epoch: {epoch}")

        for idx, (graph, data, param) in enumerate(zip(graphs, dataset, params)):
            if idx % 100 == 0:
                print(f"[KimJW - Training] {idx} / {num_graphs}")

            if graph.num_edges() == 0:
                if epoch == 1:
                    print("[KimJW - Training] Note: Zero Edge Training Sample")
                continue

            param = ((param + np.pi) % (2 * torch.pi) - torch.pi)

            # params: shape (3, 2 * n_layers)
            y = model(data)  # shape (2 * n_layers, )
            loss = criterion(y, param)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("[KimJW - Training] Training Done")
    torch.save(model.state_dict(), f"./datas/{num_graphs}_models.pt")

    return model


def eval_model(model, test_graphs, test_dataset):
    min_costs_no_model = []
    for idx, graph in enumerate(test_graphs):
        if idx % 10 == 0:
            print(f"[KimJW - Eval] No Model {idx} / {num_graphs // 10}")
        if graph.num_edges() == 0:
            print("[KimJW - Eval] Note: Zero Edge Test Sample")
            continue

        # consts = (graph.num_nodes(), n_layers, n_iteration, 'simulator', 0)
        consts = (graph.num_nodes(), n_layers, 10, 'simulator', 0)
        thetas = 2 * np.pi * (np.random.rand(2 * n_layers) - 0.5)

        min_thetas, min_cost = get_thetas_stoch(consts, graph, thetas)
        min_costs_no_model.append(min_cost)

    min_costs_no_model_array = np.array(min_costs_no_model)
    print(
        f"[KimJW - Evaluation] Average Min Cost on Testset without model: {min_costs_no_model_array.mean()}")
    print(
        f"[KimJW - Evaluation] std. of Min Cost on Testset without model: {min_costs_no_model_array.std()}\n")

    with torch.no_grad():
        model.eval()

        min_costs = []
        for idx, (graph, data) in enumerate(zip(test_graphs, test_dataset)):
            if idx % 10 == 0:
                print(f"[KimJW - Eval] No Model {idx} / {num_graphs // 10}")

            if graph.num_edges() == 0:
                print("[KimJW - Eval] Note: Zero Edge Test Sample")
                continue

            # consts = (graph.num_nodes(), n_layers, n_iteration, 'simulator', 0)
            consts = (graph.num_nodes(), n_layers, 10, 'simulator', 0)
            y = model(data).tolist()  # shape (2 * n_layers, )

            min_thetas, min_cost = get_thetas_stoch(consts, graph, y)
            min_costs.append(min_cost)

        # min_thetas_diff_L2_array = np.array(min_thetas_diff_L2)
        min_costs_array = np.array(min_costs)
        print(
            f"[KimJW - Evaluation] Average Min Cost on Testset with model: {min_costs_array.mean()}")
        print(
            f"[KimJW - Evaluation] std. of Min Cost on Testset with model: {min_costs_array.std()}\n")

        print(
            f"Result Vector: \n{min_costs_no_model}\n{min_costs}"
        )
        return min_costs_array.mean()


def test_long_graph(model):
    graph, edge_list = create_long_graph(15)
    data = rx2torch(graph)
    thetas_model = model(data).tolist()
    # thetas_no_model = 2 * np.pi * (np.random.rand(2 * n_layers) - 0.5)

    consts = (graph.num_nodes(), n_layers, 10, 'simulator', 1)

    print("Long Graph Test")
    _, min_cost_model = get_thetas_stoch(consts, graph, thetas_model)
    # _, min_cost_no_model = get_thetas_stoch(consts, graph, thetas_no_model)

    print(f"Compare: model: {min_cost_model}")
    # print(f"Compare: no_model: {min_cost_no_model}")


if __name__ == '__main__':
    num_graphs = 550  # Greater than 10
    n_layers = 2
    n_iteration = 30

    n_epochs = 100

    print("Start to run")

    if not os.path.exists(f"./datas/{num_graphs}_graphs.pt") \
            or not os.path.exists(f"./datas/{num_graphs}_datasets.pt"):
        print("[KimJW - Main] Create Datasets")
        graphs, dataset = create_graphs(num_graphs)
    else:
        print("[KimJW - Main] Load datasets from Files")
        graphs, dataset = get_graphs()
        # print("[KimJW - Main] Test for graph num-edges")
        # for graph in graphs:
        #     if graph.num_edges() == 0:
        #         print("[Error] There is a graph with 0 edges.")
        # exit()

    if not os.path.exists(f"./datas/{num_graphs}_params.pt"):
        print("[KimJw - Main] Create Targets")
        params = get_params(graphs, dataset)
    else:
        print("Load params(targets) from Files")
        params = torch.load(
            f"./datas/{num_graphs}_params.pt", weights_only=False)

    print("[KimJW - Main] Create Testsets")
    test_graphs, test_dataset = create_graphs(num_graphs // 10)

    print(f"Preparing Data Done: {params.shape}")

    print("[KimJW - Main] Training")
    if not os.path.exists(f"./datas/{num_graphs}_models.pt"):
        print("[KimJW - Main] Training Starts")
        model = train_model(graphs, dataset, params, test_graphs, test_dataset)
    else:
        print("[KimJW - Main] Load model from Files\n")
        model = Graph2QAOAParams(1, 16, 3, n_layers)
        model.load_state_dict(torch.load(f'./datas/{num_graphs}_models.pt'))

    print("Start Inference")
    # eval_model(model, test_graphs, test_dataset)
    test_long_graph(model)
