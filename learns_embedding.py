from models import Graph2QAOAParams, rx2torch
from graphs import create_random_graph
from optims import MultiTargetMSELoss

import os
import numpy as np
import torch
from torch.optim import Adam

from scipy_graph_partition import get_thetas
from stoch_graph_partition import get_thetas_stoch


def create_graphs(num_graphs):
    graphs = []
    dataset = []
    for i in range(num_graphs):
        # if i % (num_graphs // 5):
        #     print(f"[KimJW - Create] Create Graph {i / num_graphs * 100}%")
        g, _ = create_random_graph(
            5 + (i % 11), 0.3 + 0.05 * (i % 5), seed=(2021189004 + i))
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
            continue

        for _ in range(samples_per_graph):
            new_thetas = get_thetas(consts, graph, 2 * np.pi *
                                    np.random.rand(2 * n_layers))
            sub_list.append(new_thetas)

        list.append(sub_list)

    torch.save(list, f'./datas/{num_graphs}_params_temp.pt')

    array = np.array(list)
    tensor = torch.tensor(array)
    torch.save(tensor, f'./datas/{num_graphs}_params.pt')
    return tensor


def train_model(graph, dataset, params, test_graph, test_dataset):
    model = Graph2QAOAParams(1, 16, 2, n_layers)
    model.train()
    optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = MultiTargetMSELoss()

    min_cost = 10000
    state = None
    for epoch in range(1, n_epochs + 1):
        print(f"[KimJW - Training] Epoch: {epoch}")
        for graph, data, param in zip(graph, dataset, params):
            if graph.num_edges() == 0:
                if epoch == 1:
                    print("[KimJW - Training] Note: Zero Edge Training Sample")
                continue

            # params: shape (3, 2 * n_layers)
            y = model(data)  # shape (2 * n_layers, )
            loss = criterion(y, param)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cost = eval_model(model, test_graph, test_dataset)
        if cost < min_cost:
            state = model.state_dict()

    if state is not None:
        torch.save(state, f"./datas/{num_graphs}_models.pt")
    print("[KimJW - Training] Training Done")
    if state is not None:
        torch.save(state, f"./datas/{num_graphs}_models.pt")
    else:
        print("[KimJW - Training] No State are stored")

    return model


def eval_model(model, test_graph, test_dataset):
    # min_thetas_diff_L2_no_model = []
    min_costs_no_model = []
    for graph in test_graph:
        if graph.num_edges() == 0:
            print("[KimJW - Eval] Note: Zero Edge Test Sample")
            continue

        consts = (graph.num_nodes(), n_layers, n_iteration, 'simulator', 0)
        thetas = 2 * np.pi * (np.random.rand(2 * n_layers) - 0.5)

        min_thetas, min_cost = get_thetas_stoch(consts, graph, thetas)
        # min_thetas = torch.tensor(min_thetas)
        # errs = (thetas - min_thetas) ** 2
        # min_thetas_diff_L2_no_model.append(errs.norm())
        min_costs_no_model.append(min_cost)

    min_costs_no_model_array = np.array(min_costs_no_model)
    print(
        f"[KimJW - Evaluation] Average Min Cost on Testset without model: {min_costs_no_model_array.mean()}")
    print(
        f"[KimJW - Evaluation] std. of Min Cost on Testset without model: {min_costs_no_model_array.std()}\n")

    with torch.no_grad():
        model.eval()

        # min_thetas_diff_L2 = []
        min_costs = []
        for graph, data in zip(test_graph, test_dataset):
            consts = (graph.num_nodes(), n_layers, n_iteration, 'simulator', 0)

            y = model(data).tolist()  # shape (2 * n_layers, )
            min_thetas, min_cost = get_thetas_stoch(consts, graph, y)

            # min_thetas = torch.tensor(min_thetas)
            # errs = (y - min_thetas) ** 2
            # min_thetas_diff_L2.append(errs.norm())
            print(y, min_thetas)
            min_costs.append(min_cost)

        # min_thetas_diff_L2_array = np.array(min_thetas_diff_L2)
        min_costs_array = np.array(min_costs)
        print(
            f"[KimJW - Evaluation] Average Min Cost on Testset with model: {min_costs_array.mean()}")
        print(
            f"[KimJW - Evaluation] std. of Min Cost on Testset with model: {min_costs_array.std()}\n")
        # print(
        #     f"[KimJW - Evaluation] Param Shift Average: {
        #         min_thetas_diff_L2_array.mean()}"
        # )
        return min_costs_array.mean()


if __name__ == '__main__':
    num_graphs = 550  # Greater than 10
    n_layers = 2
    n_iteration = 30

    n_epochs = 50

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
        model = Graph2QAOAParams(1, 16, 2, n_layers)
        model.load_state_dict(torch.load(f'./datas/{num_graphs}_models.pt'))

    print("Start Inference")
    eval_model(model, test_graphs, test_dataset)
