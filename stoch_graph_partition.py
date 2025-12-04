import argparse
from qiskit_ibm_runtime import Session
from qpu_sampler import qpu_sampler, qpu_get_device
from qpu_estimator import qpu_estimator
from sim_sampler import sim_sampler
from sim_estimator import sim_estimator

from circuits import build_qaoa, build_hamiltonian, plot_result, print_result, draw_plot, draw_log_cost, draw_plot_result
from optims import Adam

from graphs.long_graph import create_long_graph


import numpy as np
import rustworkx as rx
from rustworkx.visualization import mpl_draw
import matplotlib.pyplot as plt
from datetime import datetime, timezone

DRAW = 1
time = datetime.now(timezone.utc).isoformat()


def maxcut_cost(run,
                edge_list: rx.WeightedEdgeList,
                thetas: list[float],
                consts):
    n_qubits, n_layers, n_iterations = consts

    qc = build_qaoa(
        edge_list, thetas[:n_layers], thetas[n_layers:], n_layers, n_qubits)
    result = run([qc])

    return result[0][0]


def maxcut_grad(run,
                edge_list: rx.WeightedEdgeList,
                thetas: list[float],
                consts):
    n_qubits, n_layers, _ = consts

    params = []
    params.append(thetas.copy())

    for idx in range(2 * n_layers):
        plus = thetas.copy()
        minus = thetas.copy()

        plus[idx] += np.pi / 4.0
        minus[idx] -= np.pi / 4.0

        params.append(plus)
        params.append(minus)

    results = run([build_qaoa(
        edge_list, param[:n_layers], param[n_layers:], n_layers, n_qubits)
        for param in params])
    cost = results[0][0]
    results = results[1:]

    exp_cut = [result[0] for result in results]
    # exp_cut = expectation(result, edge_list)

    plus = exp_cut[0:4*n_qubits:2]
    minus = exp_cut[1:4*n_qubits:2]

    grad = [(p - m) for p, m in zip(plus, minus)]
    return cost, grad


# lr = 0.15
# beta1 = 0.5
# beta2 = 0.9
# eps = 1e-8

lr = 0.1
beta1 = 0
beta2 = 0.999
eps = 1e-8


def minimize(run, thetas, edge_list, consts):
    _, _, n_iterations = consts
    optimizer = Adam(lr, beta1, beta2, eps, n_iterations)

    min_cost = len(edge_list) + 1
    min_thetas = thetas.copy()

    log_cost = []
    for iter in range(n_iterations):
        cost, grad = maxcut_grad(run, edge_list, thetas, consts)
        log_cost.append(cost)

        if cost < min_cost:
            min_cost = cost
            min_thetas = thetas.copy()

        thetas = optimizer.step(thetas, grad)
        thetas = [((theta + np.pi) % (2 * np.pi) - np.pi) for theta in thetas]

        if __name__ == "__main__":
            print(f"Iteration {iter + 1}")
            print(f"Prev. Iteration's Cost: {cost}")
            print(f"Gradient: {grad}")
            print(f"Updated Params: {thetas}\n")

    cost = maxcut_cost(run, edge_list, thetas, consts)
    log_cost.append(cost)

    if cost < min_cost:
        min_cost = cost
        min_thetas = thetas.copy()

    if __name__ == "__main__":
        print(f"Best Cost: {min_cost}")
        print(f"Best Params: {min_thetas}\n")

        if DRAW != 0:
            draw_log_cost(log_cost, time)

    return min_thetas, min_cost


def simulator(consts, graph, thetas):
    n_qubits, n_layers, _ = consts

    edge_list = graph.weighted_edge_list()

    hamiltonian = build_hamiltonian(n_qubits, edge_list)

    # Training
    thetas, min_cost = minimize(
        lambda qcs: sim_estimator(
            qcs, hamiltonian, 4096
        ),
        thetas,
        edge_list,
        consts
    )

    # Inference
    qc = build_qaoa(
        edge_list, thetas[:n_layers], thetas[n_layers:],
        n_layers, n_qubits)
    # print(f"Count Gate: {qc.count_ops()}")
    results = sim_sampler([qc])

    plot_results = plot_result(results, edge_list)
    if __name__ == "__main__":
        print_result(plot_results)

    if DRAW != 0:
        # draw_plot(results[0], time)
        draw_plot_result(plot_results[0][0], time)

    return thetas, min_cost, plot_results[0]


def qpu(consts, graph, thetas):
    n_qubits, n_layers, _ = consts
    backend = qpu_get_device()

    edge_list = graph.weighted_edge_list()

    hamiltonian = build_hamiltonian(n_qubits, edge_list)

    with Session(backend=backend) as session:
        # Training
        thetas, min_cost = minimize(
            lambda qcs: qpu_estimator(
                backend, session, qcs, hamiltonian, 4096
            ),
            thetas,
            edge_list,
            consts
        )

        # Inference
        qc = build_qaoa(
            edge_list, thetas[:n_layers], thetas[n_layers:],
            n_layers, n_qubits)
        # print(f"Count Gate: {qc.count_ops()}")
        results = qpu_sampler(backend, session, [qc], 4096)

    plot_results = plot_result(results, edge_list)
    # print_result(plot_results)

    if DRAW != 0:
        # draw_plot(results[0], time)
        draw_plot_result(plot_results[0][0], time)

    return thetas, min_cost, plot_results[0]


parser = argparse.ArgumentParser()
parser.add_argument('--draw', required=False, default=0)
parser.add_argument('--env', required=False,
                    default="simulator", choices=['simulator', 'qpu'])

parser.add_argument('--qubits', required=False, type=int, default=15)
parser.add_argument('--layers', required=False, type=int, default=2)
parser.add_argument('--iters', required=False, type=int, default=50)
args = parser.parse_args()


def get_thetas_stoch(consts, graph, thetas):
    n_qubits, n_layers, n_iteration, env, draw = consts

    consts = (n_qubits, n_layers, n_iteration)

    if draw != 0:
        plt.figure(1)
        mpl_draw(graph, with_labels=True, node_color='lightblue', font_size=15)
        plt.savefig(f"./images/{time}-1-graph.png")

    if env == "simulator":
        thetas, min_cost, plot_result = simulator(consts, graph, thetas)
    elif env == "qpu":
        thetas, min_cost, plot_result = qpu(consts, graph, thetas)
    else:
        print("No Env")
        return [], 100000

    if draw != 0:
        (dictionary, max, max_str, max_cut) = plot_result
        node_color = ['lightblue' if bit == '0'
                      else 'orange' for bit in max_str]

        plt.figure(5)
        mpl_draw(graph, with_labels=True,
                 node_color=node_color, font_size=15)
        plt.savefig(f"./images/{time}-5-result.png")

    return thetas, min_cost


if __name__ == "__main__":
    consts = (args.qubits, args.layers, args.iters, args.env, args.draw)

    DRAW = consts[4]

    graph, edge_list = create_long_graph(consts[0])
    thetas = [0.75 * np.pi for _ in range(consts[1])] + \
             [0.5 * np.pi for _ in range(consts[1])]

    get_thetas_stoch(consts, graph, thetas)
