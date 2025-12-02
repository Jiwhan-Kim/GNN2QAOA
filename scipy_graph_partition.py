import argparse
from qiskit_ibm_runtime import Session
from qpu_sampler import qpu_sampler, qpu_get_device
from qpu_estimator import qpu_estimator
from sim_sampler import sim_sampler
from sim_estimator import sim_estimator

from circuits import build_qaoa, build_hamiltonian, plot_result, print_result

from graphs.long_graph import create_long_graph

from scipy.optimize import minimize

import numpy as np
import rustworkx as rx
from rustworkx.visualization import mpl_draw
import matplotlib.pyplot as plt


def maxcut_cost(
        thetas: list[float],
        run,
        edge_list: rx.WeightedEdgeList,
        consts):
    n_qubits, n_layers, _ = consts

    qc = build_qaoa(
        edge_list, thetas[:n_layers], thetas[n_layers:], n_layers, n_qubits)
    result = run([qc])

    return result[0][0]


def simulator(consts, graph, thetas):
    n_qubits, n_layers, n_iterations = consts

    edge_list = graph.weighted_edge_list()

    if args.draw != 0:
        mpl_draw(graph, with_labels=True, node_color='lightblue', font_size=15)
        plt.show()

    hamiltonian = build_hamiltonian(n_qubits, edge_list)

    # Training
    thetas = minimize(maxcut_cost, thetas, args=(
        lambda qcs: sim_estimator(
            qcs, hamiltonian, 4096
        ),
        edge_list,
        consts),
        method="COBYLA", options={"maxiter": n_iterations})

    thetas = thetas.x
    thetas = [((theta + np.pi) % (2 * np.pi) - np.pi) for theta in thetas]

    # Inference
    qc = build_qaoa(
        edge_list, thetas[:n_layers], thetas[n_layers:], n_layers, n_qubits)
    # print("Count Gate:")
    # print(qc.count_ops())

    results = sim_sampler([qc], 4096)

    plot_results = plot_result(results, edge_list)
    # print_result(plot_results)

    return thetas


def qpu(consts, graph, thetas):
    n_qubits, n_layers, n_iterations = consts
    backend = qpu_get_device()

    edge_list = graph.weighted_edge_list()

    if args.draw != 0:
        mpl_draw(graph, with_labels=True, node_color='lightblue', font_size=15)
        plt.show()

    hamiltonian = build_hamiltonian(n_qubits, edge_list)

    with Session(backend=backend) as session:
        # Training
        thetas = minimize(maxcut_cost, thetas, args=(
            lambda qcs: qpu_estimator(
                backend, session, qcs, hamiltonian, 4096),
            edge_list,
            consts),
            method="COBYLA", options={"maxiter": n_iterations})

        thetas = thetas.x
        thetas = [((theta + np.pi) % (2 * np.pi) - np.pi) for theta in thetas]

        # Inference
        qc = build_qaoa(
            edge_list, thetas[:n_layers], thetas[n_layers:], n_layers, n_qubits)

        results = qpu_sampler(backend, session, [qc], 4096)

    plot_results = plot_result(results, edge_list)
    # print_result(plot_results)

    return thetas


parser = argparse.ArgumentParser()
parser.add_argument('--draw', required=False, default=0)
parser.add_argument('--env', required=False,
                    default="simulator", choices=['simulator', 'qpu'])

parser.add_argument('--qubits', required=False, type=int, default=3)
parser.add_argument('--layers', required=False, type=int, default=2)
parser.add_argument('--iters', required=False, type=int, default=10)
args = parser.parse_args()


def get_thetas(consts, graph, thetas):
    n_qubits, n_layers, n_iteration, env = consts

    consts = (n_qubits, n_layers, n_iteration)
    if env == "simulator":
        thetas = simulator(consts, graph, thetas)
    elif env == "qpu":
        thetas = qpu(consts, graph, thetas)
    else:
        return []

    return thetas


if __name__ == "__main__":
    consts = (args.qubits, args.layers, args.iters, args.env)
    graph, edge_list = create_long_graph(consts[0])
    thetas = [1.0 * np.pi for _ in range(consts[1])] + \
             [0.5 * np.pi for _ in range(consts[1])]

    result = get_thetas(consts, graph, thetas)
    print(result)
