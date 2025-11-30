import argparse
from qiskit_ibm_runtime import Session
from qpu_sampler import qpu_sampler, qpu_get_device
from qpu_estimator import qpu_estimator
from sim_sampler import sim_sampler
from sim_estimator import sim_estimator

from circuits import build_qaoa, build_hamiltonian, plot_result, print_result
from optims import Adam

from graphs.long_graph import create_long_graph


import numpy as np
import rustworkx as rx
from rustworkx.visualization import mpl_draw
import matplotlib.pyplot as plt


def maxcut_cost(run,
                edge_list: rx.WeightedEdgeList,
                thetas: list[float]):
    qc = build_qaoa(
        edge_list, thetas[:args.layers], thetas[args.layers:], args.layers, args.qubits)
    result = run([qc])

    return result[0][0]


def maxcut_grad(run,
                edge_list: rx.WeightedEdgeList,
                thetas: list[float]):
    params = []
    params.append(thetas.copy())

    for idx in range(2 * args.layers):
        plus = thetas.copy()
        minus = thetas.copy()

        plus[idx] += np.pi / 4.0
        minus[idx] -= np.pi / 4.0

        params.append(plus)
        params.append(minus)

    results = run([build_qaoa(
        edge_list, param[:args.layers], param[args.layers:], args.layers, args.qubits)
        for param in params])
    cost = results[0]
    results = results[1:]

    exp_cut = [result[0] for result in results]
    # exp_cut = expectation(result, edge_list)

    plus = exp_cut[0:4*args.qubits:2]
    minus = exp_cut[1:4*args.qubits:2]

    grad = [(p - m) for p, m in zip(plus, minus)]
    return cost, grad


lr = 0.05
beta1 = 0.5
beta2 = 0.9
eps = 1e-8


def minimize(run, thetas, edge_list, max_iterations: int = 30):
    optimizer = Adam(lr, beta1, beta2, eps, max_iterations)

    min_cost = len(edge_list) + 1
    min_thetas = thetas.copy()

    for iter in range(max_iterations):
        cost, grad = maxcut_grad(run, edge_list, thetas)
        if cost < min_cost:
            min_cost = cost
            min_thetas = thetas.copy()

        thetas = optimizer.step(thetas, grad)
        thetas = [((theta + np.pi) % (2 * np.pi) - np.pi) for theta in thetas]

        print(f"Iteration {iter + 1}")
        print(f"Prev. Iteration's Cost: {cost}")
        print(f"Gradient: {grad}")
        print(f"Updated Params: {thetas}\n")

    cost = maxcut_cost(run, edge_list, thetas)
    if cost < min_cost:
        min_cost = cost
        min_thetas = thetas.copy()

    print(f"Best Cost: {min_cost}")
    print(f"Best Params: {min_thetas}\n")
    return min_thetas


def simulator(edge_list):
    # thetas = 2 * np.pi * (np.random.rand(2 * args.layers) - 0.5)
    thetas = [np.pi for _ in range(args.layers)] + \
        [0.5 * np.pi for _ in range(args.layers)]
    hamiltonian = build_hamiltonian(args.qubits, edge_list)

    # Training
    thetas = minimize(
        lambda qcs: sim_estimator(
            qcs, hamiltonian, 4096
        ),
        thetas,
        edge_list,
        args.iters
    )

    # Inference
    qc = build_qaoa(
        edge_list, thetas[:args.layers], thetas[args.layers:], args.layers, args.qubits)
    print(f"Count Gate: {qc.count_ops()}")
    results = sim_sampler([qc])

    plot_results = plot_result(results, edge_list)
    print_result(plot_results)


def qpu(edge_list):
    backend = qpu_get_device()

    # thetas = 2 * np.pi * np.random.rand(2 * args.layers)
    thetas = [np.pi for _ in range(args.layers)] + \
        [0.5 * np.pi for _ in range(args.layers)]
    hamiltonian = build_hamiltonian(args.qubits, edge_list)

    with Session(backend=backend) as session:
        # Training
        thetas = minimize(
            lambda qcs: qpu_estimator(
                backend, session, qcs, hamiltonian, 4096
            ),
            thetas,
            edge_list,
            args.iters
        )

        # Inference
        qc = build_qaoa(
            edge_list, thetas[:args.layers], thetas[args.layers:], args.layers, args.qubits)
        print(f"Count Gate: {qc.count_ops()}")
        results = qpu_sampler(backend, session, [qc])

    plot_results = plot_result(results, edge_list)
    print_result(plot_results)


parser = argparse.ArgumentParser()
parser.add_argument('--draw', required=False, default=0)
parser.add_argument('--env', required=False,
                    default="simulator", choices=['simulator', 'qpu'])

parser.add_argument('--qubits', required=False, type=int, default=15)
parser.add_argument('--layers', required=False, type=int, default=2)
parser.add_argument('--iters', required=False, type=int, default=30)
args = parser.parse_args()


def main():
    graph, edge_list = create_long_graph(args.qubits)
    if args.draw != 0:
        mpl_draw(graph, with_labels=True, node_color='lightblue', font_size=15)
        plt.show()

    if args.env == "simulator":
        simulator(edge_list)
    elif args.env == "qpu":
        qpu(edge_list)
    else:
        return


if __name__ == "__main__":
    main()
