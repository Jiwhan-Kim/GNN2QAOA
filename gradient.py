"""
Optimize QAOA Circuit using Gradient Descent
"""
import argparse
import numpy as np
from qiskit_aer import AerSimulator
import rustworkx as rx
from qiskit_ibm_runtime import Session

from circuits import QAOA
from graphs import create_long_graph
from optims import Adam
from plots import plot_costs_log, plot_gradients_log
from qpu import qpu_get_device, qpu_sampler, qpu_estimator
from simulator import sim_sampler, sim_estimator

# Default Configuration
config = {
    "environment": "simulator",
    "num_qubits": 15,
    "num_layers": 2,

    "print": True,
    "draw": True,
    "draw_name": "251228-draw",

    "lr": 0.1,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "num_iterations": 30,
    "scheduler_spec": {
        "type": "onecycle",
        "max_lr": 0.50,
        "total_steps": 30,
        "pct_start": 0.3,
        "div_factor": 25.0,
        "final_div_factor": 1e4
    }
    # "scheduler_spec": {
    #     "type": "exponent",
    #     "gamma": 0.95
    # }
}


def grad_train(graph: rx.PyGraph, initial_thetas: np.ndarray, backend, session):
    def estimator(qcs, hamiltonian, shots):
        if config['environment'] == 'qpu':
            return qpu_estimator(backend, session, qcs, hamiltonian, shots)
        else:
            return sim_estimator(backend, session, qcs, hamiltonian, shots)

    def sampler(qcs, shots):
        if config['environment'] == 'qpu':
            return qpu_sampler(backend, session, qcs, shots)
        else:
            return sim_sampler(backend, None, qcs, shots)

    num_layers = config["num_layers"]

    lr = config['lr']
    beta1 = config['beta1']
    beta2 = config['beta2']
    eps = config['eps']
    num_iterations = config["num_iterations"]
    scheduler_spec = config['scheduler_spec']

    thetas = initial_thetas
    qaoa = QAOA(graph, num_layers, False, estimator, sampler, None)
    optimizer = Adam(lr, beta1, beta2, eps, num_iterations, scheduler_spec)

    min_cost = qaoa.evaluate_cost(thetas, shots=None)
    best_thetas = thetas
    if config["print"] is True:
        print(f"Initial Cost: {min_cost}\n")

    costs_log = [min_cost]
    grads_log = []
    for iteration in range(num_iterations):
        param_shift_pairs = qaoa.create_circuit_shift_param_grad(
            thetas, np.pi / 2)
        grads = qaoa.get_circuit_param_shift_grad(
            param_shift_pairs, shots=None)

        # param_shift_pairs_eps = qaoa.create_circuit_shift_param_grad(
        #     thetas, 1e-6)
        # grads_eps = qaoa.get_circuit_param_shift_grad(
        #     param_shift_pairs_eps, shots=None)

        thetas = optimizer.step(thetas, grads)
        cost = qaoa.evaluate_cost(thetas, shots=None)
        costs_log.append(cost)
        grads_log.append(np.sqrt(np.sum(grads ** 2)))

        if cost < min_cost:
            min_cost = cost
            best_thetas = thetas.copy()

        if config["print"] is True:
            print(f"Iteration {iteration + 1}")
            print(f"Gradient: {grads}")
            print(f"Size of Gradient: {grads_log[-1]}")
            # print(f"Gradient Eps: {grads_eps / 1e-6}")
            print(f"Updated Theta: {thetas}")
            print(f"Cost: {cost}\n")

    if config["draw"] is True:
        plot_costs_log(costs_log, f"{config.get('draw_name', 'gradient')}")
        plot_gradients_log(
            grads_log, f"{config.get('draw_name', 'gradient')}")

    return best_thetas


def grad_validate(graph: rx.PyGraph, thetas: np.ndarray, backend, session):
    def estimator(qcs, hamiltonian, shots):
        if config['environment'] == 'qpu':
            return qpu_estimator(backend, session, qcs, hamiltonian, shots)
        else:
            return sim_estimator(backend, session, qcs, hamiltonian, shots)

    def sampler(qcs, shots):
        if config['environment'] == 'qpu':
            return qpu_sampler(backend, session, qcs, shots)
        else:
            return sim_sampler(backend, None, qcs, shots)

    num_layers = config["num_layers"]

    qaoa = QAOA(graph, num_layers, True, estimator, sampler, None)
    parsed_results = qaoa.evaluate_sample(thetas, 8192)

    print(parsed_results[0])


parser = argparse.ArgumentParser()


if __name__ == "__main__":
    print("Gradient")
    num_qubits = config["num_qubits"]
    num_layers = config["num_layers"]

    graph, edge_list = create_long_graph(num_qubits)
    # thetas = np.random.randn(num_layers * 2) * np.pi
    thetas = np.array([2 * np.pi + np.pi * 0.2] * num_layers * 2)
    backend = qpu_get_device(
    ) if config['environment'] == 'qpu' else AerSimulator()

    with Session(backend=backend) as session:
        thetas = grad_train(graph, thetas, backend, session)
        grad_validate(graph, thetas, backend, session)
