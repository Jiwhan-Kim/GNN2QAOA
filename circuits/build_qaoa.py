import matplotlib.pyplot as plt
import rustworkx as rx
import numpy as np
from itertools import product

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp


def build_hamiltonian(n_qubits: int, edge_list: rx.WeightedEdgeList):
    pauli_list = []

    for i, j, weight in edge_list:
        pauli_list.append(("ZZ", [i, j], weight))

    return SparsePauliOp.from_sparse_list(pauli_list, n_qubits)


def qaoa_layer(qc, gamma: float, beta: float, n_qubits: int, edge_list: rx.WeightedEdgeList):
    for i, j, _ in edge_list:
        qc.cx(i, j)
        qc.rz(2 * gamma, j)
        qc.cx(i, j)

    for i in range(n_qubits):
        qc.rx(2 * beta, i)


def build_qaoa(edge_list: rx.WeightedEdgeList, gammas, betas, layers: int, n_qubits: int):
    qregs = QuantumRegister(n_qubits)
    cregs = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(qregs, cregs)

    for i in range(n_qubits):
        qc.h(i)

    for layer in range(layers):
        qaoa_layer(qc, gammas[layer], betas[layer], n_qubits, edge_list)

    qc.measure([i for i in range(n_qubits)], [i for i in range(n_qubits)])

    return qc


def maxcut_value(bitstring, edge_list: rx.WeightedEdgeList):
    # bitstring: "010101"
    # bits     : "101010"

    bits = bitstring[::-1]
    value = 0
    for i, j, _ in edge_list:
        if bits[i] != bits[j]:
            value += 1
    return value


# Get Expectation <H> from the result of Sampler
# note: Estimator directly returns the expectation
def expectation(results, edge_list: rx.WeightedEdgeList):
    list_ret = []

    for result in results:
        total = 0
        shots = 0

        for bitstr, count in result.items():
            total += maxcut_value(bitstr, edge_list) * count
            shots += count
        list_ret.append(total / shots)

    return list_ret


def plot_result(results, edge_list: rx.WeightedEdgeList):
    list_ret = []

    for result in results:
        dictionary = {k: 0 for k in range(len(edge_list) + 1)}
        max = -1
        max_str = ""
        max_cut = -1

        for bitstr, count in result.items():
            cut = maxcut_value(bitstr, edge_list)
            dictionary[cut] += count

            if count > max or (count == max and cut > max_cut):
                max = count
                max_str = bitstr[::-1]
                max_cut = cut

        list_ret.append((dictionary, max, max_str, max_cut))

    return list_ret


def draw_plot_result(dictionary, time):
    x = list(dictionary.keys())
    y = list(dictionary.values())

    plt.figure(4)
    plt.bar(x, y)
    plt.xlabel('Cuts')
    plt.ylabel('Value')
    plt.savefig(f"./images/{time}-4-distribute.png")


def print_result(plot_results):
    (dictionary, max, max_str, max_cut) = plot_results[0]

    print(f"Result of Sampler: {dictionary}")
    print(f"Result of Partitioning: {max_str}, Cuts: {
          max_cut}, Counts: {max} / 32768")


def draw_plot(counts: dict, time: str):
    result = fill_missing_bitstrings(counts)
    sorted_items = sorted(result.items(), key=lambda kv: kv[0])
    x = [k for k, _ in sorted_items]
    y = [v for _, v in sorted_items]

    plt.figure(3)
    plt.bar(x, y)
    plt.xlabel('Key')
    plt.ylabel('Value')
    plt.savefig(f"./images/{time}-3-items.png")


def fill_missing_bitstrings(counts: dict) -> dict:
    """
    Given a counts dictionary like {"00": 11, "11": 5},
    return a full dictionary including 0-count bitstrings:
    {"00": 11, "01": 0, "10": 0, "11": 5}.
    """
    # infer number of qubits from bitstring length
    if len(counts) == 0:
        raise ValueError("counts dictionary is empty")

    n = len(next(iter(counts)))  # number of qubits from key length

    all_strings = [''.join(bits) for bits in product('01', repeat=n)]
    full = {s: 0 for s in all_strings}

    for bitstr, c in counts.items():
        full[bitstr] = c

    return full


def draw_log_cost(costs, time: str):
    costs = np.array(costs).flatten()
    length = len(costs)
    x = np.arange(length)

    plt.figure(2)
    plt.plot(x, costs, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title("Minimizing Cost")
    plt.grid(True)
    plt.savefig(f"./images/{time}-2-cost-log.png")
