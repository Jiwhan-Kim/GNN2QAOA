import rustworkx as rx

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


def print_result(plot_results):
    (dictionary, max, max_str, max_cut) = plot_results[0]

    print(f"Result of Sampler: {dictionary}")
    print(f"Result of Partitioning: {max_str}, Cuts: {
          max_cut}, Counts: {max} / 32768")
