from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector, ParameterVectorElement
from qiskit.quantum_info import SparsePauliOp
import rustworkx as rx
import numpy as np


def create_bell_state() -> QuantumCircuit:
    qregs = QuantumRegister(2, 'q')
    cregs = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(qregs, cregs)

    qc.h([0, 1])
    # qc.cx(0, 1)

    return qc


class QAOA:
    def __init__(self, graph: rx.PyGraph, num_layers: int, measure: bool, estimator, sampler, state_vector):
        self.edge_list = graph.weighted_edge_list()
        self.num_qubits = graph.num_nodes()
        self.num_edges = len(self.edge_list)

        self.thetas = ParameterVector("thetas", num_layers * 2)
        self.num_layers = num_layers
        self.qc = self.__build_circuit(graph, measure)
        self.hamiltonian = self.__build_hamiltonian()

        self.estimator = estimator
        self.sampler = sampler
        self.state_vector = state_vector

    def __build_circuit(self, graph: rx.PyGraph, measure: bool = False):
        num_qubits = graph.num_nodes()
        edge_list = graph.weighted_edge_list()

        qregs = QuantumRegister(num_qubits, 'q')
        cregs = ClassicalRegister(num_qubits, 'c')
        qc = QuantumCircuit(qregs, cregs)

        gammas = self.thetas[:self.num_layers]
        betas = self.thetas[self.num_layers:]

        qc.h(range(num_qubits))

        for layer in range(self.num_layers):
            for i, j, w in edge_list:
                qc.cx(i, j)
                qc.rz(2 * w * gammas[layer], j)
                qc.cx(i, j)

            for i in range(num_qubits):
                qc.rx(2 * betas[layer], i)

        if measure is True:
            qc.measure(qregs, cregs)

        return qc

    def __build_hamiltonian(self):
        pauli_list = []

        for i, j, weight in self.edge_list:
            pauli_list.append(("ZZ", [i, j], weight))

        return SparsePauliOp.from_sparse_list(pauli_list, self.num_qubits)

    def create_circuit_shift_param_grad(self, thetas: np.ndarray, shift: float = np.pi / 2):
        bc = self.qc.assign_parameters({self.thetas: thetas})

        # Keep the idx of gate parameters to be shifted
        gate_param_shift_rz_idx = []
        gate_param_shift_rx_idx = []

        idx = self.num_qubits

        for layers in range(self.num_layers):
            rz_idx_single = []
            rx_idx_single = []

            for edge in range(self.num_edges):
                rz_idx_single.append(idx + 1)
                idx += 3

            for qubit in range(self.num_qubits):
                rx_idx_single.append(idx)
                idx += 1

            gate_param_shift_rz_idx.append(rz_idx_single)
            gate_param_shift_rx_idx.append(rx_idx_single)

        # Concat
        # [[gammas_0], [gammas_1], [betas_0], [betas_1]]
        gate_param_shift_idx = gate_param_shift_rz_idx + gate_param_shift_rx_idx

        param_shift_pairs = []
        for param_shift_idx in gate_param_shift_idx:
            bc_plus_list = []
            bc_minus_list = []

            for idx in param_shift_idx:
                bc_plus = bc.copy()
                bc_plus_operation = bc_plus.data[idx].operation.copy()
                bc_plus_operation.params = [
                    theta + shift for theta in bc_plus_operation.params]
                bc_plus.data[idx] = (
                    bc_plus_operation, bc_plus.data[idx].qubits, bc_plus.data[idx].clbits)

                bc_minus = bc.copy()
                bc_minus_operation = bc_minus.data[idx].operation.copy()
                bc_minus_operation.params = [
                    theta - shift for theta in bc_minus_operation.params]
                bc_minus.data[idx] = (
                    bc_minus_operation, bc_minus.data[idx].qubits, bc_minus.data[idx].clbits)

                bc_plus_list.append(bc_plus)
                bc_minus_list.append(bc_minus)

            param_shift_pairs.append(bc_plus_list + bc_minus_list)

        return param_shift_pairs

    def get_circuit_param_shift_grad(self, param_shift_pairs: list[list[float]], shots: int | None = None):
        if self.estimator is None:
            print("No Estimator Assigned")
            return np.zeros(len(param_shift_pairs))

        bcs = []
        lengths = []

        # Concat
        for param_shift_pair in param_shift_pairs:
            bcs += param_shift_pair
            lengths.append(len(param_shift_pair))

        # RUN
        results = self.estimator(bcs, self.hamiltonian, shots)
        results = np.array(results).squeeze()

        # Reshape
        idx = 0
        results_by_param = []
        for length in lengths:
            results_by_param.append(results[idx: idx + length])
            idx += length

        grads = []
        for result_by_param in results_by_param:
            num_gates = len(result_by_param) // 2
            plus = result_by_param[:num_gates]
            minus = result_by_param[num_gates:]

            grad = 0.5 * (sum(plus) - sum(minus))
            grads.append(grad)

        return np.array(grads)

    def evaluate_cost(self, thetas: np.ndarray, shots: int | None = 4096):
        if self.estimator is None:
            print("No Estimator Assigned")
            return 0.0

        bc = self.qc.assign_parameters({self.thetas: thetas})
        results = self.estimator([bc], self.hamiltonian, shots)
        return results[0][0]

    def __get_maxcut_value(self, bitstr):
        bitstr = bitstr[::-1]
        value = 0.0

        for i, j, w in self.edge_list:
            if bitstr[i] != bitstr[j]:
                value += w

        return value

    def __parse_results(self, results, top_k: int = 5):
        parsed_results = []

        for result_dict in results:
            result = list(result_dict.items())
            result.sort(key=lambda x: x[1], reverse=True)
            K = top_k if len(result) >= top_k else len(result)
            result = result[:K]

            parsed_result = []
            for entry in result:
                parsed_result.append(
                    (entry, self.__get_maxcut_value(entry[0])))

            parsed_results.append(parsed_result)

        return parsed_results

    def evaluate_sample(self, thetas: np.ndarray, shots: int = 4096):
        bc = self.qc.assign_parameters({self.thetas: thetas})
        results = self.sampler([bc], shots)

        parsed_results = self.__parse_results(results, 5)
        return parsed_results
