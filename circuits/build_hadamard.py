from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def make_hadamard(n_qubits: int = 3):
    qregs = QuantumRegister(n_qubits)
    cregs = ClassicalRegister(n_qubits, "c")

    qc = QuantumCircuit(qregs, cregs)
    for i in range(n_qubits):
        qc.h(i)

    qc.measure([i for i in range(n_qubits)], [i for i in range(n_qubits)])

    return qc


def make_hadamard_no_measure(n_qubits: int = 3):
    qregs = QuantumRegister(n_qubits)

    qc = QuantumCircuit(qregs)
    for i in range(n_qubits):
        qc.h(i)

    return qc
