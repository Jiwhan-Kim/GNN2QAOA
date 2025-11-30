from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def make_two_qubit_circuit():
    qregs = QuantumRegister(2)
    cregs = ClassicalRegister(2, "c")

    qc = QuantumCircuit(qregs, cregs)
    qc.x(1)
    qc.measure([0, 1], [0, 1])

    return qc
