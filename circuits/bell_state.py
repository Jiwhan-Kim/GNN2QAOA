from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def make_bell_circuit():
    # Creates a Bell State |Psi> = (|00> + |11>)/sqrt(2)
    # Fix classical registers
    qregs = QuantumRegister(2)
    cregs = ClassicalRegister(2, "c")

    qc = QuantumCircuit(qregs, cregs)
    qc.x(1)
    qc.h(0)          # |0> -> (|0> + |1>)/sqrt(2)
    qc.cx(0, 1)      # entangle -> (|00> + |11>)/sqrt(2)
    qc.measure([0, 1], [0, 1])

    return qc
