from qiskit_aer import AerSimulator
from circuits import create_bell_state

from sim_estimator import sim_estimator
from qpu_estimator import qpu_estimator
from qpu_sampler import qpu_get_device
from qiskit.quantum_info import SparsePauliOp


def main():
    est = AerSimulator()
    # qpu = qpu_get_device()

    qc = create_bell_state()
    hamiltonian = SparsePauliOp('ZZ')

    est_result = sim_estimator(est, None, [qc], hamiltonian, 4096)
    # qpu_result = qpu_estimator(qpu, qpu, [qc], hamiltonian, 4096)

    print(est_result)
    # print(qpu_result)


if __name__ == "__main__":
    main()
