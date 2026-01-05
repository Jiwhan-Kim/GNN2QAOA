from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Session
import torch
import numpy as np

from circuits import QAOA
from cores import qpu_get_device, sim_estimator
from cores.qpu import qpu_estimator
from graphs.long_graph import create_long_graph


def main(backend, session):
    def estimator(qcs, hamiltonian, shots):
        # return sim_estimator(backend, session, qcs, hamiltonian, shots)
        return qpu_estimator(backend, session, qcs, hamiltonian, shots)

    graph, _ = create_long_graph(6)
    thetas_log = torch.load("./datas/thetas_log.pt", weights_only=False)
    gradient_log = torch.load("./datas/gradient_log.pt", weights_only=False)

    qaoa = QAOA(graph, 2, False, estimator, None, None)

    thetas_log = thetas_log[0:30]
    thetas_log = thetas_log[::-1]
    gradient_log = gradient_log[::-1]

    for thetas, grads_real in zip(thetas_log, gradient_log):
        param_shift_pairs = qaoa.create_circuit_shift_param_grad(
            thetas, np.pi / 2)

        grads_noisy = qaoa.get_circuit_param_shift_grad(
            param_shift_pairs, shots=4096)
        print(f"Gradient: Real: {grads_real}, Noisy: {grads_noisy}")

        # Norms
        grads_real_norm = np.linalg.norm(grads_real)
        grads_noisy_norm = np.linalg.norm(grads_noisy)

        # Cosine
        dot = np.dot(grads_real, grads_noisy)
        denom = (np.linalg.norm(grads_real) *
                 np.linalg.norm(grads_noisy)) + 1e-8
        cosine = dot / denom

        # Distance
        diff = grads_real - grads_noisy
        l2 = np.linalg.norm(diff)

        print(f"Norms: Real: {grads_real_norm}, Noisy: {grads_noisy_norm}")
        print(f"L2 Dist: {l2}")
        print(f"Cosine: {cosine}\n")


if __name__ == "__main__":
    # backend = AerSimulator.from_backend(qpu_get_device())
    # backend = AerSimulator()
    backend = qpu_get_device()
    with Session(backend=backend) as session:
        main(backend, session)
