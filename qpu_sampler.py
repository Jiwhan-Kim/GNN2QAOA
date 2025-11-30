from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import transpile

from circuits import make_bell_circuit


def qpu_get_device():
    service = QiskitRuntimeService()

    print("Backend:")
    for backend in service.backends():
        print("-", backend.name, "| simulator:",
              backend.configuration().simulator)

    backend = service.least_busy(
        operational=True, simulator=False, min_num_qubits=5)
    print("Using backend:", backend.name)
    return backend


def qpu_sampler(backend, session, qcs, shots: int = 1024):
    if backend is None:
        raise ValueError("Backend is None. Please provide a valid backend.")

    if __name__ == "__main__":
        for qc in qcs:
            print(qc[0])

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    candidate_circuits = pm.run(qcs)

    # tqcs = [transpile(qc, backend=backend) for qc in qcs]

    sampler = SamplerV2(mode=session if session is not None else backend)
    sampler.options.default_shots = shots

    # Set simple error suppression/mitigation options
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.num_randomizations = "auto"

    # When running multiple circuits, just add in the list
    job = sampler.run(candidate_circuits, shots=shots)
    job_results = job.result()

    # When running multiple circuits, change index from 0 to idx
    result = [job_result.data['c'].get_counts() for job_result in job_results]
    # print("IBM Backend Result: ", result)

    return result


if __name__ == "__main__":
    backend = qpu_get_device()
    qpu_sampler(backend, None, [make_bell_circuit(), make_bell_circuit()])
