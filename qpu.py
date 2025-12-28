import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2, EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


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

    return result


def qpu_estimator(backend, session, qcs, hamiltonian, shots: int = 1024):
    if backend is None:
        raise ValueError("Backend is None. Please provide a valid backend.")

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    candidate_circuits = pm.run(qcs)

    estimator = EstimatorV2(mode=session if session is not None else backend)
    estimator.options.default_shots = shots

    # Set simple error suppression/mitigation options
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XY4"
    estimator.options.twirling.enable_gates = True
    estimator.options.twirling.num_randomizations = "auto"

    # When running multiple circuits, just add in the list

    pub = [(cc, hamiltonian.apply_layout(cc.layout))
           for cc in candidate_circuits]
    job = estimator.run(pub)
    job_results = job.result()

    # When running multiple circuits, change index from 0 to idx
    result = [job_result.data.evs if job_result.data.evs.size > 1
              else np.array([job_result.data.evs])
              for job_result in job_results]
    # print("IBM Backend Result: ", result)

    return result
