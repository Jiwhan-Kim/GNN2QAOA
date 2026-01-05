import numpy as np
from qiskit_aer.primitives import SamplerV2, EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def sim_sampler(backend, _, qcs, shots: int = 1024):
    sampler = SamplerV2.from_backend(backend)

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    candidate_circuits = pm.run(qcs)

    job = sampler.run(candidate_circuits, shots=shots)
    job_results = job.result()

    results = [job_result.data['c'].get_counts() for job_result in job_results]
    return results


def sim_estimator(backend, _, qcs, hamiltonian, shots: int | None = None):
    estimator = EstimatorV2.from_backend(backend)

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    candidate_circuits = pm.run(qcs)

    pub = [(cc, hamiltonian.apply_layout(cc.layout))
           for cc in candidate_circuits]
    job = estimator.run(
        pub, precision=0 if shots is None else 1 / np.sqrt(shots))
    job_results = job.result()

    # When running multiple circuits, change index from 0 to idx
    result = [job_result.data.evs if job_result.data.evs.size > 1
              else np.array([job_result.data.evs])
              for job_result in job_results]

    return result
