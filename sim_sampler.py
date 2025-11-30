import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def sim_sampler(qcs, shots: int = 1024):
    sim = AerSimulator()
    sampler = SamplerV2()

    pm = generate_preset_pass_manager(optimization_level=3, backend=sim)
    candidate_circuits = pm.run(qcs)

    job = sampler.run(candidate_circuits, shots=shots)
    job_results = job.result()

    results = [job_result.data['c'].get_counts() for job_result in job_results]
    return results
