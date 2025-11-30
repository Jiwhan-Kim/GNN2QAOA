import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def sim_estimator(qcs, hamiltonian, shots: int = 1024):
    sim = AerSimulator()
    estimator = EstimatorV2()

    pm = generate_preset_pass_manager(optimization_level=3, backend=sim)
    candidate_circuits = pm.run(qcs)

    pub = [(cc, hamiltonian.apply_layout(cc.layout))
           for cc in candidate_circuits]
    job = estimator.run(pub)
    job_results = job.result()

    # When running multiple circuits, change index from 0 to idx
    result = [job_result.data.evs if job_result.data.evs.size > 1
              else np.array([job_result.data.evs])
              for job_result in job_results]

    return result
