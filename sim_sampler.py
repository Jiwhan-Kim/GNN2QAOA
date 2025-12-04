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


if __name__ == "__main__":
    from circuits import build_qaoa, plot_result, draw_plot_result
    from graphs import create_long_graph

    graph, edge_list = create_long_graph(15)
    qc = build_qaoa(
        edge_list, [2.534, -2.839], [1.220, 0.677], 2, 15
    )
    results = sim_sampler([qc])
    plot_results = plot_result(results, edge_list)
    draw_plot_result(plot_results[0][0], 'sims1')
