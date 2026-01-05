import argparse
import numpy as np
from qiskit_ibm_runtime import Session
from qiskit_aer import AerSimulator

import models.get_qaoa_optimized

from models import grad_train, grad_validate
from graphs.long_graph import create_long_graph
from cores import qpu_get_device

# Change Configurations for testing
models.get_qaoa_optimized.config = {
    "environment": "simulator",
    "hw_noise": False,
    "shots": None,
    "num_qubits": 6,
    "num_layers": 2,

    "print": True,
    "draw": False,
    "draw_name": "251229-sim-layer3",

    "lr": 0.1,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "num_iterations": 30,
    "scheduler_spec": {
        "type": "onecycle",
        "max_lr": 0.25,
        "total_steps": 30,
        "pct_start": 0.3,
        "div_factor": 25.0,
        "final_div_factor": 1e4
    }
    # "scheduler_spec": {
    #     "type": "exponent",
    #     "gamma": 0.95
    # }
}

parser = argparse.ArgumentParser()

parser.add_argument(
    "--environment", choices=["simulator", "qpu"], help="Execution environment")
parser.add_argument(
    "--hw-noise", dest="hw_noise",
    action="store_true", help="Enable hardware noise")
parser.add_argument(
    "--no-hw-noise", dest="hw_noise",
    action="store_false", help="Disable hardware noise")
parser.set_defaults(hw_noise=False)
parser.add_argument("--shots", type=int,
                    help="Number of shots.")
parser.add_argument("--num-qubits", type=int,
                    help="Number of qubits (graph size)")
parser.add_argument("--num-layers", type=int, help="Number of QAOA layers")

# Boolean flags for print and draw (tri-state via defaults to None)
parser.add_argument("--print", dest="print",
                    action="store_true", help="Enable printing logs")
parser.add_argument("--no-print", dest="print",
                    action="store_false", help="Disable printing logs")
parser.set_defaults(print=None)

parser.add_argument("--draw", dest="draw",
                    action="store_true", help="Enable drawing plots")
parser.add_argument("--no-draw", dest="draw",
                    action="store_false", help="Disable drawing plots")
parser.set_defaults(draw=None)
parser.add_argument("--draw-name", type=str, help="Base name for plots")

parser.add_argument("--lr", type=float, help="Learning rate")
parser.add_argument("--beta1", type=float, help="Adam beta1")
parser.add_argument("--beta2", type=float, help="Adam beta2")
parser.add_argument("--eps", type=float, help="Adam epsilon")
parser.add_argument("--num-iterations", type=int, help="Training iterations")


# Scheduler configuration
parser.add_argument("--scheduler-type",
                    choices=["onecycle", "exponent"], help="Scheduler type")
parser.add_argument("--max-lr", type=float,
                    help="OneCycle: maximum learning rate")
parser.add_argument("--total-steps", type=int, help="OneCycle: total steps")
parser.add_argument("--pct-start", type=float,
                    help="OneCycle: percent of steps increasing LR")
parser.add_argument("--div-factor", type=float,
                    help="OneCycle: initial div factor")
parser.add_argument("--final-div-factor", type=float,
                    help="OneCycle: final div factor")
parser.add_argument("--gamma", type=float, help="Exponent: decay gamma")

if __name__ == "__main__":
    print("test_get_thetas.py")
    args = parser.parse_args()

    if args.environment is not None:
        models.get_qaoa_optimized.config["environment"] = args.environment
    if args.hw_noise is not None:
        models.get_qaoa_optimized.config["hw_noise"] = args.hw_noise
    if args.shots is not None:
        models.get_qaoa_optimized.config["shots"] = args.shots

    if args.num_qubits is not None:
        models.get_qaoa_optimized.config["num_qubits"] = args.num_qubits
    if args.num_layers is not None:
        models.get_qaoa_optimized.config["num_layers"] = args.num_layers

    if args.print is not None:
        models.get_qaoa_optimized.config["print"] = args.print
    if args.draw is not None:
        models.get_qaoa_optimized.config["draw"] = args.draw
    if args.draw_name is not None:
        models.get_qaoa_optimized.config["draw_name"] = args.draw_name

    if args.lr is not None:
        models.get_qaoa_optimized.config["lr"] = args.lr
    if args.beta1 is not None:
        models.get_qaoa_optimized.config["beta1"] = args.beta1
    if args.beta2 is not None:
        models.get_qaoa_optimized.config["beta2"] = args.beta2
    if args.eps is not None:
        models.get_qaoa_optimized.config["eps"] = args.eps
    if args.num_iterations is not None:
        models.get_qaoa_optimized.config["num_iterations"] = args.num_iterations

    # Scheduler overrides
    if args.scheduler_type is not None:
        models.get_qaoa_optimized.config["scheduler_spec"]["type"] = args.scheduler_type
        # Reset to minimal defaults for type
        if args.scheduler_type == "onecycle":
            spec = models.get_qaoa_optimized.config["scheduler_spec"]
            spec.update({k: v for k, v in {
                "max_lr": args.max_lr,
                "total_steps": args.total_steps,
                "pct_start": args.pct_start,
                "div_factor": args.div_factor,
                "final_div_factor": args.final_div_factor,
            }.items() if v is not None})
        elif args.scheduler_type == "exponent":
            # minimize/override to exponent fields only
            models.get_qaoa_optimized.config["scheduler_spec"] = {
                "type": "exponent"}
            if args.gamma is not None:
                models.get_qaoa_optimized.config["scheduler_spec"]["gamma"] = args.gamma
    else:
        # If type not overridden, still allow field-level overrides for existing type
        if models.get_qaoa_optimized.config["scheduler_spec"].get("type") == "onecycle":
            spec = models.get_qaoa_optimized.config["scheduler_spec"]
            for key, val in [("max_lr", args.max_lr), ("total_steps", args.total_steps),
                             ("pct_start", args.pct_start), ("div_factor",
                                                             args.div_factor),
                             ("final_div_factor", args.final_div_factor)]:
                if val is not None:
                    spec[key] = val
        elif models.get_qaoa_optimized.config["scheduler_spec"].get("type") == "exponent":
            if args.gamma is not None:
                models.get_qaoa_optimized.config["scheduler_spec"]["gamma"] = args.gamma

    # RUN!!
    environment = models.get_qaoa_optimized.config.get(
        "environment", "simulator")
    hw_noise = models.get_qaoa_optimized.config.get(
        "hw_noise", False
    )
    num_qubits = models.get_qaoa_optimized.config.get("num_qubits", 5)
    num_layers = models.get_qaoa_optimized.config.get("num_layers", 2)

    graph, edge_list = create_long_graph(num_qubits)
    # thetas = np.random.randn(2 * num_layers) * np.pi
    thetas = np.array([0.189 * np.pi] * num_layers +
                      [0.004 * np.pi] * num_layers)

    if environment == "qpu":
        backend = qpu_get_device()
    elif environment == "simulator" and hw_noise is True:
        print("Noisy Sim")
        backend = AerSimulator.from_backend(qpu_get_device())
    elif environment == "simulator" and hw_noise is False:
        backend = AerSimulator()
    else:
        raise Exception("Wrong environment setting.")

    with Session(backend=backend) as session:
        thetas = grad_train(graph, thetas, backend, session)
        grad_validate(graph, thetas, backend, session)
