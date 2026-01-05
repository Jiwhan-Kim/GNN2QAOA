from qiskit_ibm_runtime import Session

from graphs import create_long_graph
from models import create_datasets
from cores import qpu_get_device

import models.get_qaoa_optimized

if __name__ == "__main__":
    graphs = [create_long_graph(i)[0] for i in range(5, 16)]
    backend = qpu_get_device()

    with Session(backend=backend) as session:
        create_datasets(graphs, 'output', backend, session)
