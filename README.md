# QAOA
## Quantum Approximate Optimization Algorithm
Two methods to run QAOA: via a QPU of Yonsei (`ibm_yonsei`) or a CPU.

## Initialization
```shell
git clone https://github.com/Jiwhan-Kim/QAOA-Simple.git
cd ./QAOA-Simple

conda env create -f ./environment.yml
conda activate qiskit312
```

If you want to run on `ibm_yonsei`, follow the instructions below.
Otherwise, you can only run the code on a simulator.


### Step 1.
Create an API key on https://quantum.cloud.ibm.com/.

### Step 2.
Create an `.env` file.
```shell
touch .env
```

Then write your `IBM_TOKEN` and `IBM_INSTANCE` to `.env` file.
```text
IBM_TOKEN="YOUR_IBM_TOKEN"
IBM_INSTANCE="YOUR_IBM_INSTANCE"
```

### Step 3.
Save the account to your local computer.
```shell
python ibm_setup.py
```

## Simulator and QPU

There are two modes at the quantum comptuing: `Estimator` and `Sampler`.

Both simulator and QPU(`ibm_yonsei`) supports these two modes.

These are implemented in `sim_estimator`, `sim_sampler`, `qpu_estimator`, and `qpu_sampler`.

You can either import them or directly run the codes.

```shell
# Runs QAOA with given parameters.
# A graph max-cut is plotted and saved in ./images/sims1-4-distribute.png

python sim_sampler.py
```

## Run QAOA with COBYLA-based Optimization

```shell
# Run QAOA with COBYLA-based Optimization
python scipy_graph_partition.py

# Options
## Draw an input graph
## Use a simulator
## Find max-cut of 15-qubit Graph
## A 3-layer QAOA layer
## 50 iterations when optimization
python scipy_graph_partition.py --draw 1 --env simulator --qubits 15 --layers 3 --iters 50

## Use a QPU - ibm_yonsei 
## Find max-cut of 10-qubit Graph
## A 2-layer QAOA layer
## 30 iterations when optimization
python scipy_graph_partition.py --env qpu --qubits 10 --layers 2 --iters 30
```

The example here is for searching 'max-cut' partition for a graph with 15 nodes and 14 edges.
The maximum cut is 14(Partitioning into 1010...0101 or 0101...1010).

## Run QAOA with Gradient-Descent-based Optimization

```shell
# Run QAOA with Gradient-based Optimization
python stoch_graph_partition.py
```

The code will run QAOA with Gradient-Descent-based Optimization

- `[--draw <1 | 0>]` determines whether the code plots the result or not.
- `[--env <qpu | simulator>]` sets environments: `ibm_yonsei` or `simulator`.
- `[--qubits <qubit>]` sets the number of nodes.
- `[--layers <layer>]` sets the number of layers in QAOA.
- `[--iters <iters>]` sets the number of iterations at optimizing parameters.

The example here is for searching 'max-cut' partition for a graph with 15 nodes and 14 edges.
The maximum cut is 14(Partitioning into 1010...0101 or 0101...1010).

## GNN-based Initial Parameter Estimation for QAOA

```shell
# GNN-based Initial Parameter Estimation for QAOA
python learns_embedding.py
```

The code trains GNN (GraphSAGE) model for predicting initial parameters for QAOA.

It stores data(Graph Dataset, Training Dataset, Model) in `./datas/*.pt`.
The example dataset is already generated(`./datas/550_*.pt`).
You can now test the model by pre-existing model.

Or you can change the dataset by changing `num_graphs = 550` to other values like `1100`.

## Tips
Run the code while you are in `tmux`.
A task requested to `ibm_yonsei` is queued and it responses after all the previous tasks are done.

When running on `tmux`, set `--draw 0`(recommended).
A plotting when no display are attached would occur errors.

