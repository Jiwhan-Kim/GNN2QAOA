import numpy as np
import matplotlib.pyplot as plt


def plot_costs_log(costs, name: str):
    epochs = np.arange(len(costs))

    plt.plot(epochs, costs, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title("Minimizing Cost")
    plt.grid(True)
    plt.savefig(f"./images/{name}-costs-log.png")
    plt.close()


def plot_gradients_log(gradients, name: str):
    epochs = np.arange(len(gradients))

    plt.plot(epochs, gradients, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title("Gradients")
    plt.grid(True)
    plt.savefig(f"./images/{name}-gradients-log.png")
    plt.close()
