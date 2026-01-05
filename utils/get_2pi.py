import numpy as np


def get_2pi(thetas: np.ndarray):
    # theta -> [-np.pi, np.pi)
    return (thetas + np.pi) % (2 * np.pi) - np.pi
