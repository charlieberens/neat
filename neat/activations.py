import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def modified_sigmoid(x):
    return 1 / (1 + np.exp(-4.9 * x))
