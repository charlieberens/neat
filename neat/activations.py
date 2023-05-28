import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def modified_sigmoid(x):
    return 1 / (1 + math.exp(-4.9 * x))
