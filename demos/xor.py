import sys
import os

from neat.organism import Organism
from neat.population import Population
from neat.config import get_config
from neat.reporter import PrintReporter

def eval_xor(organism):
    """
    XOR fitness function
    """
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [0, 1, 1, 0]
    err = 0
    for i, o in zip(inputs, outputs):
        output = organism.evaluate(i)
        err += abs(o - output[0])
    return (4- err)**2

def main():
    config = get_config("demos/xor.config.json")
    p = Population(200, config)
    p.add_reporter(PrintReporter())
    w = p.run(eval_xor, 300)
    w.draw()

if __name__ == "__main__":
    main()