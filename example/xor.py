import sys
import os

from neat.organism import Organism
from neat.population import Population
from neat.config import get_config
from neat.reporter import PrintReporter, ProgressReporter, StatReporter

"""
Before running this, run `source ./demos/xor.config.json` to set the correct directory for module imports.
"""

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
    config = get_config("example/xor.config.json")
    p = Population(200, config)
    p.add_reporter(PrintReporter())
    p.add_reporter(StatReporter(["best_fitness", "avg_fitness", "worst_fitness"], frequency=1))
    p.add_reporter(ProgressReporter(frequency=50))

    w = p.run(eval_xor, 150)

    w.draw()

if __name__ == "__main__":
    main()
