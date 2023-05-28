import random
import string
import time
from neat.activations import sigmoid
from neat.organism import Organism, BaseOrganism
from neat.reproduction import Reproducer
from itertools import count


class Population:
    def __init__(
        self,
        n: int,
        config: dict,
        id: str = "".join(random.choices(string.digits + string.ascii_uppercase, k=6)),
    ):
        """
        Parameters
        ----------
            n: int
                Number of genomes in the population
            config: dict
                Configuration dictionary
            id: str
                ID of the population. Used for saving progress, statistics, etc.
        """
        self.n = n
        self.input_nodes = config["input_nodes"]
        self.output_nodes = config["output_nodes"]
        self.config = config
        self.generation = 0
        self.innovation_number = count(0)
        self.organisms = set()
        self.species = set()
        self.reporters = []
        self.id = id
        self.timers = [0,0,0]

    def add_reporter(self, *reporters):
        for reporter in reporters:
            reporter.population = self
            self.reporters.append(reporter)

    def evaluate_generation(self, eval_func) -> Organism:
        """
        Evaluate the current generation
        """
        best = None

        for organism in self.organisms:
            organism.layers = organism.calculate_layers()
            organism.fitness = eval_func(organism)
            organism.adjusted_fitness = organism.fitness / len(organism.species.members)
            if not best or organism.fitness > best.fitness:
                best = organism

        return best

    def run(self, eval_func, generations) -> Organism:
        """
        Run the population for a number of generations
        """
        reproducer = Reproducer(self, self.innovation_number)
        reproducer.create_initial_generation()

        for i in range(generations):
            self.best = self.evaluate_generation(eval_func)

            for reporter in self.reporters:
                reporter.report()

            if self.config["goal_fitness"] != None:
                if self.best.fitness >= self.config["goal_fitness"]:
                    return self.end_training(self.best)
            reproducer.reproduce()
            self.generation += 1

        return self.end_training(self.best)

    def end_training(self, return_value):
        for reporter in self.reporters:
            reporter.complete()
        return return_value
