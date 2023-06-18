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
        self.config = config
        self.generation = 0
        self.innovation_number = count(config["input_nodes"] + config["output_nodes"] + 1)
        self.organisms = set()
        self.species = set()
        self.reporters = []
        self.id = id

    def add_reporter(self, *reporters):
        for reporter in reporters:
            reporter.population = self
            self.reporters.append(reporter)

    def evaluate_generation(self, eval_func, en_masse) -> Organism:
        """
        Evaluate the current generation
        """
        best = None

        if en_masse:
            fitnesses = eval_func(self.organisms)
            for i, organism in enumerate(self.organisms):
                organism.fitness = fitnesses[i]
                organism.adjusted_fitness = organism.fitness / len(organism.species.members)
                if not best or organism.fitness > best.fitness:
                    best = organism
        else:
            for organism in self.organisms:
                organism.fitness = eval_func(organism)
                organism.adjusted_fitness = organism.fitness / len(organism.species.members)
                if not best or organism.fitness > best.fitness:
                # if not best or organism.node_count > best.node_count:
                    best = organism

        return best

    def run(self, eval_func, generations, en_masse=False) -> Organism:
        """
        Run the population for a number of generations
        """
        reproducer = Reproducer(self, self.innovation_number)
        reproducer.create_initial_generation()

        for i in range(generations):
            self.best = self.evaluate_generation(eval_func, en_masse)

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
