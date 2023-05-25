import time
from activations import sigmoid
from organism import Organism, BaseOrganism
from reproduction import Reproducer
from itertools import count


class Population:
    def __init__(
        self,
        n: int,
        config: dict,
    ):
        """
        Parameters
        ----------
            n: int
                Number of genomes in the population
        """
        self.n = n
        self.input_nodes = config["input_nodes"]
        self.output_nodes = config["output_nodes"]
        self.config = config
        self.generation = 0
        self.innovation_number = count(0)
        self.organisms = set()
        self.species = set()

    def evaluate_generation(self, eval_func) -> Organism:
        """
        Evaluate the current generation
        """
        best = None

        for organism in self.organisms:
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
            print(
                "Generation: {} - ID: {} - Best: {} - Species: {} - Organisms: {} - Avg Nodes: {} - Node Range: {} - Connection Range: {}".format(
                    self.generation,
                    self.best.id,
                    self.best.fitness,
                    len(self.species),
                    len(self.organisms),
                    sum([len(o.nodes) for o in self.organisms]) / len(self.organisms),
                    [
                        min([len(o.nodes) for o in self.organisms]),
                        max([len(o.nodes) for o in self.organisms]),
                    ],
                    [
                        min(
                            [
                                len([c for c in o.connections if c.enabled])
                                for o in self.organisms
                            ]
                        ),
                        max(
                            [
                                len([c for c in o.connections if c.enabled])
                                for o in self.organisms
                            ]
                        ),
                    ],
                )
            )
            reproducer.reproduce()
            self.generation += 1

        return self.best
