import random
from organism import Organism, BaseOrganism, InputNode, OutputNode, Connection, Node
from itertools import count


class Reproducer:
    def __init__(self, population):
        self.population = population
        self.config = population.config
        self.genome_number = count(0)

    def asexual_reproduction(self, organism):
        """
        Create a new organism from a single parent
        """
        new_organism = organism.copy(
            "{}_{:02X}".format(self.population.generation, next(self.genome_number))
        )
        # Mutate the new organism
        new_organism.mutate()
        return new_organism

    def create_initial_generation(self):
        """
        Create the initial population
        """

        for i in range(self.population.n):
            self.population.organisms.append(
                BaseOrganism(
                    "{}_{:02X}".format(
                        self.population.generation, next(self.genome_number)
                    ),
                    self.population.input_nodes,
                    self.population.output_nodes,
                    self.config,
                )
            )

    def reproduce(self):
        """
        Reproduce the current generation
        """
        self.genome_number = count(0)

        # Select which organisms will reproduce
        # Sort organisms by fitness (this seems really slow, maybe there's a better way?)
        self.population.organisms.sort(key=lambda x: x.fitness, reverse=True)
        spaces = int(self.population.n * self.config["elimination_threshold"])
        self.population.organisms = self.population.organisms[0:spaces]
        while spaces > 0:
            # Select a random organism to reproduce
            organism = random.choice(self.population.organisms)
            self.population.organisms.append(self.asexual_reproduction(organism))
            spaces -= 1

        for organism in self.population.organisms:
            for layer in organism.layers:
                for node in layer:
                    node.value = node.bias
