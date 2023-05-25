import math
import random
import time
from organism import Organism, BaseOrganism, InputNode, OutputNode, Connection, Node
from itertools import count
from species import Species


class Reproducer:
    def __init__(self, population, innovation_number_counter: count):
        self.population = population
        self.config = population.config
        self.generation_organism_number = count(0)
        self.innovation_number_counter = innovation_number_counter
        self.species_number = count(0)

    def asexual_reproduction(self, organism):
        """
        Create a new organism from a single parent
        """
        new_organism = organism.copy(
            "{}_{:02X}".format(
                self.population.generation, next(self.generation_organism_number)
            )
        )
        # Mutate the new organism
        new_organism.mutate()
        return new_organism

    def create_initial_generation(self):
        """
        Create the initial population
        """

        for i in range(self.population.n):
            self.population.organisms.add(
                BaseOrganism(
                    "{}_{:02X}".format(
                        self.population.generation,
                        next(self.generation_organism_number),
                    ),
                    self.innovation_number_counter,
                    self.config,
                )
            )

        # Speciate em'
        # This isn't really random, but it's the first generation so it doesn't matter
        independents = set(self.population.organisms)
        while len(independents):
            representative = independents.pop()
            species = Species(representative, next(self.species_number), self.config)
            representative.species = species

            for organism in independents.copy():
                if species.compatible(organism):
                    species.add_organism(organism)
                    independents.remove(organism)

            self.population.species.add(species)

    def speciate(self):
        """
        Speciate the current generation
        """
        for species in self.population.species:
            species.represetative = random.choice(list(species.members))
            species.members = set()

        independents = set(self.population.organisms)
        while len(independents):
            candidate = independents.pop()
            for species in self.population.species:
                if species.compatible(candidate):
                    species.add_organism(candidate)
                    break
            else:
                self.population.species.add(
                    Species(candidate, next(self.species_number), self.config)
                )
                species.add_organism(candidate)

    def calculate_species_allocation(self):
        for species in self.population.species.copy():
            if len(species.members) == 0:
                self.population.species.remove(species)

        for species in self.population.species:
            species.total_adjusted_fitness = max(
                0, sum([o.adjusted_fitness for o in species.members])
            )

        population_total_adjusted_fitness = sum(
            [s.total_adjusted_fitness for s in self.population.species]
        )

        total_allocation = 0
        for species in self.population.species.copy():
            if population_total_adjusted_fitness > 0:
                species.allocation = max(
                    self.config["min_species_size"],
                    math.floor(
                        self.population.n
                        * (
                            species.total_adjusted_fitness
                            / population_total_adjusted_fitness
                        )
                    ),
                )
            else:
                species.allocation = self.config["min_species_size"]
            total_allocation += species.allocation

        # Normalize the allocations so they roughly add up to the population size
        norm = self.population.n / total_allocation
        for species in self.population.species:
            species.allocation = max(
                self.config["min_species_size"], math.floor(species.allocation * norm)
            )

    def reproduce(self):
        """
        Reproduce the current generation
        """
        self.generation_organism_number = count(0)
        self.population.organisms = set()
        print(len(self.population.organisms))

        self.calculate_species_allocation()

        for species in self.population.species:
            # Remove the worst performing organisms
            members = sorted(list(species.members), key=lambda o: o.adjusted_fitness)
            if len(members) > 5:
                self.population.organisms.add(members[-1])
            for i in range(species.allocation - 1):
                # Choose a random parent weighted by fitness
                parent = random.choices(
                    members, weights=[o.adjusted_fitness for o in members]
                )[0]
                self.population.organisms.add(self.asexual_reproduction(parent))

        # for i in range(self.population.n):
        #     species = random.choice(list(self.population.species))
        #     parent = random.choice(list(species.members))
        #     self.population.organisms.add(self.asexual_reproduction(parent))

        # Reset the organisms
        for organism in self.population.organisms:
            for node in organism.nodes:
                node.value = node.bias

        self.speciate()
