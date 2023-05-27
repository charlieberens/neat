import math
import random
import time
from organism import (
    CrossOverOrganism,
    Organism,
    BaseOrganism,
    InputNode,
    OutputNode,
    Connection,
    Node,
)
from itertools import count
from species import Species


class Reproducer:
    def __init__(self, population, innovation_number_counter: count):
        self.population = population
        self.config = population.config
        self.generation_organism_number = count(0)
        self.innovation_number_counter = innovation_number_counter
        self.species_number = count(0)
        self.species: Species = set()
        self.generation_innovations = {}

        def innovation_number_tracker(innovation):
            if innovation in self.generation_innovations.keys():
                return self.generation_innovations[innovation]
            else:
                self.generation_innovations[innovation] = next(
                    self.innovation_number_counter
                )
                return self.generation_innovations[innovation]

        self.generation_innovation_number_tracker = innovation_number_tracker

    def asexual_reproduction(self, organism):
        """
        Create a new organism from a single parent
        """
        new_organism = organism.copy(
            "{}-{:02X}".format(
                self.population.generation, next(self.generation_organism_number)
            )
        )
        # Mutate the new organism
        new_organism.mutate()
        return new_organism

    def sexual_reproduction(self, parent1, parent2):
        """
        Crossover
        """
        new_organism = CrossOverOrganism(
            "{}-{:02X}".format(
                self.population.generation, next(self.generation_organism_number)
            ),
            self.generation_innovation_number_tracker,
            parent1,
            parent2,
            self.config,
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
                    "{}-{:02X}".format(
                        self.population.generation,
                        next(self.generation_organism_number),
                    ),
                    self.generation_innovation_number_tracker,
                    self.config,
                    self.population,
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
            species.age += 1
            species.best_fitness_age += 1
            for member in species.members:
                if member.fitness > species.best_fitness:
                    species.best_fitness = member.fitness
                    species.best_fitness_age = 0

            if len(species.members) == 0 or (
                len(self.population.species) >= 2
                and self.config["stagnation_threshold"] > 0
                and species.best_fitness_age >= self.config["stagnation_threshold"]
            ):
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

        test = self.population.organisms.pop()

        self.generation_organism_number = count(0)
        self.population.organisms = set()

        self.calculate_species_allocation()

        for species in self.population.species:
            species.age += 1
            # Remove the worst performing organisms
            members = sorted(list(species.members), key=lambda o: o.adjusted_fitness)
            if len(members) > 5:
                self.population.organisms.add(members[-1])
            for i in range(species.allocation - 1):
                if random.random() < self.config["interspecies_mating_rate"]:
                    species2 = random.choice(list(self.population.species))
                    parent1 = random.choice(list(species.members))
                    parent2 = random.choice(list(species2.members))
                    self.population.organisms.add(
                        self.sexual_reproduction(parent1, parent2)
                    )
                else:
                    if random.random() < self.config["crossover_rate"]:
                        parent1 = random.choice(members)
                        parent2 = random.choice(members)
                        self.population.organisms.add(
                            self.sexual_reproduction(parent1, parent2)
                        )
                    else:
                        parent = random.choice(members)
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
