import random
import string
import time
from neat.activations import sigmoid
from neat.organism import Organism
from neat.species import Species
from neat.reproduction import Reproducer
from itertools import count
import pickle

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
        self.create_initial_generation()

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
            organism_list = list(self.organisms)
            fitnesses = eval_func(organism_list)
            for i, organism in enumerate(organism_list):
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

    def create_initial_generation(self):
        self.reproducer = Reproducer(self, self.innovation_number)
        self.reproducer.create_initial_generation()


    def run(self, eval_func, generations, en_masse=False) -> Organism:
        """
        Run the population for a number of generations
        """

        for i in range(generations):
            self.best = self.evaluate_generation(eval_func, en_masse)
            self.reproducer.reproduce()

            for reporter in self.reporters:
                reporter.report()

            if self.config["goal_fitness"] != None:
                if self.best.fitness >= self.config["goal_fitness"]:
                    return self.end_training(self.best)
                
            self.generation += 1

        return self.end_training(self.best)

    def end_training(self, return_value):
        for reporter in self.reporters:
            reporter.complete()
        return return_value
    
    def to_dict(self):
        dictionary = {
            "meta": {
                "config": self.config,
                "id": self.id
            },
            "species": [species.to_dict() for species in self.species],
            "organisms": [organism.to_dict() for organism in self.organisms],
            "n": self.n,
            "generation": self.generation,
            "innovation_number": next(self.innovation_number),
        }
        return dictionary

    def to_file(self, path):
        dictionary = self.to_dict()
        with open(path, "wb") as f:
            pickle.dump(dictionary, f)
        
        print(f"Saved population to {path}")
            
    @staticmethod
    def from_dict(dictionary):
        """
        Load a population from a dictionary. Note: this will not load the reporters.
        """
        population = Population(dictionary["n"], dictionary["meta"]["config"], dictionary["meta"]["id"])
        population.generation = dictionary["generation"]
        population.innovation_number = count(dictionary["innovation_number"])
        population.organisms = {Organism.from_dict(organism, population=population) for organism in dictionary["organisms"]}
        population.species = {Species.from_dict(species, population.config, population.organisms) for species in dictionary["species"]}
        population.reproducer = Reproducer(population, population.innovation_number)

        return population
    
    @staticmethod
    def from_file(path: str):
        """
        Load a population from a file. Note: this will not load the reporters.
        """
        with open(path, "rb") as f:
            dictionary = pickle.load(f)
        return Population.from_dict(dictionary)


        
