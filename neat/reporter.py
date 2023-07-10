import os
from typing import List
from neat.population import Population

class Reporter:
    """
    Reporters process information about the population.
    """

    def __init__(self):
        self.population = None

    def report(self):
        pass

    def complete(self):
        pass


class PrintReporter(Reporter):
    """
    The Print Reporter prints generation information to the console.
    """

    def __init__(self, species=False):
        self.species = species
        super().__init__()

    def report(self):
        print(
            "Generation: {} - ID: {} - Best: {:.4f} - Species: {} - Organisms: {} - Avg Nodes: {:.4f} - Node Range: {} - Connection Range: {}".format(
                self.population.generation,
                self.population.best.id,
                self.population.best.fitness,
                len(self.population.species),
                len(self.population.organisms),
                sum([o.node_count for o in self.population.organisms])
                / len(self.population.organisms),
                [
                    min([o.node_count for o in self.population.organisms]),
                    max([o.node_count for o in self.population.organisms]),
                ],
                [
                    min(
                        [
                            len([c for c in o.connections])
                            for o in self.population.organisms
                        ]
                    ),
                    max(
                        [
                            len([c for c in o.connections])
                            for o in self.population.organisms
                        ]
                    ),
                ],
            )
        )
        if self.species:
            print("\n")
            print("Max Fitness\tAvg Fitness\tN")
            print("-----------\t-----------\t-")
            for s in self.population.species:
                print(
                    "{:11.4f}\t{:11.4f}\t{}".format(
                        max([o.fitness for o in s.members]),
                        sum([o.fitness for o in s.members]) / len(s.members),
                        len(s.members),
                    )
                )
            


class StatReporter:
    """
    The data reporter saves statistics about the population to a csv.

    Stats
    -----
        "best_fitness": population.best.fitness,
        "avg_fitness": average of all organisms' fitness,
        "worst_fitness": population.worst.fitness,

    NOTE: Gernerations will be reported automatically

    """

    def __init__(
        self,
        stats: List[str],
        filename: str = None,
        by_species: bool = False,
        frequency: int = 1,
    ):
        # TODO - Implement by_species
        self.stats = stats
        self.filename = filename
        self.by_species = by_species
        self.frequency = frequency
        self.population = None
        self.rows = []

    def get_stat(self, stat):
        mapper = {
            "best_fitness": max(
                self.population.organisms, key=lambda o: o.fitness
            ).fitness,
            "avg_fitness": sum([o.fitness for o in self.population.organisms])
            / len(self.population.organisms),
            "worst_fitness": min(
                self.population.organisms, key=lambda o: o.fitness
            ).fitness,
        }
        return mapper.get(stat, None)

    def report(self):
        if self.population.generation % self.frequency == 0:
            self.rows.append(
                {
                    "generation": self.population.generation,
                    **{stat: self.get_stat(stat) for stat in self.stats},
                }
            )

    def write_to_csv(self):
        self.filename = self.filename or "{}.csv".format(self.population.id)
        path = os.path.join(self.population.config["stat_directory"], self.filename)
        # Check if stat directory exists
        if not os.path.exists(self.population.config["stat_directory"]):
            os.makedirs(self.population.config["stat_directory"])
        
        arr = [["generation"] + [stat for stat in self.stats]] + [[r[col] for col in r] for r in self.rows]
        
        if os.path.exists(path):
            with open(path, "a") as f:
                for row in arr[1:]:
                    f.write(",".join([str(col) for col in row]) + "\n")
        else:
            with open(path, "w") as f:
                for row in arr:
                    f.write(",".join([str(col) for col in row]) + "\n")

    def complete(self):
        self.write_to_csv()


class ProgressReporter:
    """
    The Progress Reporter saves the population to a pickle file every frequency generations.
    """

    def __init__(self, by_species: bool = False, frequency: int = 1):
        # TODO - Implement by_species
        self.population = None
        self.by_species = by_species
        self.frequency = frequency
        self.rows = []

    def report(self, force=False):
        if self.population.generation % self.frequency == 0 or force:
            if not os.path.exists(self.population.config["progress_directory"]):
                os.makedirs(self.population.config["progress_directory"])
            self.population.to_file(os.path.join(self.population.config["progress_directory"],"{}-{}".format(self.population.id, self.population.generation)))

    def complete(self):
        self.report(force = True)
