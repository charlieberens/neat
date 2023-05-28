import os
from typing import List
from neat.population import Population
import pandas as pd


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

    def __init__(self):
        super().__init__()

    def report(self):
        print(
            "Generation: {} - ID: {} - Best: {} - Species: {} - Organisms: {} - Avg Nodes: {} - Node Range: {} - Connection Range: {}".format(
                self.population.generation,
                self.population.best.id,
                self.population.best.fitness,
                len(self.population.species),
                len(self.population.organisms),
                sum([len(o.nodes) for o in self.population.organisms])
                / len(self.population.organisms),
                [
                    min([len(o.nodes) for o in self.population.organisms]),
                    max([len(o.nodes) for o in self.population.organisms]),
                ],
                [
                    min(
                        [
                            len([c for c in o.connections if c.enabled])
                            for o in self.population.organisms
                        ]
                    ),
                    max(
                        [
                            len([c for c in o.connections if c.enabled])
                            for o in self.population.organisms
                        ]
                    ),
                ],
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

    def write_to_csv(self, df):
        self.filename = self.filename or "{}.csv".format(self.population.id)
        path = os.path.join(self.population.config["stat_directory"], self.filename)
        # Check if stat directory exists
        if not os.path.exists(self.population.config["stat_directory"]):
            os.makedirs(self.population.config["stat_directory"])
        if os.path.exists(path):
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)

    def complete(self):
        df = pd.DataFrame(self.rows)
        self.write_to_csv(df)


class ProgressReporter:
    """
    The Progress Reporter saves the best organism from each generation (or generation and species) to a csv.
    """

    def __init__(self, filename: str, by_species: bool = False, frequency: int = 1):
        # TODO - Implement by_species
        self.filename = filename
        self.by_species = by_species
        self.frequency = frequency
        self.population = None
        self.rows = []

    def report(self):
        pass
