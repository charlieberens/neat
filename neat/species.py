from neat.organism import Organism


class Species:
    def __init__(self, representative, species_number, config: dict):
        self.config = config
        self.representative: Organism = representative
        self.species_number = species_number
        self.members = set([representative])
        self.age = 0
        self.best_fitness = -10000
        self.best_fitness_age = 0
        self.total_adjusted_fitness = 0

    @staticmethod
    def from_dict(species_dict, config, organisms):
        species = Species(Organism.from_dict(species_dict["representative"]), species_dict["species_number"], config)
        species.age = species_dict["age"]
        species.best_fitness = species_dict["best_fitness"]
        species.best_fitness_age = species_dict["best_fitness_age"]
        species.total_adjusted_fitness = species_dict["total_adjusted_fitness"]
        species.members = set()

        for organism in organisms:
            if organism.id in species_dict["members"]:
                species.add_organism(organism)

        return species

    def to_dict(self):
        return {
            "species_number": self.species_number,
            "age": self.age,
            "representative": self.representative.to_dict(),
            "best_fitness": self.best_fitness,
            "best_fitness_age": self.best_fitness_age,
            "total_adjusted_fitness": self.total_adjusted_fitness,
            "members": [member.id for member in self.members],
        }

    def add_organism(self, organism):
        self.members.add(organism)
        organism.species = self

    def compatability_distance(self, organism):
        # This could be so much simpler, but for whatever reason
        # I think that this is faster. Pure instinct, no real evidence.
        organism_genes = {}
        representative_genes = {}


        for gene in organism.connections:
            organism_genes[gene.innovation_number] = gene
        for gene in self.representative.connections:
            representative_genes[gene.innovation_number] = gene
        
        smaller_innovation_number = min(
            organism.connections[-1].innovation_number,
            self.representative.connections[-1].innovation_number,
        )
        shared_genes = set(organism_genes.keys()) & set(representative_genes.keys())
        different_genes = set(organism_genes.keys() ^ representative_genes.keys()).union(set(representative_genes.keys() ^ organism_genes.keys()))

        W_bar = sum(
            [
                abs(organism_genes[i].weight - representative_genes[i].weight)
                for i in shared_genes
            ]
        ) / max(len(shared_genes), 1)

        disjoint_genes = len(
            [g for g in different_genes if g <= smaller_innovation_number]
        )
        excess_genes = len(different_genes) - disjoint_genes
        N = max(len(organism_genes), len(representative_genes))
        N = 5

        delta = (
            self.config["c1"] * excess_genes + self.config["c2"] * disjoint_genes
        ) / N + self.config["c3"] * W_bar / (
            self.config["weight_max_value"] - self.config["weight_min_value"]
        )

        return delta

    def compatible(self, organism):
        return self.compatability_distance(organism) <= self.config["delta_thresh"]
