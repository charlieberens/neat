from organism import Organism, Connection


class Species:
    def __init__(self, representative, species_number, config: dict):
        self.config = config
        self.representative: Organism = representative
        self.species_number = species_number
        self.members = set([representative])

    def add_organism(self, organism):
        self.members.add(organism)
        organism.species = self

    def compatability_distance(self, organism):
        # This could be so much simpler, but for whatever reason
        # I think that this is faster. Pure instinct, no real evidence.
        organism_genes = {}
        representative_genes = {}
        m_organism_innovation_number = 0
        for connection in organism.connections:
            organism_genes[connection.innovation_number] = connection
            if connection.innovation_number > m_organism_innovation_number:
                m_organism_innovation_number = connection.innovation_number
        m_representative_innovation_number = 0
        for connection in self.representative.connections:
            representative_genes[connection.innovation_number] = connection
            if connection.innovation_number > m_representative_innovation_number:
                m_representative_innovation_number = connection.innovation_number

        smaller_innovation_number = min(
            m_organism_innovation_number, m_representative_innovation_number
        )
        shared_genes = set(organism_genes.keys()) & set(representative_genes.keys())
        different_genes = set(organism_genes.keys()) ^ set(representative_genes.keys())

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
