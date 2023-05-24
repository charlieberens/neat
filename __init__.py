# The following is based on "Evolving Neural Networks through Augmenting Topologies" by Kenneth O. Stanley and Risto Miikkulainen.
# https://doi.org/10.1162/106365602320169811
#
# Genomes are a list of node pairs
# Each connection gene contains the following information:
#  - In Node
#  - Out Node
#  - Weight
#  - Enabled
#  - Innovation Number. Innovation number communicates if two genes are homologous. Every time a new mutation occurs,
#   the global innovation number is incremented and assigned to the connection gene corresponding to the mutation. If a mutation
#   adds a node, then the two new connection genes are assigned innovation number n+1 and n+2, where n is the innovation number.
#       - We keep a list of mutations that occur within a generation. If a mutation occurs that has already occurred in that
#       generation, then we assign the new mutation the same innovation number as the previous mutation.
#
# Note: The original paper added a bias node to the input layer. Instead of doing this, we will just add a bias weight to each
# input node. It's cooler, and I'm much cooler than the original authors. Don't @ me.
#
# Connection Mutation:
# - Similar to any NE implementation. Each connection weight is either perturbed or not perturbed.
#
# Structural Mutation:
# (1) Add Connection
#   - Two previously disconnected nodes are connected
#   - The connection weight is randomly assigned
# (2) Add Node
#   - An existing connection is split
#   - The old connection is disabled and two new connections are added
#   - The weight of the connection leading into the new Node is 1
#   - The weight leading out is the same as the old connection
#
# Crossing Over:
# - For each matching gene (gene that both parents share), the offspring will randomly recieve the gene from either parent
# - The offspring will inherit all disjoint or excess genes (genes that are not shared by both parents) from their fitter parent
#
# Speciation:
# - We define a metric called compatability distance as follows:
#    - delta = c1 * E / N + c2 * D / N + c3 * W_bar
#       - E = number of excess genes
#       - D = number of disjoint genes
#       - W_bar = average weight difference of matching genes
#       - N = number of genes in the larger genome
#           - NOTE: The paper says that N can be set to 1 if both genomes are small, but I don't understand see why this is
#           necessary. Maybe just a small speedup?
#       - The paper used the following values: c1 = 1.0, c2 = 1.0, c3 = 0.4, delta_thresh = 3.0
#           - It hurts me a little bit that we're using one more variable than is necessary. But, that's fine. I'M FINE.
# - Each species is represented by a single genome called the representative genome from the previous generation.
# - Each genome is assigned to a species by computing the compatability distance between the genome and the representative genome
# - Why do we speciate? To protect topological innovations.
# - Wouldn't this just create garbage then protect it? Clearly not. I think it's because it takes time for new species to develop.
#   - TODO: Figure this out!
#
# Fitness Sharing:
# - Neat uses explicit fitness sharing, where the fitness of each genome is divided by the number of genomes in its species.
#   - This is done to prevent a single species from dominating the population
#   - fp_i = f_i / N_i
#       - fp_i = adjusted fitness of genome i
#       - f_i = raw fitness of genome i
#       - N_i = number of genomes in the species of genome i
#   - The number of offspring each species is allowed to produce is proportional to the sum of the adjusted fitnesses of the
#     genomes in the species. (This simplifies to the average raw fitness of the genomes in the species)
#       (*) In rare cases, when the fitness of the entire population doesn't improve for 20 generations. Only the top two species
#       are allowed to reproduce. This is done to prevent stagnation. (This seems arbitrary)
#   - Species then reproduce by first eliminating the lowest performing genomes in the species, and then randomly selecting
#     genomes to reproduce until the number of offspring is reached. (Possibly(?), the genomes are selected with a probability))
#       - I don't see why we eliminate the lowest performing genomes. It seems more natural to select genomes with
#         a probability proportional to their fitness.
#
# Parameters from Paper:
# - Transfer function: modified sigmoid
# - Speciation: c1 = 1.0, c2 = 1.0, c3 = 0.4, delta_thresh = 3.0
# - The best genome from each species is copied into the next generation without mutation (for species with more than 5 genomes)
# - Chance of weight mutation: 80%
#   - Given weight mutation, chance of perturbation for each weight: 90%. Chance of assigning new random weight: 10%
# - Chance a disabled gene (in both parents) is enabled in offspring: 25%
# - Percent of offspring produced by crossover: 75%
# - Inter-species mating rate: 0.001
# - Chance of structural mutation (nodal): 3%
# - Chance of structural mutation (connection): 5%
# - Stagnant species are removed after 15 generations

from population import Population
from organism import Organism
from config import config


def eval_function(organism):
    return sum(organism.evaluate([0, 0]))


def main():
    population = Population(150, 2, 2, config)
    population.run(eval_function, 20)


if __name__ == "__main__":
    main()
