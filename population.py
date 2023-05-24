from activations import sigmoid
from organism import Organism, BaseOrganism
from reproduction import Reproducer


class Population:
    def __init__(
        self,
        n: int,
        input_nodes: int,
        output_nodes: int,
        config: dict,
    ):
        """
        Parameters
        ----------
            n: int
                Number of genomes in the population
            input_nodes: int
                Number of input nodes
            output_nodes: int
                Number of output nodes
            transfer_function: function
                Transfer function for the nodes
            c1: float
                Speciation parameter
            c2: float
                Speciation parameter
            c3: float
                Speciation parameter
            delta_thresh: float
                Speciation parameter
            weight_mut_rate: float
                Chance of weight mutation
            weight_perturb_rate: float
                Chance of perturbing weight
            weight_max_value: float
                Maximum value of a weight
            weight_min_value: float
                Minimum value of a weight
            bias_mut_rate: float
                Chance of bias mutation
            bias_perturb_rate: float
                Chance of perturbing bias
            bias_max_value: float
                Maximum value of a bias
            bias_min_value: float
                Minimum value of a bias
            nodal_mut_rate: float
                Chance of nodal mutation
            connection_mut_rate: float
                Chance of connection mutation
            enable_rate: float
                Chance of enabling a disabled gene
            crossover_rate: float
                Chance of crossover
            interspecies_mating_rate: float
                Chance of interspecies mating
            stagnation_threshold: int
                Number of generations without improvement before a species is considered stagnant
            elimination_threshold: int
                Proportion of organisms to eliminate from a species
        """
        self.n = n
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.config = config
        self.generation = 0
        self.innovation_number = 0
        self.organisms = []

    def evaluate_generation(self, eval_func) -> Organism:
        """
        Evaluate the current generation
        """
        best = None

        for organism in self.organisms:
            organism.fitness = eval_func(organism)
            if not best or organism.fitness > best.fitness:
                best = organism

        return best

    def run(self, eval_func, generations):
        """
        Run the population for a number of generations
        """
        reproducer = Reproducer(self)
        reproducer.create_initial_generation()

        for i in range(generations):
            self.best = self.evaluate_generation(eval_func)
            print(
                "Generation: {} - ID: {} - Best: {}".format(
                    self.generation, self.best.id, self.best.fitness
                )
            )
            reproducer.reproduce()
            self.generation += 1
