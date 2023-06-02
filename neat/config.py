import json
from neat.activations import activation_mapper


"""
Parameters
----------
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
    weight_perturb_amount: float
        Amount to perturb weight by
    weight_max_value: float
        Maximum value of a weight
    weight_min_value: float
        Minimum value of a weight
    bias_mut_rate: float
        Chance of bias mutation
    bias_perturb_rate: float
        Chance of perturbing bias
    bias_perturb_amount: float
        Amount to perturb bias by
    bias_max_value: float
        Maximum value of a bias
    bias_min_value: float
        Minimum value of a bias
    single_structural_mutation: bool
        Whether to only allow one structural mutation per mutation
    node_mut_rate: float
        Chance of nodal mutation
    connection_mut_rate: float
        Chance of connection mutation
    enable_mut_rate: float
        Chance of enabling a disabled gene
    crossover_rate: float
        Chance of crossover
    interspecies_mating_rate: float
        Chance of interspecies mating
    stagnation_threshold: int
        Number of generations without improvement before a species is considered stagnant
    elimination_threshold: int
        Proportion of organisms to eliminate from a species
    min_species_size: int
        Minimum number of organisms in a species
"""

"""
Other Config
------------
    stat_directory:
        Directory to save statistics to (used by StatReporter)
    progress_directory:
        Directory to save progress to (used by ProgressReporter)
"""

defaults = {
    "input_nodes": 2,
    "output_nodes": 1,
    "goal_fitness": None,
    "transfer_function": "modified_sigmoid",
    "c1": 1.0,
    "c2": 1.0,
    "c3": 0.4,
    "delta_thresh": 3.0,
    "weight_mut_rate": 0.8,
    "weight_perturb_rate": 0.9,
    "weight_perturb_amount": 5,
    "weight_max_value": 10.0,
    "weight_min_value": -10.0,
    "bias_mut_rate": 0.8,
    "bias_perturb_rate": 0.9,
    "bias_perturb_amount": 1,
    "bias_max_value": 2,
    "bias_min_value": -2,
    "single_structural_mutation": True,
    "node_mut_rate": 0.03,
    "connection_mut_rate": 0.05,
    "enable_mut_rate": 0,
    "crossover_rate": 0.75,
    "interspecies_mating_rate": 0.001,
    "stagnation_threshold": 20,
    "elimination_threshold": 0.25,
    "min_species_size": 5,
    "stat_directory": "statistics",
    "progress_directory": "progress",
    "organism_directory": "organisms",
}

config = defaults.copy()


def get_config(file=None):
    """
    Load a config from a file
    """
    file_config = {}
    if file:
        with open(file, "r") as f:
            file_config = json.load(f)
    config.update(file_config)
    try:
        config["activation"] = activation_mapper[config["transfer_function"]]
    except KeyError:
        raise KeyError(
            f"Transfer function {config['transfer_function']} is not a valid activation"
        )
    return config