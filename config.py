from activations import sigmoid

"""
Parameters
----------
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

defaults = {
    "transfer_function": sigmoid,
    "c1": 1.0,
    "c2": 1.0,
    "c3": 0.4,
    "delta_thresh": 3.0,
    "weight_mut_rate": 0.8,
    "weight_perturb_rate": 0.9,
    "weight_perturb_amount": 0.5,
    "weight_max_value": 10.0,
    "weight_min_value": -10.0,
    "bias_mut_rate": 0.2,
    "bias_perturb_rate": 0.9,
    "bias_perturb_amount": 0.5,
    "bias_max_value": 10.0,
    "bias_min_value": -10.0,
    "nodal_mut_rate": 0.03,
    "connection_mut_rate": 0.05,
    "enable_rate": 0.25,
    "crossover_rate": 0.75,
    "interspecies_mating_rate": 0.001,
    "stagnation_threshold": 15,
    "elimination_threshold": 0.5,
}

config = defaults.copy()
