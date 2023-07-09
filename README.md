# NEAT

## Algorithm

This is a Python implementation of the **NeuroEvolution of Augmenting Topologies (NEAT)** algorithm described [here](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf).

I stay mostly faithful to the original paper.

## Installation

```
pip install git+https://github.com/charlieberens/neat.git
```

## Usage

(1) Create a json file in your working directory with config values. Any values specified here will overwrite those in config.py. Create a config object by calling `get_config(<file_name>)`.

(2) Create a population with `Population(<number_of_organisms>, <config_object>)`.

(3) Add reporters by calling `<population>.add_reporter(<reporter_object_1>, <reporter_object_2>,...)`.

(4) Define a function that calculates the fitness for an organism.

-   Note: Be sure to scale your input values to (0,1). Not doing so can result in errors or armageddon or something.

(5) Call `<population>.run(<eval_function>, <generation_count>`. This will return the best organism after <generation_count> generations.

## Evalution

### Population Options

#### One at a time

This is default option. The evaluation function you supply takes an organism and returns its fitness.

#### En Masse

You can select this option by calling run with `en_masse=True`. The evaluation function you supply takes a list of organisms and returns a list of fitnesses.

This is useful when used with `get_network_structure` to calculate fitnesses in parallel on the GPU.

### Evaluation Options

#### `organism.evaluate`

`organism.evaluate` takes a list of m inputs (m is specified in the config), and returns k outputs (the size of which is also specified in the config).

#### `organism.get_network_structure`

`organism.get_network_structure` takes no arguments and returns (`layers`, `connection_pairs`, `connection_weights`).

`layers` is an array representing the nodes of the network. The first m inputs are the input nodes and the last k are the output nodes. Each entry in `layers` is the corresponding node's bias.

`connection_pairs` is an array of tuples of the form `(in_node, out_node)`, where `in_node` and `out_node` are integers representing the index of the in and out node in the `layers` list. Connections are ordered so that they can be evaluated in order. (A connection b's in node will have all of its in-connections evaluated before b is evaluated).

`connection_weights` is an array of floats representing the weight of the ith connection in `connection_pairs`.

NOTE: You will have to implement your own transfer function if you use this option.

## Reporters

Reporters communicate information about a population.

### `PrintReporter`

`PrintReporter` takes no arguments during initialization. It will print information about the population every generation.

### `StatReporter`

`StatReporter` takes a list of stats, a filename, and a frequency. It will save the value of each stat every `<frequency>` generations to `statistics/<filename>.csv`.

| Stat            | Description      |
| --------------- | ---------------- |
| "best_fitness"  | self explanatory |
| "avg_fitness"   | self explanatory |
| "worst_fitness" | self explanatory |

## Saving

You can save an organism to `progress/<filename>` by calling `organism.to_file(<filename>)`. If no filename is provided, a name will be selected automatically.

You can create an organism from a file with `organism = Organsim.from_file(<file_path>)`.

## Configuration Parameters

| parameter                  | type     | description                                                                       | default          |
| -------------------------- | -------- | --------------------------------------------------------------------------------- | ---------------- |
| input_nodes                | int      | Number of input nodes                                                             | 2                |
| output_nodes               | int      | Number of output nodes                                                            | 1                |
| transfer_function          | function | Transfer function for the nodes                                                   | modified_sigmoid |
| c1                         | float    | Speciation parameter                                                              | 1.0              |
| c2                         | float    | Speciation parameter                                                              | 1.0              |
| c3                         | float    | Speciation parameter                                                              | 0.4              |
| delta_thresh               | float    | Speciation threshold                                                              | 3.0              |
| weight_mut_rate            | float    | Chance of weight mutation                                                         | .8               |
| weight_perturb_rate        | float    | Chance of perturbing weight                                                       | .9               |
| weight_perturb_amount      | float    | Maximum amount to perturb weight by                                               | 5.0              |
| weight_max_value           | float    | Maximum value of a weight                                                         | 10.0             |
| weight_min_value           | float    | Minimum value of a weight                                                         | -10.0            |
| bias_mut_rate              | float    | Chance of bias mutation                                                           | .8               |
| bias_perturb_rate          | float    | Chance of perturbing bias                                                         | .9               |
| bias_perturb_amount        | float    | Maximum amount to perturb bias by                                                 | 1.0              |
| bias_max_value             | float    | Maximum value of a bias                                                           | 2.0              |
| bias_min_value             | float    | Minimum value of a bias                                                           | -2.0             |
| single_structural_mutation | bool     | Whether to only allow one structural mutation per mutation                        | True             |
| node_mut_rate              | float    | Chance of nodal mutation                                                          | 0.03             |
| connection_mut_rate        | float    | Chance of connection mutation                                                     | 0.05             |
| enable_mut_rate            | float    | Chance of enabling a disabled gene                                                | 0.0              |
| crossover_rate             | float    | Chance of crossover                                                               | 0.75             |
| interspecies_mating_rate   | float    | Chance of interspecies mating                                                     | 0.001            |
| stagnation_threshold       | int      | Number of generations without improvement before a species is considered stagnant | 20               |
| elimination_threshold      | float    | Proportion of organisms to eliminate from a species                               | .25              |
| min_species_size           | int      | Minimum number of organisms in a species                                          | 5                |
| stat_directory             | string   | Directory to save statistics to (used by StatReporter)                            | "statistics"     |
| progress_directory         | string   | Directory to save progress to (used by ProgressReporter)                          | "progress"       |
