# NEAT
## Algorithm
This is a Python implementation of the **NeuroEvolution of Augmenting Topologies (NEAT)** algorithm described [here](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf).

**NEAT** is an evolutionary algorithm for reinforcement learning based on evolving neural networks. It provides a set of mutations that change connection weights and network topology, and preserves new mutations by dividing networks into species that are protected for a number of generations.

I stay  mostly faithful to the original paper. The one major change that I made (that stuck) is how I implement biases. The authors of the originial paper simply added an input pin with a value of 1 to their networks. I, on the other hand, give each node its own bias value.   

## Installation
```
pip install berens-neat
```
## Usage
(1) Create a json file in your working directory with config values. Any values specified here will overwrite those in config.py. Create a config object by calling `get_config(<file_name>)`.

(2) Create a population with `Population(<number_of_organisms>, <config_object>)`.

(3) Add reporters by calling `<population>.add_reporter(<reporter_object_1>, <reporter_object_2>,...)`.

(4) Define a function that calculates the fitness for an organism.

(5) Call `<population>.run(<eval_function>, <generation_count>`. This will return the best organism after <generation_count> generations.

### Example XOR
```python
from neat.population import Population
from neat.config import get_config
from neat.reporter import PrintReporter

config = get_config() # We will use the default settings for this example
p = Population(100, config)
p.add_reporter(PrintReporter())

def eval_func(organism):
  inputs = [[0,0],[0,1], [1,0], [1,1]]
  expected_outputs = [0,1,1,0]
  err = 0
  for i in range(4):
    actual_output = organism.evaluate(inputs[i])
    err += abs(actual_output - expected_outputs[i])
  
  return 16 - err*err
  
winner = p.run(eval_func, 100)

for input in [[0,0],[0,1], [1,0], [1,1]]:
  print(p.evaluate(input))
```

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

`layers` is an array representing the nodes of the network. The first m inputs are the input nodes and the last k are the output nodes. Each entry in `layers` is the corresponding node's bias. ____ < Shouldn't it be the connections that are sorted > 

`connection_pairs` is an array of tuples of the form `(in_node, out_node)`, where `in_node` and `out_node` are integers representing the index of the in and out node in the `layers` list.

`connection_weights` is an array of floats representing the weight of the ith connection in `connection_pairs`.

## Reporters
Reporters communicate information about a population.

### `PrintReporter`
`PrintReporter` takes no arguments during initialization. It will print information about the population every generation.

### `StatReporter`
`StatReporter` takes a list of stats, a filename, and a frequency. It will save the value of each stat every `<frequency>` generations to `statistics/<filename>.csv`.

| Stat | Description | 
| --|--|
|"best_fitness" | self explanatory |
|"avg_fitness" | self explanatory |
|"worst_fitness" | self explanatory |

## Saving
You can save an organism to `progress/<filename>` by calling `organism.to_file(<filename>)`. If no filename is provided, a name will be selected automatically. 

You can create an organism from a file with `organism = Organsim.from_file(<file_path>)`. 
