# NEAT
## Algorithm
This is a Python implementation of the **NeuroEvolution of Augmenting Topologies (NEAT)** algorithm described [here](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf).

**NEAT** is an evolutionary algorithm for reinforcement learning based on evolving neural networks. It provides a set of mutations that change connection weights and network topology, and preserves new mutations by dividing networks into species that are protected for a number of generations.

I stay  mostly faithful to the original paper. The one major change that I made (that stuck) is how I implement biases. The authors of the originial paper simply added an input pin with a value of 1 to their networks. I, on the other hand, give each node its own bias value.   

## Installation
```
pip install berens-neat
```

