import random
from utils import clamp
from copy import deepcopy


class Organism:
    def __init__(self, organism_id: str, config: dict):
        self.id = organism_id
        self.config = config

    def mutate(self):
        """
        Mutate the organism
        """
        if random.random() < self.config["weight_mut_rate"]:
            for connection in self.connections:
                if random.random() < self.config["weight_perturb_rate"]:
                    connection.perturb_weight()
                else:
                    connection.randomize_weight()
        if random.random() < self.config["bias_mut_rate"]:
            for layer in self.layers:
                for node in layer:
                    if random.random() < self.config["weight_perturb_rate"]:
                        node.perturb_bias()
                    else:
                        node.randomize_bias()

    def copy(self, organism_id: str):
        """
        Copy the organism
        """
        new_organism = Organism(organism_id, self.config)
        new_organism.layers = deepcopy(self.layers)
        new_organism.connections = deepcopy(self.connections)
        return new_organism

    def evaluate(self, inputs):
        """
        Takes inputs [-1, -2, -3, ... -n] and returns outputs [0, 1, 2, ... m-1]
        Where n is the number of input nodes and m is the number of output nodes
        """
        if len(inputs) != len(self.layers[0]):
            raise ValueError(
                f"Expected {len(self.layers[0])} inputs, got {len(inputs)}"
            )

        for i, input_node in enumerate(self.layers[0]):
            input_node.value = inputs[i]

        for connection in self.connections:
            if connection.enabled:
                connection.out_node.value += (
                    self.config["transfer_function"](connection.in_node.value)
                    * connection.weight
                )

        return [
            self.config["transfer_function"](output_node.value)
            for output_node in self.layers[-1]
        ]


class BaseOrganism(Organism):
    """
    self.nodes is an array of layers of nodes
    """

    def __init__(
        self, organism_id: str, input_nodes: int, output_nodes: int, config: dict
    ):
        super().__init__(organism_id, config)
        self.layers = [
            [InputNode(-1 - i, random.random(), config) for i in range(input_nodes)],
            [OutputNode(i, random.random(), config) for i in range(output_nodes)],
        ]
        self.connections = []

        for input_node_index in range(len(self.layers[0])):
            for output_node_index in range(len(self.layers[-1])):
                self.connections.append(
                    Connection(
                        self,
                        (0, input_node_index),
                        (1, output_node_index),
                        random.random(),
                        0,
                        config,
                    )
                )


class Node:
    def __init__(self, bias: float, config: dict):
        self.bias = bias
        self.value = self.bias
        self.config = config

    def perturb_bias(self):
        self.bias += random.uniform(
            -self.config["bias_perturb_amount"], self.config["bias_perturb_amount"]
        )
        self.bias = clamp(
            self.bias, self.config["bias_min_value"], self.config["bias_max_value"]
        )

    def randomize_bias(self):
        self.bias = random.randrange(
            self.config["bias_min_value"], self.config["bias_max_value"]
        )

    def copy(self):
        return Node(self.bias, self.config)


class InputNode(Node):
    def __init__(self, index, bias: float, config: dict):
        self.index = index
        super().__init__(bias, config)


class OutputNode(Node):
    def __init__(self, index, bias: float, config: dict):
        self.index = index
        super().__init__(bias, config)


class Connection:
    def __init__(
        self,
        organism: Organism,
        in_node: tuple,
        out_node: tuple,
        weight: float,
        innovation_number: int,
        config: dict,
        enabled: bool = True,
    ):
        self.in_node_number = in_node
        self.out_node_number = out_node
        self.in_node = organism.layers[in_node[0]][in_node[1]]
        self.out_node = organism.layers[out_node[0]][out_node[1]]
        self.weight = weight
        self.enabled = enabled
        self.innovation_number = innovation_number
        self.config = config

    def copy(self):
        return Connection(
            self.in_node_number[:],
            self.out_node_number[:],
            self.weight,
            self.innovation_number,
            self.config,
            self.enabled,
        )

    def perturb_weight(self):
        self.weight += random.uniform(
            -self.config["weight_perturb_amount"], self.config["weight_perturb_amount"]
        )
        self.weight = clamp(
            self.weight,
            self.config["weight_min_value"],
            self.config["weight_max_value"],
        )

    def randomize_weight(self):
        self.weight = random.randrange(
            self.config["weight_min_value"], self.config["weight_max_value"]
        )
