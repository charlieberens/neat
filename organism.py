import random
from utils import clamp
from copy import deepcopy
from itertools import count


class Organism:
    def __init__(
        self, organism_id: str, innovation_number_counter: count, config: dict
    ):
        self.id: str = organism_id
        self.config: dict = config
        self.node_count: count = count(0)
        self.innovation_number_counter: count = innovation_number_counter
        self.species = None

    def copy(self, organism_id: str):
        """
        Copy the organism
        """
        new_organism = Organism(
            organism_id, self.innovation_number_counter, self.config
        )
        new_organism.nodes = [n.copy() for n in self.nodes]
        new_organism.connections = [c.copy(new_organism) for c in self.connections]
        new_organism.node_count = deepcopy(self.node_count)
        return new_organism

    def create_node(self):
        """
        Add a node to the organism
        """
        split_connection = random.choice(self.connections)
        split_connection.enabled = False
        new_node = Node(0, next(self.node_count), self.config)
        self.nodes.append(new_node)
        connection_1 = Connection(
            self,
            split_connection.in_node.index,
            new_node.index,
            1,
            next(self.innovation_number_counter),
            self.config,
        )
        connection_2 = Connection(
            self,
            new_node.index,
            split_connection.out_node.index,
            split_connection.weight,
            next(self.innovation_number_counter),
            self.config,
        )
        self.connections.append(connection_1)
        self.connections.append(connection_2)

    def create_connection(self):
        """
        Add a connection to the network
        """
        # Select two random nodes
        in_node = random.choice(self.nodes)
        out_node = random.choice(self.nodes)

        # Check if the out_node is an input_node
        if out_node.index < self.config["input_nodes"]:
            return

        # Check if the connection starts with an output node
        if in_node.index in range(
            self.config["input_nodes"],
            self.config["input_nodes"] + self.config["output_nodes"],
        ):
            return

        # Check if the connection already exists
        for connection in self.connections:
            if connection.in_node == in_node and connection.out_node == out_node:
                return

        # Check if the connection creates a cycle
        if self.creates_cycle(
            Connection(self, in_node.index, out_node.index, 1, 0, self.config)
        ):
            return

        # Create the connection
        self.connections.append(
            Connection(
                self,
                in_node.index,
                out_node.index,
                random.uniform(
                    self.config["weight_min_value"], self.config["weight_max_value"]
                ),
                next(self.innovation_number_counter),
                self.config,
            )
        )

    def mutate(self):
        """
        Mutate the organism
        """
        # Mutate weights
        if random.random() < self.config["weight_mut_rate"]:
            connection = random.choice(self.connections)
            if random.random() < self.config["weight_perturb_rate"]:
                connection.perturb_weight()
            else:
                connection.randomize_weight()
        # Mutate biases
        if random.random() < self.config["bias_mut_rate"]:
            node = random.choice(self.nodes)
            if random.random() < self.config["weight_perturb_rate"]:
                node.perturb_bias()
            else:
                node.randomize_bias()
        # Mutate nodes
        if self.config["single_structural_mutation"]:
            r = random.random()
            if r < self.config["node_mut_rate"]:
                self.create_node()
            elif r < self.config["node_mut_rate"] + self.config["connection_mut_rate"]:
                self.create_connection()
        else:
            if random.random() < self.config["node_mut_rate"]:
                self.create_node()
            if random.random() < self.config["connection_mut_rate"]:
                self.create_connection()

    def get_input_layer(self):
        """
        Return the input layer
        """
        return self.nodes[0 : self.config["input_nodes"]]

    def creates_cycle(self, connection):
        """
        Check if the connection creates a cycle
        """
        if connection.in_node == connection.out_node:
            return True

        visited = set()
        visited.add(connection.in_node)
        while True:
            num_added = 0
            for connection in self.connections:
                if connection.out_node in visited:
                    return True
                if connection.in_node in visited:
                    visited.add(connection.out_node)
                    num_added += 1

            if num_added == 0:
                # No new nodes were added this cycle
                return False

    def calculate_layers(self):
        """
        Take the graph of nodes and connections and calculate a possible set of layers
        """

        layers = self.get_input_layer()
        s = set(layers)
        # print("s", s)
        while True:
            # Get all next nodes
            c = (
                set(
                    [
                        connection.out_node
                        for connection in self.connections
                        if connection.in_node in s and connection.out_node not in s
                    ]
                )
                - s
            )
            # print([connection.out_node for connection in self.connections])
            # print([connection.in_node for connection in self.connections])

            # Keep only the nodes that have all their inputs in s
            t = set()
            for node in c:
                if all(
                    connection.in_node in s
                    for connection in self.connections
                    if connection.out_node == node
                ):
                    t.add(node)

            if not t:
                break

            layers.append(t)
            s = s.union(t)
        return list(s)

    def evaluate(self, inputs):
        """
        Takes inputs [-1, -2, -3, ... -n] and returns outputs [0, 1, 2, ... m-1]
        Where n is the number of input nodes and m is the number of output nodes
        """
        layers = self.calculate_layers()

        if len(inputs) != self.config["input_nodes"]:
            raise ValueError(f"Expected {len(layers[0])} inputs, got {len(inputs)}")

        for i, input_node in enumerate(layers[0 : self.config["input_nodes"]]):
            input_node.value = inputs[i]

        for node in self.nodes:
            node.value += node.bias

        for connection in self.connections:
            if connection.enabled:
                connection.out_node.value += (
                    self.config["transfer_function"](connection.in_node.value)
                    * connection.weight
                )

        output = [
            self.config["transfer_function"](output_node.value)
            for output_node in self.nodes[
                self.config["input_nodes"] : self.config["input_nodes"]
                + self.config["output_nodes"]
            ]
        ]

        for node in self.nodes:
            node.value = 0

        return output


class BaseOrganism(Organism):
    """
    self.nodes is an array of nodes
    """

    def __init__(
        self, organism_id: str, innovation_number_counter: count, config: dict
    ):
        super().__init__(organism_id, innovation_number_counter, config)
        self.nodes = [
            InputNode(
                -1 - i,
                next(self.node_count),
                random.uniform(
                    self.config["bias_min_value"], self.config["bias_max_value"]
                ),
                config,
            )
            for i in range(self.config["input_nodes"])
        ] + [
            OutputNode(
                i,
                next(self.node_count),
                random.uniform(
                    self.config["bias_min_value"], self.config["bias_max_value"]
                ),
                config,
            )
            for i in range(self.config["output_nodes"])
        ]

        self.connections = []

        for input_node_index in range(self.config["input_nodes"]):
            for output_node_index in range(
                self.config["input_nodes"],
                self.config["output_nodes"] + self.config["input_nodes"],
            ):
                self.connections.append(
                    Connection(
                        self,
                        input_node_index,
                        output_node_index,
                        random.uniform(
                            config["weight_min_value"], config["weight_max_value"]
                        ),
                        next(self.innovation_number_counter),
                        config,
                    )
                )


class Node:
    def __init__(self, bias: float, index: int, config: dict):
        self.bias: float = bias
        self.value: float = self.bias
        self.config: dict = config
        self.index: int = index

    def perturb_bias(self):
        self.bias += random.uniform(
            -self.config["bias_perturb_amount"], self.config["bias_perturb_amount"]
        )
        self.bias = clamp(
            self.bias, self.config["bias_min_value"], self.config["bias_max_value"]
        )

    def randomize_bias(self):
        self.bias = random.uniform(
            self.config["bias_min_value"], self.config["bias_max_value"]
        )

    def copy(self):
        return Node(self.bias, self.index, self.config)


class InputNode(Node):
    def __init__(self, pin_index: int, index: int, bias: float, config: dict):
        self.pin_index = pin_index
        super().__init__(bias, index, config)


class OutputNode(Node):
    def __init__(self, pin_index: int, index: int, bias: float, config: dict):
        self.pin_index = pin_index
        super().__init__(bias, index, config)


class Connection:
    def __init__(
        self,
        organism: Organism,
        in_node: int,
        out_node: int,
        weight: float,
        innovation_number: int,
        config: dict,
        enabled: bool = True,
    ):
        self.in_node_number: int = in_node
        self.out_node_number: int = out_node
        self.in_node: Node = organism.nodes[in_node]
        self.out_node: Node = organism.nodes[out_node]
        self.weight: float = weight
        self.enabled: bool = enabled
        self.innovation_number: int = innovation_number
        self.config: dict = config

    def copy(self, new_organism: Organism):
        return Connection(
            new_organism,
            self.in_node_number,
            self.out_node_number,
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
        self.weight = random.uniform(
            self.config["weight_min_value"], self.config["weight_max_value"]
        )
