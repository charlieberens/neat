import datetime
import os
import pickle
import random
import time
from neat.utils import clamp
from copy import deepcopy
from itertools import count


class Organism:
    def __init__(
        self, organism_id: str, innovation_number_tracker, config: dict, population
    ):
        self.id: str = organism_id
        self.config: dict = config
        self.node_count: count = count(0)
        self.innovation_number_tracker = innovation_number_tracker
        self.species = None
        self.population = population

    def from_file(filename: str):
        """
        Load an organism from a file
        """
        organism_dict = pickle.load(open(filename, "rb"))
        return Organism.from_dict(organism_dict)

    def from_dict(organism_dict: dict):
        id = organism_dict["meta"]["id"]
        config = organism_dict["config"]
        organism = Organism(id, None, config, None)

        organism.nodes = [Node.from_dict(n, config) for n in organism_dict["nodes"]]
        organism.connections = [
            Connection.from_dict(c, organism, config)
            for c in organism_dict["connections"]
        ]
        return organism

    def copy(self, organism_id: str):
        """
        Copy the organism
        """
        new_organism = Organism(
            organism_id, self.innovation_number_tracker, self.config, self.population
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
            self.innovation_number_tracker(
                (split_connection.in_node.index, new_node.index)
            ),
            self.config,
        )
        connection_2 = Connection(
            self,
            new_node.index,
            split_connection.out_node.index,
            split_connection.weight,
            self.innovation_number_tracker(
                (
                    new_node.index,
                    split_connection.out_node.index,
                )
            ),
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
                self.innovation_number_tracker(
                    (
                        in_node.index,
                        out_node.index,
                    )
                ),
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

        if random.random() < self.config["enable_mut_rate"]:
            connection = random.choice(self.connections)
            connection.enabled = True

    def get_input_layer(self):
        """
        Return the input layer
        """
        return self.nodes[0 : self.config["input_nodes"]]

    def creates_cycle(self, test_connection):
        """
        Check if the connection creates a cycle
        """
        if test_connection.in_node == test_connection.out_node:
            return True

        visited = set()
        visited.add(test_connection.in_node)
        while True:
            num_added = 0
            for connection in self.connections:
                if connection.in_node in visited and connection.out_node not in visited:
                    if connection.out_node == test_connection.in_node:
                        return True
                    visited.add(connection.out_node)
                    num_added += 1
                    break

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

            s = s.union(t)
        return list(s)
    
    def order_connections(self):
        """
        Order the connections in the network based on node order
        """
        # This is absurdly inefficient, but do I care? No!
        layers = self.calculate_layers()
        ordered_connections = []
        for layer in layers:
            for connection in self.connections:
                if connection.out_node == layer:
                    ordered_connections.append(connection)
        return ordered_connections

    def to_file(self, filename=None):
        """
        filename: str
            filename relative to the organism directory
        Save the essential information to a file.
        """
        readable_time = datetime.datetime.fromtimestamp(time.time()).strftime(
            "%Y-%m-%d"
        )
        filename = filename or "{}_{}_{}".format(
            readable_time, self.species.species_number, self.id
        )
        organism_dictionary = self.to_dict()
        organism_dictionary["config"] = self.config
        path = os.path.join(self.config["organism_directory"], filename)
        if not os.path.exists(self.config["organism_directory"]):
            os.makedirs(self.config["organism_directory"])
        with open(path, "wb") as file:
            pickle.dump(organism_dictionary, file)

    def to_dict(self):
        nodes = [{"bias": n.bias, "index": n.index} for n in self.nodes]
        connections = [
            {
                "weight": c.weight,
                "enabled": c.enabled,
                "in_node_number": c.in_node_number,
                "out_node_number": c.out_node_number,
                "innovation_number": c.innovation_number,
            }
            for c in self.connections
        ]
        meta = {"id": self.id}
        return {"nodes": nodes, "connections": connections, "meta": meta}

    def get_network_structure(self):
        """
            Returns a list of layers and connections which can be quickly evaluated
        """
        #  Ensure that the last layer is the output layer
        layers = [
            node.bias for node in self.layers if node not in self.nodes[self.config["input_nodes"]:self.config["input_nodes"]+self.config["output_nodes"]]
        ] + [node.bias for node in self.nodes[self.config["input_nodes"]:self.config["input_nodes"]+self.config["output_nodes"]]]
        connection_pairs = []
        connection_weights = []
        for connection in self.ordered_connections:
            if connection.enabled:
                # find the layer of the in_node
                in_node = 0
                out_node = 0

                for i,node in enumerate(self.layers):
                    if connection.in_node == node:
                        in_node = i
                    if connection.out_node == node:
                        out_node = i

                connection_pairs.append((in_node, out_node))
                connection_weights.append(connection.weight)
        return layers, connection_pairs, connection_weights
        
    def evaluate(self, inputs):
        """
        Takes inputs [-1, -2, -3, ... -n] and returns outputs [0, 1, 2, ... m-1]
        Where n is the number of input nodes and m is the number of output nodes
        """

        if len(inputs) != self.config["input_nodes"]:
            raise ValueError("Expected {} inputs, got {}".format(self.config["input_nodes"], len(inputs)))
        for i,node in enumerate(self.layers):
            if i < self.config["input_nodes"]:
                node.value = inputs[i] + node.bias
            else:
                node.value = node.bias
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

        return output


class BaseOrganism(Organism):
    """
    self.nodes is an array of nodes
    """

    def __init__(
        self, organism_id: str, innovation_number_tracker, config: dict, population
    ):
        super().__init__(organism_id, innovation_number_tracker, config, population)
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
                        self.innovation_number_tracker(
                            (
                                input_node_index,
                                output_node_index,
                            )
                        ),
                        config,
                    )
                )


class CrossOverOrganism(Organism):
    def __init__(
        self,
        organism_id: str,
        innovation_number_tracker,
        parent1,
        parent2,
        config: dict,
    ):
        super().__init__(
            organism_id, innovation_number_tracker, config, parent1.population
        )

        self.connections = []
        self.parent1 = parent1
        self.parent2 = parent2
        self.generate_connections()

    def generate_connections(self):
        i = 0
        j = 0
        a_shared = []
        b_shared = []
        a_disjoint = []
        b_disjoint = []
        a_excess = []
        b_excess = []

        while True:
            if i > len(self.parent1.connections) - 1:
                b_excess = self.parent2.connections[j:]
                break
            if j > len(self.parent2.connections) - 1:
                a_excess = self.parent1.connections[i:]
                break

            a_gene = self.parent1.connections[i]
            b_gene = self.parent2.connections[j]

            if a_gene.innovation_number == b_gene.innovation_number:
                a_shared.append(a_gene)
                b_shared.append(b_gene)
                i += 1
                j += 1
            else:
                if a_gene.innovation_number < b_gene.innovation_number:
                    a_disjoint.append(a_gene)
                    i += 1
                else:
                    b_disjoint.append(b_gene)
                    j += 1

        # TODO - Mutate Bias
        if self.parent1.fitness > self.parent2.fitness:
            self.nodes = [n.copy() for n in self.parent1.nodes]
        else:
            self.nodes = [n.copy() for n in self.parent2.nodes]

        for a_gene, b_gene in zip(a_shared, b_shared):
            if random.random() < 0.5:
                self.connections.append(a_gene.copy(self))
            else:
                self.connections.append(b_gene.copy(self))

        if self.parent1.fitness > self.parent2.fitness:
            self.connections += [gene.copy(self) for gene in a_disjoint]
            self.connections += [gene.copy(self) for gene in a_excess]
        else:
            self.connections += [gene.copy(self) for gene in b_disjoint]
            self.connections += [gene.copy(self) for gene in b_excess]


class Node:
    def __init__(self, bias: float, index: int, config: dict):
        self.bias: float = bias
        self.value: float = self.bias
        self.config: dict = config
        self.index: int = index

    def from_dict(dictionary, config):
        """
        Creates a node from a dictionary
        """
        return Node(
            dictionary["bias"],
            dictionary["index"],
            config,
        )

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

    def from_dict(dictionary, organism: Organism, config):
        """
        Creates a connection from a dictionary
        """
        return Connection(
            organism,
            dictionary["in_node_number"],
            dictionary["out_node_number"],
            dictionary["weight"],
            dictionary["innovation_number"],
            config,
            dictionary["enabled"],
        )

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
