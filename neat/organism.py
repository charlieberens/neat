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
        self.innovation_number_tracker = innovation_number_tracker
        self.species = None
        self.population = population
        self.connections = []
        self.weights = []
        self.node_count = 0
        self.node_innovation_numbers = []
        self.adjusted_fitness = None
        self.history = []

    def from_file(filename: str):
        """
        Load an organism from a file
        """
        organism_dict = pickle.load(open(filename, "rb"))
        return Organism.from_dict(organism_dict)

    def from_dict(organism_dict: dict):
        # id = organism_dict["meta"]["id"]
        # config = organism_dict["config"]
        # organism = Organism(id, None, config, None)

        # organism.nodes = [Node.from_dict(n, config) for n in organism_dict["nodes"]]
        # organism.connections = [
        #     Connection.from_dict(c, organism, config)
        #     for c in organism_dict["connections"]
        # ]
        return
        return organism

    def __copy__(self):
        """
        Copy the organism
        """
        o = Organism(self.id, self.innovation_number_tracker, self.config, None)
        # Copy all attributes
        o.connections = deepcopy(self.connections)
        o.node_count = self.node_count
        o.node_innovation_numbers = self.node_innovation_numbers[:]
        o.history = self.history[:]
        return o

    def create_connection(self, input_node, output_node, enabled, weight=None):
        """
        Add a connection to the network
        """
        if weight is None:
            weight = random.uniform(self.config["weight_min_value"], self.config["weight_max_value"])

        c = ConnectionGene(input_node, output_node, weight, enabled, self.innovation_number_tracker((self.node_innovation_numbers[input_node], self.node_innovation_numbers[output_node])))
        self.connections.append(c)

    def mut_create_node(self):
        """
        Splits a connection thus creating a new node.
        The original connection is disabled and two new connections are created.
        The first connection has the same weight as the original connection.
        The second connection has a weight of 1.
        """
        split_connection = random.choice(self.connections)
        split_connection.enabled = False

        # TODO - Handle innovation numbers 
        self.node_count += 1
        self.node_innovation_numbers.append(self.innovation_number_tracker((split_connection.input_node, split_connection.output_node), node=True))

        self.create_connection(
            split_connection.input_node,
            self.node_count - 1,
            True,
            split_connection.weight),
        self.create_connection(
            self.node_count - 1,
            split_connection.output_node,
            True,
            1),

        self.history.append(self.node_count - 1)
            
    def mut_create_connection(self):
        """
        Create a new connection between two nodes
        """
        # TODO - Handle innovation numbers
        input_node = random.choice(range(self.node_count - 1))
        output_node = random.choice(range(self.node_count - 1))

        if input_node == output_node:
            return
        
        if output_node in range(self.config["input_nodes"]+1) or input_node in range(self.config["input_nodes"]+1,self.config["input_nodes"]+self.config["output_nodes"]+1):
            return

        # Check if connection already exists
        for c in self.connections:
            if c.input_node == input_node and c.output_node == output_node:
                return

        # Check if connection creates a cycle
        if self.creates_cycle(input_node, output_node):
            return
        
        self.create_connection(input_node, output_node, True)

    def mutate(self):
        """
        Mutate the organism
        """
        # Mutate weights
        if random.random() < self.config["weight_mut_rate"]:
            c = random.choice(self.connections)
            if random.random() < self.config["weight_perturb_rate"]:
                c.weight += random.uniform(-self.config["weight_perturb_amount"], self.config["weight_perturb_amount"])
                c.weight = clamp(c.weight, self.config["weight_min_value"], self.config["weight_max_value"])
            else:
                c.weight = random.uniform(self.config["weight_min_value"], self.config["weight_max_value"])

        # Mutate nodes
        if self.config["single_structural_mutation"]:
            r = random.random()
            if r < self.config["node_mut_rate"]:
                self.mut_create_node()
            elif r < self.config["node_mut_rate"] + self.config["connection_mut_rate"]:
                self.mut_create_connection()
        else:
            if random.random() < self.config["node_mut_rate"]:
                self.mut_create_node()
            if random.random() < self.config["connection_mut_rate"]:
                self.mut_create_connection()

        if random.random() < self.config["enable_mut_rate"]:
            c = random.choice(self.connections)
            c.enabled = not c.enabled

    def get_input_layer(self):
        """
        Return the input layer
        """
        pass

    def creates_cycle(self, input_node, output_node):
        """
        Check if the connection creates a cycle
        """
        v = set([output_node])

        while True:
            t = set()
            for c in self.connections:
                if c.input_node in v and c.output_node not in v:
                    if c.output_node == input_node:
                        return True
                    t.add(c.output_node)

            if len(t) == 0:
                break                
            v = v.union(t)

        return False
        

    def calculate_layers(self):
        """
        Take the graph of nodes and connections and calculate a possible set of layers
        """
        layers = [range(self.config["input_nodes"] + 1)]
        v = set(layers[0])

        while True:
            t = set()
            for c in self.connections:
                if c.output_node not in v:
                    if all([i in v for i in [con.input_node for con in self.connections if con.output_node == c.output_node]]):
                        t.add(c.output_node)
            if len(t) == 0:
                break
            v = v.union(t)
            layers.append(t)
        return layers

    def get_layered_connections(self):
        """
        Order the connections in the network based on node order
        """
        node_layers = self.calculate_layers()
        connection_layers = []
        unused_connections = set(self.connections)

        for l in node_layers:
            layer_connections = []
            for c in unused_connections:
                if c.input_node in l:
                    layer_connections.append(c)
            unused_connections = unused_connections.difference(set(layer_connections))
            connection_layers.append(layer_connections)

        return connection_layers, node_layers

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
        pass

    def get_network_structure(self):
        """
            Returns a list of layers and connections which can be quickly evaluated
        """
        pass
            
    def evaluate(self, inputs):
        """
        Takes inputs [-1, -2, -3, ... -n] and returns outputs [0, 1, 2, ... m-1]
        Where n is the number of input nodes and m is the number of output nodes
        """
        if len(inputs) != self.config["input_nodes"]:
            raise ValueError("Incorrect number of inputs")

        node_values = [0] * self.node_count
        node_values[:self.config["input_nodes"]] = inputs
        node_values[self.config["input_nodes"]] = 1        

        layered_connections, layered_nodes = self.get_layered_connections()

        for connection_layer, node_layer in zip(layered_connections, layered_nodes):
            for n in node_layer:
                node_values[n] = self.config["transfer_function"](node_values[n])
            for c in connection_layer:
                if c.enabled:
                    try:
                        node_values[c.output_node] += node_values[c.input_node] * c.weight
                    except IndexError:
                        print(self.node_count, [(c.input_node, c.output_node) for c in self.connections])
                        print(self.history)
                        raise IndexError("Index error in evaluate")
        
        return [node_values[n] for n in layered_nodes[-1]]
                




class BaseOrganism(Organism):
    """
    """

    def __init__(
        self, organism_id: str, innovation_number_tracker, config: dict, population
    ):
        super().__init__(organism_id, innovation_number_tracker, config, population)
        # We include one extra node for the bias node
        self.node_innovation_numbers = [i for i in range(self.config["input_nodes"] + self.config["output_nodes"] + 1)]

        for i in range(self.config["input_nodes"] + 1):
            for j in range(self.config["output_nodes"]):
                self.create_connection(i, j + self.config["input_nodes"] + 1, True)
        
        self.node_count = self.config["input_nodes"] + self.config["output_nodes"] + 1

class ConnectionGene:
    def __init__(self, input_node: int, output_node: int, weight: float, enabled: bool, innovation_number:int):
        self.input_node = input_node
        self.output_node = output_node
        self.weight = weight
        self.enabled = enabled
        self.innovation_number = innovation_number

    def __copy__(self):
        c = ConnectionGene(
            self.input_node,
            self.output_node,
            self.weight,
            self.enabled,
            self.innovation_number,
        )
        for attr in self.__dict__:
            setattr(c, attr, getattr(self, attr))
        return c
    
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

        # Calculate disjoint and excess genes
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

        if self.parent1.fitness > self.parent2.fitness:
            self.node_count = self.parent1.node_count
            self.node_innovation_numbers = self.parent1.node_innovation_numbers[:]


        else:
            self.node_count = self.parent2.node_count
            self.node_innovation_numbers = self.parent2.node_innovation_numbers[:]

        for a_gene, b_gene in zip(a_shared, b_shared):
            if random.random() < 0.5:
                new_gene = a_gene.__copy__()
            else:
                new_gene = b_gene.__copy__()
            new_gene.enabled = b_gene.enabled or a_gene.enabled
            new_gene.input_node, new_gene.output_node = (a_gene.input_node, a_gene.output_node) if self.parent1.fitness > self.parent2.fitness else (b_gene.input_node, b_gene.output_node)
            self.connections.append(new_gene)

        if self.parent1.fitness > self.parent2.fitness:
            self.connections += [gene.__copy__() for gene in a_disjoint]
            self.connections += [gene.__copy__() for gene in a_excess]
        else:
            self.connections += [gene.__copy__() for gene in b_disjoint]
            self.connections += [gene.__copy__() for gene in b_excess]

        b = max([c.output_node for c in self.connections])
        if b >= self.node_count:
            print("ZOINKS")
            print([(c.input_node, c.output_node) for c in self.connections])
            print([(c.input_node, c.output_node) for c in self.parent1.connections])
            print([(c.input_node, c.output_node) for c in self.parent2.connections])
            print(self.parent1.node_count, self.parent2.node_count)

            print(self.parent1.fitness, self.parent2.fitness)