import json
import matplotlib.pyplot as plt

def plot_species(filename: str, min_generation: int = 0, max_generation: int = None):
    with open(filename, "r") as f:
        data = json.load(f)

    species = {}

    first_species = list(data[0].keys())[0]
    datapoints = [datapoint for datapoint in data[0][first_species]]

    for row in data:
        for s in row:
            if s not in species:
                species[s] = {}
                for datapoint in datapoints:
                    species[s][datapoint] = []
    
    for row in data:
        for s in species:
            if s in row:
                for datapoint in datapoints:
                    species[s][datapoint].append(row[s][datapoint])
            else:
                for datapoint in datapoints:
                    species[s][datapoint].append(None)
    

    datapoints = [datapoint for datapoint in datapoints if type(species[first_species][datapoint][0]) == int or type(species[first_species][datapoint][0]) == float]

    for datapoint in [d for d in datapoints]:
        plt.figure()
        plt.title(datapoint)
        plt.xlabel("Generation")
        plt.ylabel("Value")
        plt.ylim(0, max([max([v for v in species[s][datapoint] if v]) for s in species]))
        for s in species:
            plt.plot(species[s][datapoint], label=s)
        plt.legend()
        plt.show()

    
