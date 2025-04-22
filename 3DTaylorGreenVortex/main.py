from data import get_data
from pinn import PINN_experiment
from traditional import traditional_experiment
from graphing import graph_data

#Get the data for the experiments
data = get_data()
noise = [0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1]
results = []

#Run the PINN and baseline experiments
results.append(PINN_experiment(data, noise))
results.append(traditional_experiment(data, noise))

#Graph the results
graph_data(results)