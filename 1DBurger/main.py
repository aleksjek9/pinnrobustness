from data import get_data
from pinn import PINN_experiment
from graphing import graph_data
from traditional import traditional_experiment

#Get the data for the experiments
data = get_data()
noise = [0, 0.5, 1, 2, 3, 5, 7, 10, 25]
results = []

#Run the PINN and baseline experiments
#results.append(PINN_experiment(data, noise))
results.append(traditional_experiment(data, noise))

#Graph the results
graph_data(results)
