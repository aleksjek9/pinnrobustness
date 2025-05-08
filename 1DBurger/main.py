from data import get_data
from pinn import PINN_experiment
from graphing import graph_data
from traditional import traditional_experiment


def main():
    """
    Get the data for the experiments and define the
    noise levels for the PINN experiment.
    """
    
    data = get_data()
    noise_levels = [0, 0.5, 1, 2, 3, 5, 7, 10, 25]
    results = []

    # Run the PINN and baseline experiments
    results.append(PINN_experiment(data, noise_levels))
    results.append(traditional_experiment(data, noise_levels))

    # Graph the results
    graph_data(results)


if __name__ == "__main__":
    main()
