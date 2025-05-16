import matplotlib.pyplot as plt
import numpy as np
import statistics

def graph_data(result):
    """Graphs results from the experiments."""

    # Seperates PINN and baseline results
    pinn_results = result[0]
    traditional_results = result[1]

    # Collects the different types of saved results
    result_types = ["rmse", "parameter", "parameter_error", "fem_error"]
    noise_levels = [0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1]
    pinn_means = {}
    pinn_stdevs = {}

    #Type of result -> Noise level -> Sample
    for i, result_type in enumerate(pinn_results):
        pinn_means[result_types[i]] = []
        pinn_stdevs[result_types[i]] = []
        for j, noise_level in enumerate(result_type):
            mean = statistics.mean(noise_level)
            std = 0 if len(noise_level) == 1 else statistics.stdev(noise_level)
            pinn_means[result_types[i]].append(mean)
            pinn_stdevs[result_types[i]].append(std)

    trad_means = {}
    trad_stdevs = {}

    #Type of result -> Noise level -> Sample
    for i, result_type in enumerate(traditional_results):
        trad_means[result_types[i]] = []
        trad_stdevs[result_types[i]] = []

        for j, noise_level in enumerate(result_type):
            mean = statistics.mean(noise_level)
            std = 0 if len(noise_level) == 1 else statistics.stdev(noise_level)
            trad_means[result_types[i]].append(mean)
            trad_stdevs[result_types[i]].append(std)

    # Prediction accuracy plot
    plt.figure()
    plt.plot(noise_levels, pinn_means["rmse"], label="PINN", color="blue")
    plt.scatter(noise_levels, pinn_means["rmse"], color="blue")
    plt.errorbar(
                noise_levels, pinn_means["rmse"], 
                yerr=pinn_stdevs["rmse"], fmt='o', 
                color="black",capsize=5
    )

    plt.plot(noise_levels, trad_means["rmse"], label="FEM/SLSQP", color="violet")
    plt.scatter(noise_levels, trad_means["rmse"], color="violet")
    plt.errorbar(noise_levels, trad_means["rmse"], yerr=trad_stdevs["rmse"], fmt='o', color="black",capsize=5)
    
    
    
    
    plt.plot(noise_levels, pinn_means["fem_error"], label="PINN/FEM", color="orange")
    plt.scatter(noise_levels, pinn_means["fem_error"], color="orange")
    plt.errorbar(
                noise_levels, pinn_means["fem_error"], 
                yerr=pinn_stdevs["fem_error"], fmt='o', 
                color="black",capsize=5
    )

    plt.xlabel('Sigma')
    plt.ylabel('RMSE')
    # plt.ylim(-0.2, 0.2)
    plt.title("2D Taylor-Green Vortex, prediction accuracy")
    plt.legend()
    plt.savefig('./plots/prediction_accuracy.png')

    # Parameter identification plot
    plt.figure()
    plt.plot(noise_levels, pinn_means["parameter"], label="PINN", color="blue")
    plt.scatter(noise_levels, pinn_means["parameter"], color="blue")
    plt.errorbar(
                noise_levels, pinn_means["parameter"], 
                yerr=pinn_stdevs["parameter"], fmt='o', 
                color="black",capsize=5
    )

    plt.plot(noise_levels, trad_means["parameter"], label="FEM/SLSQP", color="violet")
    plt.scatter(noise_levels, trad_means["parameter"], color="violet")
    plt.errorbar(noise_levels, trad_means["parameter"], yerr=trad_stdevs["rmse"], fmt='o', color="black",capsize=5)

    plt.plot(noise_levels, [0.1] * len(noise_levels), label="Ground truth", color="green")

    plt.xlabel('Sigma')
    plt.ylabel('Parameter') 
    plt.title("2D Taylor-Green Vortex, parameter discovered")
    # plt.ylim(-0.05, 0.05)
    plt.legend()
    plt.savefig('./plots/parameter_identification_accuracy.png')

    print("Made plots from results.")
