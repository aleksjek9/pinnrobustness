import os
import torch, sys
import numpy as np
from modules import Model
from data import prepare_tensor, add_noise
from fem import burgers_1d
from sklearn.metrics import root_mean_squared_error
import multiprocessing as mp
import subprocess, time, pickle, random

"""
The amount of times to run each experiment
in order to get a standard deviation.
"""
samples = 30
test_set_size = 22272
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def worker(noise_level, seed):
    subprocess.run([sys.executable, "single_pinn.py", "--noise", str(noise_level), "--seed", str(seed)], check=True)

def PINN_experiment(data, noise, verbose=True, rerun=False):
    """
    Runs the full PINN experiments.
    rmse := metric to estimate the difference between the actual and predicted solution.
    estimated_parameter := viscosity for the burgers equation.
    parameter_error := the difference betweeen the estimated parameter and the actual parameter value.
    fem_error := the difference between the FEM solution and the actual solution.
    -Skips experiment if results are already saved
    """

    if os.path.isfile("./results/pinn_results.npy") and not rerun:
        print("Loaded PINN results.")
        all_data = np.load("./results/pinn_results.npy", allow_pickle=True)
        return all_data

    data = prepare_tensor(data)

    rmse = []
    estimated_parameter = []
    parameter_error = []
    fem_error = []
    stats = []

    for noise_level in noise:
        noise_rmse = []
        noise_estimated_parameter = []
        noise_parameter_error = []
        noise_fem_error = []
        noise_stats = []

        for sample in range(samples):
            # Add noise to data
            x_test, y_test, x_train, y_train, x_bc, y_bc, x_ic, y_ic, x_val, y_val, pde_x, random_indices = prepare_tensor(data)

            seed = random.randint(0, 2**32 - 1)
            mp.set_start_method("spawn", force=True)
            p = mp.Process(target=worker, args=(noise_level, seed))
            p.start()
            p.join()
            clock_time, cpu_time, peak_memory = np.load("timings.npy")
            noise_stats.append([clock_time, cpu_time, peak_memory])

            PINN = Model(name=str(noise_level))
            PINN.to(device)
            PINN.load_state_dict(torch.load("model.pth", map_location=device))
            viscosity = 10 ** PINN.visc.item()

            # Save RMSE on test set
            pred = PINN.forward(x_test.to(device))
            error = root_mean_squared_error(pred.detach().cpu().numpy(), y_test.reshape(test_set_size, 1))
            noise_rmse.append(error)

            # Save RMSE using estimated parameters with FEM
            initial_condition = -1 * np.sin(np.linspace(-1, 1, 256) * np.pi)
            fem_result = prepare_tensor(burgers_1d(viscosity, initial_condition, excluded_indices = random_indices))
            error = root_mean_squared_error(fem_result.reshape(test_set_size, 1), y_test.reshape(test_set_size, 1))
            noise_fem_error.append(error)

            # Save estimated parameter and parameter error
            noise_estimated_parameter.append(viscosity)
            error = root_mean_squared_error([viscosity], [0.01 / np.pi])
            noise_parameter_error.append(error)

            if verbose:
                print("Sample: ", str(sample + 1), " out of ", str(samples))
                print("Noise level:" + str(noise_level))
                print("Estimated parameter:" + str(noise_estimated_parameter[-1]))
                print("Test set, RMSE: " + str(noise_rmse[-1]))
                print("Test set, RMSE with FEM: " + str(noise_fem_error[-1]))
                print(noise)
                print(noise_rmse)
                print(noise_estimated_parameter)
                print(noise_parameter_error)
                print(noise_fem_error)
                print(noise_stats)

            if sample == samples - 1:
                # Save everything from the last sample 
                rmse.append(noise_rmse)
                estimated_parameter.append(noise_estimated_parameter)
                parameter_error.append(noise_parameter_error)
                fem_error.append(noise_fem_error)
                stats.append(noise_stats)

                noise_rmse = []
                noise_estimated_parameter = []
                noise_parameter_error = []
                noise_fem_error = []
                noise_stats = []

    all_results = [np.array(rmse, dtype=object), np.array(estimated_parameter, dtype=object), np.array(parameter_error, dtype=object), np.array(fem_error, dtype=object), np.array(stats, dtype=object)]
    arr = np.empty(len(all_results), dtype=object)
    arr[:] = all_results
    np.save("./results/pinn_results.npy", np.array(arr, dtype=object))
    print("PINN test complete.")

    return all_results
