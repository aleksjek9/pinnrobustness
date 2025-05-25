import os
import torch
import numpy as np
from modules import Model
from data import prepare_tensor, add_noise
from fem import burgers_1d
from sklearn.metrics import root_mean_squared_error 

"""
The amount of times to run each experiment
in order to get a standard deviation.
"""
samples = 30
test_set_size = 22272
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def PINN_experiment(data, noise, verbose=True, rerun=False):
    """
    Runs the full PINN experiments.
    rmse := metric to estimate the difference between the actual and predicted solution.
    estimated_parameter := viscosity for the burgers equation.
    parameter_error := the difference betweeen the estimated parameter and the actuall parameter value.
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

    for noise_level in noise:
        noise_rmse = []
        noise_estimated_parameter = []
        noise_parameter_error = []
        noise_fem_error = []

        for sample in range(samples):
            # Add noise to data  
            x_test, y_test, x_train, y_train, x_bc, y_bc, x_ic, y_ic, x_val, y_val, pde_x, random_indices = prepare_tensor(data)
            y_train_noise, y_val_noise = add_noise([y_train, y_val], noise_level=noise_level)

            # Train model
            PINN = Model(name=str(noise_level))
            PINN.to(device)
            PINN.train_model(
                            [x_bc, y_bc], [x_ic, y_ic], 
                            [x_train, y_train_noise], [x_val, y_val_noise], 
                            pde_x, iterations=4000
            )
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

            if sample == samples - 1:
                # Save everything from the last sample 
                rmse.append(noise_rmse)
                estimated_parameter.append(noise_estimated_parameter)
                parameter_error.append(noise_parameter_error)
                fem_error.append(noise_fem_error)

                noise_rmse = []
                noise_estimated_parameter = []
                noise_parameter_error = []
                noise_fem_error = []

    all_results = [rmse, estimated_parameter, parameter_error, fem_error]
    np.save("./results/pinn_results.npy", all_results)
    print("PINN test complete.")

    return all_results
