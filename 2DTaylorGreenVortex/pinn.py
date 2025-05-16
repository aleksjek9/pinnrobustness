import os
import time
import secrets
import numpy as np
import torch
from data import prepare_tensor,add_noise
from fem import tgv_vortex
from modules import Model, gradient
from sklearn.metrics import root_mean_squared_error 

seed = secrets.randbelow(1_000_000)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

"""
The amount of times to run each experiment
in order to get a standard deviation.
"""
samples = 5
device = "cuda"

def PINN_experiment(data, noise, verbose=True, rerun=False):
    """
    Runs the full PINN experiments.
    -Skips experiment if results are already saved.
    """

    if os.path.isfile("./results/pinn_results.npy") and not rerun:
        print("Loaded PINN results.")
        all_data = np.load("./results/pinn_results.npy", allow_pickle=True)
        return all_data

    data = prepare_tensor(data)
    x_test, y_test, x_train, y_train, x_val, y_val, pde_x = data

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
            x_test, y_test, x_train, y_train, x_val, y_val, pde_x = data
            y_train_noise, y_val_noise = np.array(y_train), np.array(y_val)
            y_train_noise, y_val_noise = add_noise([y_train_noise, y_val_noise], noise_level=noise_level)

            x_train = x_train.to(device)
            y_train_noise = y_train.to(device)
            x_val = x_val.to(device)
            y_val_noise = y_val.to(device)
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            pde_x = pde_x.to(device)
            
            # Train model
            PINN = Model(name=str(noise_level))
            PINN.to(device)
            PINN.train_model(
                            [x_train, y_train_noise], [x_val, y_val_noise], 
                            pde_x, iterations=200000
            )
            viscosity = torch.nn.functional.softplus(PINN.visc).item() + 0.00314159265

            # Save RMSE on test set
            y_test = y_test[~np.isclose(x_test[:, 2].detach().cpu().numpy(), 0.0), 0:2]
            x_test = x_test[x_test[:, 2] != 0]
            x, y, t = x_test[:, 0], x_test[:, 1], x_test[:, 2]
            x.requires_grad, y.requires_grad, t.requires_grad = True, True, True
            x_test = torch.stack((x, y, t), dim=1)

            pred = PINN.forward(x_test)

            u_pred = gradient(pred[:, 0], x, create=False)
            v_pred = -1 *gradient(pred[:, 0], y, create=False)
            # p_pred = pred[:, 1]

            pred = torch.stack((u_pred, v_pred), dim=1)
            pred = pred.detach().cpu()
            y_test = y_test.cpu()

            error = root_mean_squared_error(pred, y_test)
            noise_rmse.append(error)

            # Save RMSE using estimated parameters with FEM
            fem_result = tgv_vortex([viscosity], pinn=x_test)
            error = root_mean_squared_error(np.array(fem_result)[:, 0:2], y_test)
            noise_fem_error.append(error)

            # Save estimated parameter
            noise_estimated_parameter.append(viscosity)

            # Save parameter error
            error = root_mean_squared_error([viscosity], [0.1])
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

            if sample == (samples - 1):
                # At the last sample, we have to save everything
                rmse.append(noise_rmse)
                estimated_parameter.append(noise_estimated_parameter)
                parameter_error.append(noise_parameter_error)
                fem_error.append(noise_fem_error)

        with open('results/updates.txt', 'a') as f:
            f.write(f"{noise}, {rmse}, {estimated_parameter}, {parameter_error}, {fem_error} ")

    all_results = np.array([rmse, estimated_parameter, parameter_error, fem_error], dtype="object")
    np.save("./results/pinn_results.npy", all_results)
    print("PINN test complete.")

    return all_results
