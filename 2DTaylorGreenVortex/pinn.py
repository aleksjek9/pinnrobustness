import os
import sys
import numpy as np
import torch
from data import prepare_tensor,add_noise
from fem import tgv_vortex
from modules import Model, gradient
from sklearn.metrics import root_mean_squared_error
import multiprocessing as mp
import subprocess, random


"""
The amount of times to run each experiment
in order to get a standard deviation.
"""
samples = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def worker(noise_level, seed):
    subprocess.run([sys.executable, "single_pinn.py", "--noise", str(noise_level), "--seed", str(seed)], check=True)

def PINN_experiment(data, noise, step, verbose=True, rerun=False):
    """
    Runs the full PINN experiments.
    -Skips experiment if results are already saved.
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
            x_test, y_test, x_train, y_train, x_val, y_val, pde_x, ic, bc = data

            x_test = x_test.to(device)
            y_test = y_test.to(device)
            y_test = y_test[~np.isclose(x_test[:, 2].detach().cpu().numpy(), 0.0), 0:2]
            x_test = x_test[x_test[:, 2] != 0]

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
            viscosity = torch.nn.functional.softplus(PINN.visc).item() + 0.00314159265

            # Save RMSE on test set
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
                print(noise_stats)

            if sample == (samples - 1):
                # At the last sample, we have to save everything
                rmse.append(noise_rmse)
                estimated_parameter.append(noise_estimated_parameter)
                parameter_error.append(noise_parameter_error)
                fem_error.append(noise_fem_error)
                stats.append(noise_stats)

                all_results = [np.array(rmse, dtype=object), np.array(estimated_parameter, dtype=object), np.array(parameter_error, dtype=object), np.array(fem_error, dtype=object), np.array(stats, dtype=object)]
                arr = np.empty(len(all_results), dtype=object)
                arr[:] = all_results
                np.save("./results/pinn_results" + str(step) + "temp_progress.npy", arr)

        with open('results/updates.txt', 'a') as f:
            f.write(f"{noise}, {rmse}, {estimated_parameter}, {parameter_error}, {fem_error}, {stats} ")

    all_results = [np.array(rmse, dtype=object), np.array(estimated_parameter, dtype=object), np.array(parameter_error, dtype=object), np.array(fem_error, dtype=object), np.array(stats, dtype=object)]
    arr = np.empty(len(all_results), dtype=object)
    arr[:] = all_results
    np.save("./results/pinn_results" + str(step) + ".npy", arr)
    print("PINN test complete.")

    return all_results

