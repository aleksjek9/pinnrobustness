import os, torch
import numpy as np
from data import prepare_tensor,add_noise
from fem import tgv_vortex
from modules import Model, calc_grad
from sklearn.metrics import root_mean_squared_error 

#The amount of times to run each experiment
#in order to get a standard deviation
samples = 5
device = "cpu"

def PINN_experiment(data, noise, verbose=True, rerun=False):
    #Runs the full PINN experiments
    
    #Skips experiment if results are already saved
    if os.path.isfile("./results/pinn_results.npy") and not rerun:
        print("Loaded PINN results.")
        all_data = np.load("./results/pinn_results.npy", allow_pickle=True)
        return all_data

    data = prepare_tensor(data) 
    x_train, y_train, x_val, y_val, x_test, y_test, pde_x = prepare_tensor(data)

    rmse = []
    estimated_parameter = []
    parameter_error = []
    fem_error = []

    for noise_level in noise:

        noise_rmse = []
        noise_estimated_parameter = []
        noise_parameter_error = []
        noise_fem_error = []

        for sample in range(0, samples):

            #Add noise to data
            x_train, y_train, x_val, y_val, x_test, y_test, pde_x = prepare_tensor(data)
            y_train = torch.stack(add_noise(y_train, noise_level=noise_level))
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            pde_x = pde_x.to(device)
            
            #Train model
            PINN = Model()
            PINN.to(device)
            PINN.train_model([x_train, y_train], [x_test, y_test], pde_x, iterations=1)
            viscosity = PINN.visc.item()

            #Save RMSE on test set
            pred = PINN.forward(x_test)
            pred = pred.detach().cpu()
            y_test = y_test.cpu()

            error = root_mean_squared_error(pred[:, 0:3], y_test[:, 0:3])
            noise_rmse.append(error)

            #Save RMSE using estimated parameters with FEM
            fem_result = prepare_tensor(tgv_vortex([viscosity], pinn=x_test))
            error = root_mean_squared_error(np.array(fem_result)[:, 0:3], y_test[~np.isclose(x_test[:, 3].detach().numpy(), 0.0), 0:3])
            noise_fem_error.append(error)

            #Save estimated parameter
            noise_estimated_parameter.append(viscosity)

            #Save parameter error
            error = root_mean_squared_error([viscosity], [0.01])
            noise_parameter_error.append(error)

            if verbose:
                print("Sample: ", str(sample + 1), " out of ", str(samples))
                print("Noise level:" + str(noise_level))
                print("Estimated parameter:" + str(noise_estimated_parameter[-1]))
                print("Test set, RMSE: " + str(noise_rmse[-1]))
                #print("Test set, RMSE with FEM: " + str(noise_fem_error[-1]))
                print(noise)
                print(noise_rmse)
                print(noise_estimated_parameter)
                print(noise_parameter_error)
                print(noise_fem_error)

            if (sample == (samples - 1)):
                #At the last sample, we have to save everything
                rmse.append(noise_rmse)
                estimated_parameter.append(noise_estimated_parameter)
                parameter_error.append(noise_parameter_error)
                fem_error.append(noise_fem_error)

                noise_rmse = []
                noise_estimated_parameter = []
                noise_parameter_error = []
                noise_fem_error = []

        with open('results/updates.txt', 'a') as f:
            f.write(f"{noise}, {rmse}, {estimated_parameter}, {parameter_error}, {fem_error} ")

        
    all_results = np.array([rmse, estimated_parameter, parameter_error, fem_error], dtype="object")
    np.save("./results/pinn_results.npy", all_results)
    print("PINN test complete.")

    return all_results