import os
import numpy as np
from sklearn.metrics import root_mean_squared_error
from traditional_optimizer import Optimizer
from data import add_noise
from bayes_opt import BayesianOptimization

"""How many times to run each experiment."""
samples = 30


def single_experiment(l2_lambda, parameter_optimizer):
    '''
    Runs a single experiment and returns validation error.
    Helper function for finding L2 lambda.
    '''

    print("l2_lambda", l2_lambda)
    parameter_optimizer.l2_lambda = l2_lambda
    result = parameter_optimizer.run()
    parameter_optimizer.viscosity = result["x"][0]

    return -parameter_optimizer.validation() 
    

def traditional_experiment(data, noise, verbose=True, rerun=False, lambdas=[0,0,0,0,0,0,0,0,0]):
    '''Runs the full baseline experiments.'''

    # Skips experiment if results are already saved
    if os.path.isfile("./results/traditional_results.npy") and not rerun:
        print("Loaded traditional results.")
        all_data = np.load("./results/traditional_results.npy", allow_pickle=True)
        return all_data
    
    # Save results from all samples
    rmse = []
    estimated_parameter = []
    parameter_error = []

    # Repeat experiments for every noise level and save the results
    for i, noise_level in enumerate(noise):
        noise_estimated_parameter = []
        noise_rmse = []
        noise_parameter_error = []

        # Runs each experiment multiple times
        for sample in range(samples):

            # Add noise to data
            x_test, y_test, x_train, y_train, _, _, _, _, x_val, y_val, _, indexes = data
            initial_condition = -1 * np.sin(np.linspace(-1, 1, 256) * np.pi)
            y_train_noise, y_val_noise = add_noise([np.array(y_train), np.array(y_val)], noise_level=noise_level)

            #Create optimizer
            parameter_optimizer = Optimizer(
                [y_test, y_train_noise, y_val_noise], 
                indexes, 
                initial_condition,
            )
            
            # If L2 lambda is not provided, Bayesian search calculates the best L2 lambda instead
            if len(lambdas) == 0:

                # Bayesian optimization setting
                pbounds = {'x': (0, 20000)}

                bayesian_optimizer = BayesianOptimization(
                    f=lambda x: single_experiment(x, parameter_optimizer),
                    pbounds=pbounds,
                    random_state=1,
                )

                bayesian_optimizer.maximize(
                    init_points=5,
                    n_iter=10,
                )

                best_l2_lambda = bayesian_optimizer.max['params']['x']
                parameter_optimizer.l2_lambda = best_l2_lambda
                parameter_optimizer.viscosity = parameter_optimizer.run()["x"][0]
            else:
                # Just runs experiment to solve inverse problem using provided L2 lambda
                parameter_optimizer.l2_lambda = lambdas[i]
                parameter_optimizer.viscosity = parameter_optimizer.run()["x"][0]

            # Get results on test set and save
            best_viscosity = parameter_optimizer.viscosity
            rms = parameter_optimizer.test()
            noise_rmse.append(rms)
            noise_estimated_parameter.append(best_viscosity)
            noise_parameter_error.append(root_mean_squared_error([best_viscosity], [0.01 / np.pi]))

            # Outputs statistics while running
            print("Sample: ", str(sample + 1), " out of ", str(samples))
            print("Noise level:" + str(noise_level))
            print("Estimated parameter:" + str(noise_estimated_parameter[-1]))
            print("Test set, RMSE: " + str(noise_rmse[-1]))
            
            if (sample == samples - 1) or noise_level == 0:
                # After the last sample, we have to save everything
                rmse.append(noise_rmse)
                estimated_parameter.append(noise_estimated_parameter)
                parameter_error.append(noise_parameter_error)
                break #For noise_level = 0

    all_results = [np.array(rmse, dtype=object),np.array(estimated_parameter, dtype=object), np.array(parameter_error, dtype=object)]
    np.save("./results/traditional_results.npy", all_results)
    print("Traditional test complete.")

    return all_results
