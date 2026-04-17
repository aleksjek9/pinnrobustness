import os, sys
import numpy as np
from sklearn.metrics import root_mean_squared_error
import multiprocessing as mp
import subprocess, random

"""How many times to run each experiment."""
samples = 1

def worker(noise_level, seed):
    subprocess.run(
    [
        "mpirun",
        "-n", "10",
        sys.executable,   # <-- use the same Python interpreter
        "single_trad.py",
        "--noise", str(noise_level),
        "--seed", str(seed)
    ],
    check=True
    )

def traditional_experiment(data, noise, step, verbose=True, rerun=False, lambdas=[0,0,0,0,0,0,0,0,0]):
    """Runs the full baseline experiments."""

    # Skips experiment if results are already saved
    if os.path.isfile("./results/traditional_results.npy") and not rerun:
        print("Loaded traditional results.")
        all_data = np.load("./results/traditional_results.npy", allow_pickle=True)
        return all_data
    
    # Holds results from all samples
    rmse = []
    estimated_parameter = []
    parameter_error = []
    stats = []

    # Repeat experiments for every noise level
    for i, noise_level in enumerate(noise):
        # Holds results for all the samples for this noise level
        noise_estimated_parameter = []
        noise_rmse = []
        noise_parameter_error = []
        noise_stats = []

        # Runs each experiment multiple times
        for sample in range(samples):
            
            seed = random.randint(0, 2**32 - 1)
            mp.set_start_method("spawn", force=True)
            p = mp.Process(target=worker, args=(noise_level, seed))
            p.start()
            p.join()
            best_viscosity, rms = np.load("trad_results.npy")
            max_clock_time, total_cpu_time, max_memory, avg_memory = np.load("timings.npy")
            noise_stats.append([max_clock_time, total_cpu_time, max_memory, avg_memory])

            noise_rmse.append(rms)
            noise_estimated_parameter.append(best_viscosity)
            noise_parameter_error.append(root_mean_squared_error([best_viscosity], [0.1]))

            # Outputs statistics while running
            print("Sample: ", str(sample + 1), " out of ", str(samples))
            print("Noise level:" + str(noise_level))
            print("Estimated parameter:" + str(noise_estimated_parameter[-1]))
            print("Test set, RMSE: " + str(noise_rmse[-1]))
            print(noise_stats)
            
            if sample == samples - 1 or noise_level == 0:
                # After the last sample, we have to save everything
                rmse.append(noise_rmse)
                estimated_parameter.append(noise_estimated_parameter)
                parameter_error.append(noise_parameter_error)
                stats.append(noise_stats)
                
                all_results = [np.array(rmse, dtype=object),np.array(estimated_parameter, dtype=object), np.array(parameter_error, dtype=object), np.array(stats, dtype=object)]
                arr = np.empty(len(all_results), dtype=object)
                arr[:] = all_results
                np.save("./results/traditional_results" + str(step) + "temp_progress.npy", arr)
                break #For noise_level = 0

    all_results = [np.array(rmse, dtype=object),np.array(estimated_parameter, dtype=object), np.array(parameter_error, dtype=object), np.array(stats, dtype=object)]
    arr = np.empty(len(all_results), dtype=object)
    arr[:] = all_results
    np.save("./results/traditional_results" + str(step) + ".npy", arr)
    print("Traditional test complete.")

    return all_results