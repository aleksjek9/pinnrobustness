import os
import random
import numpy as np
import torch


def prepare_tensor(data):
    """Makes tensors out of lists."""

    if len(data) == 1:
        return torch.tensor(data)
    return [torch.tensor(entry) for entry in data]


def add_noise(data, noise_level=0):
    """Adds noise to data."""

    rng = np.random.default_rng()
    return [entry + rng.normal(0, noise_level, entry.shape) for entry in data]

def log_SNR(output):

    std_levels = [0, 0.5, 1, 2, 3, 5, 7, 10, 25]
    y = np.array(output).flatten()
    signal_variance = np.var(y)
    levels = []

    for sigma in std_levels:
        if sigma == 0:
            levels.append(f"σ = {sigma},  SNR = ∞ (no noise)")
            continue
        
        noise_variance = sigma ** 2
        snr = signal_variance / noise_variance
        db = 10 * np.log10(snr)
        
        levels.append(f"σ = {sigma:<5}, SNR = {snr:.4f}, SNR(dB) = {db:.2f}")

    with open("snr.txt", "w") as f:
        for line in levels:
            f.write(line + "\n")
        

def load_data():
    """Loads Burgers' data from:
    https://github.com/lululxvi/deepxde/tree/master."""

    input, output = [], []          

    data = np.load("./data/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T

    for i in range(len(t)):
        if t[i][0] == 0:
            continue

        for j in range(len(x)):
            input.append(np.array([x[j][0], t[i][0]]))
            output.append(np.array([exact[i][j]]))

    log_SNR(output)

    return input, output


def create_training_data(x_test, y_test):
    """Takes 10 time slices to create training data
    and 2 time slices to create validation data."""

    random_indices = random.sample(range(99)[1:], 12)
    slice_size=256
    x_train, y_train, x_val, y_val = [], [], [], []

    for ind in random_indices[:-2]:
        x_train.extend(x_test[ind * slice_size : ind * slice_size + slice_size])
        y_train.extend(y_test[ind * slice_size : ind * slice_size + slice_size])

    for ind in random_indices[-2:]:
        x_val.extend(x_test[ind * slice_size : ind * slice_size + slice_size])
        y_val.extend(y_test[ind * slice_size : ind * slice_size + slice_size])

    for ind in sorted(random_indices, reverse=True):
        del x_test[ind * slice_size : (ind + 1) * slice_size]
        del y_test[ind * slice_size : (ind + 1) * slice_size]

    return x_test, y_test, x_train, y_train, x_val, y_val, random_indices


def create_bc_data():
    """Creates data for training boundary condition."""

    x_bc, y_bc = [], []

    for i in range(80):
        x_bc.append(np.array([np.random.choice([-1, 1]), np.random.uniform(0, 1)]))
        y_bc.append(np.array([0]))

    return x_bc, y_bc


def create_ic_data():
    """Creates data for training the initial condition."""

    x_ic, y_ic = [], []

    for i in range(160):
        x = np.random.uniform(-1, 1)
        x_ic.append(np.array([x, 0]))
        y_ic.append(np.array([-np.sin(x*np.pi)]))  

    return x_ic, y_ic


def create_pde_data():
    """Creates random data for the physics training loss."""

    pde_x = []

    for i in range(2540):
        pde_x.append(np.array([np.random.uniform(-1, 1), np.random.uniform(0, 1)]))

    return pde_x


def get_data():
    """Gets all the data and packs it for use."""

    if os.path.isfile("./data/all_data.npy"):
        print("Loaded data.")
        all_data = np.load("./data/all_data.npy", allow_pickle=True)
        return all_data
    
    x_test, y_test = load_data()
    x_test, y_test, x_train, y_train, x_val, y_val, random_indices = create_training_data(x_test, y_test)
    x_bc, y_bc = create_bc_data()
    x_ic, y_ic = create_ic_data()
    pde_x = create_pde_data()

    all_data = np.array(
        [x_test, y_test, x_train, y_train, x_bc, y_bc, x_ic, y_ic, x_val, y_val, pde_x, random_indices],
        dtype=object
    )

    np.save("./data/all_data.npy", all_data)

    print("Created data.")
    
    return all_data
