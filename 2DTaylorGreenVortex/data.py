import os, random, torch
import numpy as np

#Configurations
wave_number = 1
visc = 0.1
L = 2*np.pi
dt = 0.1
T = 2.5
dx = 0.05
dy = 0.05

def add_noise_pinn(data, noise_level=0):
    '''Adds noise to data.'''

    noisy_data = []

    for entry in data:
        noisy_data.append(entry + np.random.normal(0, noise_level, entry.shape))

    return noisy_data[0]

def add_noise(data, noise_level=0):
    '''Adds noise to data.'''

    noisy_data = []

    for entry in data:
        noisy_data.append(entry + np.random.normal(0, noise_level, entry.shape))

    return noisy_data

def prepare_tensor(data):
    '''Makes tensors out of lists.'''
    
    tensor_data = []
    
    if len(data) == 1:
        return torch.tensor(data)

    for entry in data:
        tensor_data.append(torch.tensor(entry))
        
    return tensor_data

def create_data():
    '''For each time step and point space in domain, creates input and output.'''

    input = []
    output = []

    for t in np.arange(0, T + dt, dt):
        for x in np.linspace(0, L, int(L/dx) + 1):
            for y in np.linspace(0, L, int(L/dy) + 1):

                input.append([x, y, t])

                u = np.exp(-2*visc * wave_number**2 * t) * np.sin(wave_number * x) * np.cos(wave_number * y)
                v = -1 * np.exp(-2*visc * wave_number**2 * t) * np.cos(wave_number * x) * np.sin(wave_number * y)
                p = 0.25 * np.exp(-4 * visc * wave_number**2 * t) * (np.cos(2 * wave_number * x) + np.cos(2 * wave_number * y))

                output.append([u, v, p])

    return input, output

def create_training_data(x_test, y_test):
    '''From full dataset, samples for training and validation set.'''

    #Making sure that selection is without replacement
    indices = random.sample(range(len(x_test)), 6000)

    random_indices = sorted(indices[:5000])
    random_indices1 = sorted(indices[5000:])

    x_train, y_train, x_val, y_val = [], [], [], []

    for ind in random_indices:
        x_train.append(x_test[ind])
        y_train.append(y_test[ind])

    for ind in random_indices1:
        x_val.append(x_test[ind])
        y_val.append(y_test[ind])

    #Removes training and validation data from test set
    for i in sorted(indices, reverse=True):
        del x_test[i]
        del y_test[i]


    return x_train, y_train, x_val, y_val, x_test, y_test

def get_data():
    '''Loads data, if it exists, otherwise creates.'''

    if os.path.isfile("./data/all_data.npy"):
        print("Loaded data.")
        all_data = np.load("./data/all_data.npy", allow_pickle=True)
        return all_data

    x_test, y_test = create_data()
    x_train, y_train, x_val, y_val, x_test, y_test = create_training_data(x_test, y_test)
    pde_x = x_train #Uses same data for physics loss as for data loss

    all_data = np.array([x_test, y_test, x_train, y_train, x_val, y_val, pde_x], dtype=object)

    np.save("./data/all_data.npy", all_data)

    print("Created data.")
    return all_data