import numpy as np
import pickle
import os
import random
import torch
import secrets

def prepare_tensor(data):
    '''Makes tensors out of lists.'''
    
    if len(data) == 1:
        return torch.tensor(data)
    return [torch.tensor(entry) for entry in data]
    
def create_ic():
    '''Randomly samples observations from initial condition.'''

    input, output = [], []
    
    for _ in range(0, 16000):
        x, y, z, t = np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi), np.random.uniform(0, 2*np.pi), 0
        input.append([x, y, z, t])
        
        u = (
                np.sin(x) * np.cos(y) * np.cos(z)
            )

        v = (
            -1 * np.cos(x) * np.sin(y) * np.cos(z) 
        )
        
        w = 0

        p = 0

        output.append([u, v, w, p])
        
    return [input, output]
    
def create_bc():
    '''Loads boundary condition data from generated files.'''

    x_bc = list(np.load("./data/x_bc.pkl", allow_pickle=True))
    y_bc = list(np.load("./data/y_bc.pkl", allow_pickle=True))
    
    return [x_bc, y_bc]
    
def get_data():
    '''Retrieves data from existing file and prepares for testing.'''
    
    if os.path.isfile("./data/all_data.npy"):
        print("Loaded data.")
        all_data = np.load("./data/all_data.npy", allow_pickle=True)
        return all_data

    with open('data/x_test.pkl', 'rb') as f:
        x_test = pickle.load(f)

    with open('data/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    #Training and validation indices
    random_indices = random.sample(range(len(x_test)), 400000)
    random_indices1 = random_indices[300000:]
    random_indices = random_indices[:300000]
    random_indices.sort()
    random_indices1.sort()

    #Create training, physics and validation sets
    x_train, y_train, x_val, y_val = [], [], [], []

    for x in random_indices:
        x_train.append(x_test[x])
        y_train.append(y_test[x])

    for x in random_indices1:
        x_val.append(x_test[x])
        y_val.append(y_test[x])

    for x in sorted(random_indices + random_indices1, reverse=True):
        del x_test[x]
        del y_test[x]
    
    x_test, y_test = np.array(x_test), np.array(y_test)
    sorted_indices = np.argsort(x_test[:, 3])
    x_test = x_test[sorted_indices]
    y_test = y_test[sorted_indices]
   
    #Physics data is same as training data
    pde_x = x_train
    
    ic = create_ic()
    bc = create_bc()

    #Collect dataset x_train, y_train, x_val, y_val, x_test, y_test, pde_x, ic, bc
    data = np.array([np.array(x_train), np.array(y_train), np.array(x_val),
            np.array(y_val), np.array(x_test), np.array(y_test), np.array(pde_x), np.array(ic), np.array(bc)], dtype=object)
            
    np.save("./data/all_data.npy", data)

    return data

def add_noise(data, noise_level=0):
    """Adds noise to data."""

    return [entry + np.random.normal(0, noise_level, entry.shape) for entry in data]
