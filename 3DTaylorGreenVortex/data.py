import numpy as np
import pickle
import random
import torch
import secrets

def prepare_tensor(data):
    '''Makes tensors out of lists.'''
    
    if len(data) == 1:
        return torch.tensor(data)
    return [torch.tensor(entry) for entry in data]
    
def get_data():
    '''Retrieves data from existing file and prepares for testing.'''

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

    #Collect dataset x_train, y_train, x_val, y_val, x_test, y_test, pde_x
    data = np.array([np.array(x_train), np.array(y_train), np.array(x_val),
            np.array(y_val), np.array(x_test), np.array(y_test), np.array(pde_x)], dtype=object)

    return data

def add_noise(data, noise_level=0):
    """Adds noise to data."""

    return [entry + np.random.normal(0, noise_level, entry.shape) for entry in data]
