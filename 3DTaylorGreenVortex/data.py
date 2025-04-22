import numpy as np
import pickle, random, torch

def prepare_tensor(data):
    '''Makes tensors out of lists.'''
    
    tensor_data = []
    
    if len(data) == 1:
        return torch.tensor(data)

    for entry in data:
        tensor_data.append(torch.tensor(entry))
        
    return tensor_data

def get_data():
    '''Retrieves data from existing file and prepares for testing.'''

    with open('data/x_test.pkl', 'rb') as f:
        x_test = pickle.load(f)

    with open('data/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    #Training and validation indices
    random_indices = random.sample(range(11)[1:], 3)
    random_indices1 = random.sample(range(11)[1:], 1)
    random_indices.sort()
    random_indices1.sort()

    #Create training, physics and validation sets
    x_train, y_train, x_val, y_val, x_all, y_all = [], [], [], [], [], []

    for ind in random_indices:
        #Gets time slices that are multiples of t=0.25 within domain
        v = np.where(np.all(x_test < (2*np.pi), axis=1) & (x_test[:, 3] == ind * 0.25))[0]
        x_train.extend(x_test[v])
        y_train.extend(y_test[v])

    for ind in random_indices1:
        #Gets time slices that are multiples of t=0.25 within domain
        v = np.where(np.all(x_test < (2*np.pi), axis=1) & (x_test[:, 3] == ind * 0.25))[0]
        x_val.extend(x_test[v])
        y_val.extend(y_test[v])

    for ind in np.arange(1, 11):
        #Gets time slices that are multiples of t=0.25 within domain
        v = np.where(np.all(x_test < (2*np.pi), axis=1) & (x_test[:, 3] == ind * 0.25))[0]
        x_all.extend(x_test[v])
        y_all.extend(y_test[v])
   
    #Physics data is same as training data
    pde_x = x_train

    #Collect dataset x_train, y_train, x_val, y_val, x_test, y_test, pde_x
    data = (np.array(x_train), np.array(y_train), np.array(x_val),
            np.array(y_val), np.array(x_all), np.array(y_all), np.array(pde_x))

    return data

def add_noise(data, noise_level=0):
    '''Adds noise to existing data.'''

    noisy_data = []

    for entry in data:
        noisy_data.append(entry + np.random.normal(0, noise_level, entry.shape))

    return noisy_data