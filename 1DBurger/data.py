import os, random, torch
import numpy as np

def prepare_tensor(data):
    '''Makes tensors out of lists.'''
    
    tensor_data = []
    
    if len(data) == 1:
        return torch.tensor(data)

    for entry in data:
        tensor_data.append(torch.tensor(entry))
        
    return tensor_data

def add_noise(data, noise_level=0):
    '''Adds noise to data.'''

    noisy_data = []

    for entry in data:
        noisy_data.append(entry + np.random.normal(0, noise_level, entry.shape))

    return noisy_data
        

def load_data():
    '''Loads Burgers' data from https://github.com/lululxvi/deepxde/tree/master.'''

    input, output = [], []

    data = np.load("./data/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T

    for i in range(0, len(t)):
        if t[i][0] == 0:
            continue
        for j in range(0, len(x)):
            input.append(np.array([x[j][0], t[i][0]]))
            output.append(np.array([exact[i][j]]))

    return input, output

def create_training_data(x_test, y_test):
    '''Takes 10 time slices to create training data
    and 2 time slices to create validation data.'''

    random_indices = random.sample(range(100)[1:], 12)

    x_train, y_train, x_val, y_val = [], [], [], []

    for ind in random_indices[:-2]:
        x_train.extend(x_test[ind * 256: ind * 256 + 256])
        y_train.extend(y_test[ind * 256: ind * 256 + 256])

    for ind in random_indices[-2:]:
        x_val.extend(x_test[ind * 256: ind * 256 + 256])
        y_val.extend(y_test[ind * 256: ind * 256 + 256])

    for i in sorted(random_indices, reverse=True):
        del x_test[i]
        del y_test[i]

    return x_test, y_test, x_train, y_train, x_val, y_val, random_indices

def create_bc_data():
    '''Creates data for training boundary condition.'''

    x_bc, y_bc = [], []

    for i in range(0, 80):
        x_bc.append(np.array([np.random.choice([-1, 1]), np.random.uniform(0, 1)]))
        y_bc.append(np.array([0]))

    return x_bc, y_bc

def create_ic_data():
    '''Creates data for training the initial condition.'''

    x_ic, y_ic = [], []

    for i in range(0, 160):
        x = np.random.uniform(-1, 1)
        x_ic.append(np.array([x, 0]))
        y_ic.append(np.array([-1 * np.sin(x*np.pi)]))

    return x_ic, y_ic

def create_pde_data():
    '''Creates random data for the physics training loss.'''

    pde_x = []

    for i in range(0, 2540):
        pde_x.append(np.array([np.random.uniform(-1, 1), np.random.uniform(0, 1)]))

    return pde_x

def get_data():
    '''Gets all the data and packs it for use.'''

    if os.path.isfile("./data/all_data.npy"):
        print("Loaded data.")
        all_data = np.load("./data/all_data.npy", allow_pickle=True)
        return all_data
    
    x_test, y_test = load_data()
    x_test, y_test, x_train, y_train, x_val, y_val, random_indices = create_training_data(x_test, y_test)
    x_bc, y_bc = create_bc_data()
    x_ic, y_ic = create_ic_data()
    pde_x = create_pde_data()

    all_data = np.array([x_test, y_test, x_train, y_train, x_bc, y_bc, x_ic, y_ic, x_val, y_val, pde_x, random_indices], dtype=object)

    np.save("./data/all_data.npy", all_data)

    print("Created data.")
    return all_data
