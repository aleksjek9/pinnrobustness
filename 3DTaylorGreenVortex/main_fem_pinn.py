import numpy as np
import os
import random
from sklearn.metrics import root_mean_squared_error
from fem import tgv_vortex
from data import get_data
import numpy as np
import torch

#This was mpirun with 55 cores for each noise level set

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

noise_level_index_to_run = 0 #Out of 8 noise levels (0-7), which set of parameters to run
fem_errors = [] #Where PINN/FEM results are saved

#Get x_test and y_test data to evaluate error after
data = get_data()
x_test, y_test, x_train, y_train, x_val, y_val, pde_x, _, _ = data
y_test = y_test[~np.isclose(x_test[:, 3], 0.0), 0:3]
x_test = x_test[x_test[:, 3] != 0.0]

#Get parameters for noise level we are testing
pinn = np.load("pinn_parameters.npy")
pinn_parameters = pinn[noise_level_index_to_run] #shape 8, 30
set_seed(noise_level_index_to_run)

#For each parameter, solve with FEM
for y in pinn_parameters:
    fem_result = tgv_vortex([y], pinn=x_test)
    error = root_mean_squared_error(np.array(fem_result)[:, 0:3], y_test[:, 0:3])
    fem_errors.append(error)

#Save the noise level we have tested
np.save("./results/pinn_fem_" + str(noise_level_index_to_run) + ".npy", fem_errors)