import os
import numpy as np
from data import get_data, add_noise, prepare_tensor
from fem import tgv_vortex
from sklearn.metrics import root_mean_squared_error 
import time, resource, argparse, torch, random, traceback, sys
from mpi4py import MPI

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--noise", type=float, default=0.1, help="Noise level")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()
viscosity = args.noise
seed = args.seed

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
passed = False

def single_run():

    set_seed(seed)
    data = get_data()
    x_train, y_train, x_val, y_val, x_test, y_test, pde_x, ic, bc = data
    y_test = y_test[~np.isclose(x_test[:, 3], 0.0), 0:3]
    x_test = x_test[x_test[:, 3] != 0.0]

    fem_result = torch.stack(prepare_tensor(tgv_vortex([viscosity], pinn=np.array(x_test))))
    error = root_mean_squared_error(np.array(fem_result)[:, 0:3], np.array(y_test))

    np.save("pinn_fem_results.npy", [error])

while not passed:
    try:
        single_run()
        local_passed = True
    except Exception as e:
        #print(f"Rank {rank} failed:", e)
        traceback.print_exc()
        sys.stderr.flush() 
        local_passed = False

    # Check if ALL ranks succeeded
    passed = comm.allreduce(local_passed, op=MPI.LAND)