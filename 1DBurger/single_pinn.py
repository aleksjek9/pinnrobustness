import os
import torch
import numpy as np
from modules import Model
from data import get_data, prepare_tensor, add_noise
import subprocess, time, sys, pickle, resource, argparse, random

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
noise_level = args.noise
seed = args.seed

set_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = get_data()
data = prepare_tensor(data)

# Add noise to data  
x_test, y_test, x_train, y_train, x_bc, y_bc, x_ic, y_ic, x_val, y_val, pde_x, random_indices = prepare_tensor(data)
y_train_noise, y_val_noise = add_noise([y_train, y_val], noise_level=noise_level)

clock_start = time.time()
cpu_start = time.process_time()

PINN = Model(name=str(noise_level))
PINN.to(device)
PINN.train_model(
                [x_bc, y_bc], [x_ic, y_ic], 
                [x_train, y_train_noise], [x_val, y_val_noise], 
                pde_x, 4000, [x_test, y_test]
)

cpu_end = time.process_time()
clock_end = time.time()
cpu_time = cpu_end - cpu_start
clock_time = clock_end - clock_start

usage = resource.getrusage(resource.RUSAGE_SELF)
max_rss_kb = usage.ru_maxrss
max_rss_mb = max_rss_kb / 1024

torch.save(PINN.state_dict(), "model.pth")
np.save("timings.npy", [clock_time, cpu_time, max_rss_mb])