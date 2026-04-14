import os
import numpy as np
from traditional_optimizer import Optimizer
from data import prepare_tensor, get_data, add_noise
import subprocess, time, sys, pickle, resource, argparse, random, torch

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
data = get_data()

# Add noise to data
x_test, y_test, x_train, y_train, _, _, _, _, x_val, y_val, _, indexes = data
initial_condition = -1 * np.sin(np.linspace(-1, 1, 256) * np.pi)
y_train_noise, y_val_noise = add_noise([np.array(y_train), np.array(y_val)], noise_level=noise_level)

#Create optimizer
parameter_optimizer = Optimizer(
[y_test, y_train_noise, y_val_noise], 
indexes, 
initial_condition,
)

parameter_optimizer.l2_lambda = 0

clock_start = time.time()
cpu_start = time.process_time()

result = parameter_optimizer.run()
parameter_optimizer.viscosity, parameter_optimizer.advection = result.x

cpu_end = time.process_time()
clock_end = time.time()
cpu_time = cpu_end - cpu_start
clock_time = clock_end - clock_start

usage = resource.getrusage(resource.RUSAGE_SELF)
max_rss_kb = usage.ru_maxrss
max_rss_mb = max_rss_kb / 1024

# Get results on test set and save
best_viscosity = parameter_optimizer.viscosity
best_advection = parameter_optimizer.advection
rms = parameter_optimizer.test()

np.save("trad_results.npy", [best_viscosity, best_advection, rms])
np.save("timings.npy", [clock_time, cpu_time, max_rss_mb])