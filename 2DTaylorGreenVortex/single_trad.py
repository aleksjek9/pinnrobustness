import os
import numpy as np
from traditional_optimizer import Optimizer
from data import get_data, add_noise
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
noise_level = args.noise
seed = args.seed

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
passed = False

def single_run():

    set_seed(seed)
    data = get_data()

    # Add noise to data
    x_test, y_test, x_train, y_train, x_val, y_val, pde_x, _, _ = data

    if rank == 0:
        y_train_noise, y_val_noise = np.array(y_train), np.array(y_val)
        y_train_noise, y_val_noise = add_noise([y_train_noise, y_val_noise], noise_level=noise_level)
    else:
        y_train_noise, y_val_noise = None, None

    y_train_noise = comm.bcast(y_train_noise, root=0)  # Broadcast noise to other ranks
    y_val_noise = comm.bcast(y_val_noise, root=0)      # Broadcast noise to other ranks

    parameter_optimizer = Optimizer([x_test, y_test, x_train, y_train_noise, y_val_noise, x_val])

    parameter_optimizer.l2_lambda = 0

    clock_start = time.time()
    cpu_start = time.process_time()

    parameter_optimizer.viscosity = parameter_optimizer.run()["x"]

    cpu_end = time.process_time()
    clock_end = time.time()
    cpu_time = cpu_end - cpu_start
    clock_time = clock_end - clock_start
    best_viscosity = parameter_optimizer.viscosity[0]

    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss_kb = usage.ru_maxrss
    max_rss_mb = max_rss_kb / 1024

    # Get results on test set and save
    best_viscosity = parameter_optimizer.viscosity[0]
    rms = parameter_optimizer.test()

    all_data = comm.gather((cpu_time, clock_time, max_rss_mb), root=0)

    if rank == 0:
        cpu_times = [x[0] for x in all_data]
        clock_times = [x[1] for x in all_data]
        max_mems = [x[2] for x in all_data]
    
        total_cpu_time = sum(cpu_times)
        max_memory = max(max_mems)
        avg_memory = sum(max_mems)/len(max_mems)
        max_clock_time = max(clock_times)
        
        np.save("trad_results.npy", [best_viscosity, rms])
        np.save("timings.npy", [max_clock_time, total_cpu_time, max_memory, avg_memory])

while not passed:
    try:
        single_run()
        local_passed = True
    except Exception as e:
        #print(f"Rank {rank} failed:", e)
        traceback.print_exc()
        sys.stderr.flush() 
        local_passed = False
        print(noise_level, seed)

    # Check if ALL ranks succeeded
    passed = comm.allreduce(local_passed, op=MPI.LAND)