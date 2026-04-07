import os
import torch
import numpy as np
from modules import Model
from data import get_data, prepare_tensor, add_noise
import time, resource, argparse, random, traceback

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
passed = False

def single_run():
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = get_data()
    data = prepare_tensor(data)

    x_train, y_train, x_val, y_val, x_test, y_test, pde_x, ic, bc = data
    y_train_noise, y_val_noise = add_noise([y_train, y_val], noise_level=noise_level)

    x_train = x_train.to(device)
    y_train_noise = y_train_noise.to(device)
    x_val = x_val.to(device)
    y_val_noise = y_val_noise.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    pde_x = pde_x.to(device)
    ic = ic.to(device)
    bc = bc.to(device)

    y_test = y_test[~np.isclose(x_test[:, 3].detach().cpu().numpy(), 0.0), 0:3]
    x_test = x_test[x_test[:, 3] != 0.0]

    clock_start = time.time()
    cpu_start = time.process_time()

    # Train model
    PINN = Model(name=str(noise_level))
    PINN.to(device)
    PINN.train_model(
            [x_train, y_train_noise], [x_val, y_val_noise], 
            pde_x, iterations=200000, tests=[x_test, y_test], icbc=[bc, ic]
    )

    cpu_end = time.process_time()
    clock_end = time.time()
    cpu_time = cpu_end - cpu_start
    clock_time = clock_end - clock_start

    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss_kb = usage.ru_maxrss
    max_rss_mb = max_rss_kb / 1024

    torch.save(PINN.state_dict(), "model.pth")
    np.save("timings_PINN.npy", [clock_time, cpu_time, max_rss_mb])

while not passed:
    try:
        single_run()
        passed = True
    except Exception as e:
        print("Test failed:", e)
        print(noise_level, seed)
        traceback.print_exc()