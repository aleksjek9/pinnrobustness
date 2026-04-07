import os
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1)

from data import get_data
from pinn import PINN_experiment
from traditional import traditional_experiment


def main():
    """
    Get the data for the experiments and define the
    noise levels for the PINN experiment.
    """


    data = get_data()

    for step in range(0, 30):
        data = get_data()
        noise_levels = [0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1]

        #PINN_experiment(data, noise_levels, step)
        traditional_experiment(data, noise_levels, step)


if __name__ == "__main__":
    main()
