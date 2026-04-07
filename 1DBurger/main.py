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

set_seed(10)

from data import get_data
from pinn import PINN_experiment
from graphing import graph_data
from traditional import traditional_experiment

def main():
    """
    Get the data for the experiments and define the
    noise levels for the PINN experiment.
    """
    
    data = get_data()
    noise_levels = [0, 0.5, 1, 2, 3, 5, 7, 10, 25]

    # Run the PINN and baseline experiments
    PINN_experiment(data, noise_levels)
    traditional_experiment(data, noise_levels)


if __name__ == "__main__":
    main()