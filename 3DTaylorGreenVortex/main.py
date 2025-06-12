from data import get_data
from pinn import PINN_experiment
from traditional import traditional_experiment
from graphing import graph_data

from mpi4py import MPI


def main():
    """
    Get the data for the experiments and define the
    noise levels for the PINN experiment.

    Run the PINN and baseline experiments and graph the results.

    Only rank 0 loads/creates the data to ensure that all cores load
    and share the same dataset.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    data = get_data() if rank == 0 else None
    data = comm.bcast(data, root=0)

    noise_levels = [0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1]
    results = []

    if rank == 0:
        pinn_result = PINN_experiment(data, noise_levels)
    else:
        pinn_result = None
        
    pinn_result = comm.bcast(pinn_result, root=0)

    results.append(pinn_result)
    results.append(traditional_experiment(data, noise_levels))

    graph_data(results) if rank == 0 else None


if __name__ == "__main__":
    main()