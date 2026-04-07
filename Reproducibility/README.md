Since we had to use partly outdated Fenics code, we recommend creating an image from the docker files provided to make sure it works.

## 1D Burgers' equation

Simply run main.py from the container provided.

## 2D Taylor-Green Vortex

Since we ran multiple instances of this in parallell to complete the experiments in time, to reproduce
our runs one has to run main.py with different seeds for different steps to repeat the calculations.

Both the seed and step is controlled in main.py. In line 15, you change set_seed(1) to set_seed(2), etc.
The steps run are controlled in line 27. For example, you change "for step in range(0, 30):" to "for step in range(0, 2):".

Seed: 1, Steps: 0, 1, 2.

Seed: 2, Steps: 0, 1.

Seed: 3, Steps: 0, 1, 2.

Seed: 4, Steps: 0, 1, 2.

Seed: 5, Steps: 0, 1, 2, 3, 4.

Seed: 6, Steps: 0, 1.

Seed: 7, Steps: 0, 1.

Seed: 8, Steps: 0, 1.

Seed: 9, Steps: 0, 1.

Total steps: 24.

## 3D Taylor-Green Vortex

Same, except we also ran the FEM and PINN parts in parallell with each their main_pinn.py and main_fem.py.

### main_fem.py

Seed: 1, Steps: 0.

Seed: 2, Steps: 0.

Seed: 3, Steps: 0, 1.

Seed: 4, Steps: 0.

Seed: 5, Steps: 0, 1.

Seed: 6, Steps: 0, 1.

Seed: 7, Steps: 0, 1.

Seed: 8, Steps: 0, 1.

Seed: 9, Steps: 0.

Seed: 10, Steps: 0.

Seed: 12, Steps: 0.

Total steps: 16.

### main_pinn.py

Seed: 1, Steps: 0.

Seed: 2, Steps: 0.

Seed: 3, Steps: 0, 1, 2.

Seed: 4, Steps: 0, 1.

Seed: 9, Steps: 0, 1.

Seed: 10, Steps: 0.

Seed: 11, Steps: 0.

Seed: 12, Steps: 0, 1.

Seed: 13, Steps: 0.

Total steps: 14.

Note that the PINN/FEM calculations were done seperately, in main_fem_pinn.py, collecting all results from main_pinn.py.