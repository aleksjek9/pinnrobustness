# pinnrobustness

To run the FEM code, it is probably easiest to create a Docker container using the files provided in the Reproducibility folder.
Alternatively, you can try to create a container from `https://github.com/scientificcomputing/packages/pkgs/container/fenics-gmsh`.
After you have to install PyTorch, Sklearn, Bayesian hyperoptimizer and an older version of NumPy like "`pip install 'numpy<2'`".
Finally, you can install fenics-adjoint using "`python3 -m pip install git+https://github.com/dolfin-adjoint/dolfin-adjoint.git@main`".
Please contact aleksandra.jekic@ntnu.no if you have any questions about or problems with the code in this repository.
