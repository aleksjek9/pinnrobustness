# pinnrobustness

To run the FEM code, it is easiest to create a Docker container from `https://github.com/scientificcomputing/packages/pkgs/container/fenics-gmsh`.
After you have to install PyTorch, Sklearn, Bayesian hyperoptimizer and an older version of NumPy like "`pip install 'numpy<2'`".
Finally, you can install fenics-adjoint using "`python3 -m pip install git+https://github.com/dolfin-adjoint/dolfin-adjoint.git@main`".
Please contact aleksandra.jekic@ntnu.no if you have problems running the code or something does not make sense.
