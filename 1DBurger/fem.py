#This code is distributed under the GNU LGPL license.

from fenics import *
from fenics_adjoint import *
import numpy as np

def get_prediction_indexes(results, indexes):
    '''From the predictions made by the FEM model,
    get the predictions which we have training data for.'''
    
    pred = []
    
    for index in indexes:
        pred.append(results[0][index])

    return pred

def get_solution_indexes(y_train, indexes, solution_comparison, map):
    '''Get the training data from the full solution.'''

    new_y_train = []
    y_train = np.array(y_train).reshape(10, 256)

    for step in y_train:
        solution_comparison.vector()[:] = step
        solution_comparison.vector()[:] = solution_comparison.vector().get_local(map)
        new_y_train.append(solution_comparison.copy(deepcopy=True))
    
    return new_y_train

def get_gradient(l2_lambda, indexes, results, y_train, control_variable, solution_comparison, map):
    '''Calculate the error and gradient for the training data.'''

    pred = get_prediction_indexes(results, indexes)
    y_train = get_solution_indexes(y_train, indexes, solution_comparison, map)
    
    combined = zip(y_train, pred)
    viscosity = float(control_variable.tape_value())

    K = (float(l2_lambda*viscosity)**2) + assemble(sum(inner(true - computed, true - computed) * dx for (true, computed) in combined))

    error = float(K)

    gradient_calculation = ReducedFunctional(K, control_variable)
    gradient_calculation.optimize_tape()
    gradient = float(gradient_calculation.derivative())

    return error, gradient

def burgers_1d(viscosity, initial_condition, gradient_mode=False):
    '''Inspired by https://people.math.sc.edu/Burkardt/fenics_src/burgers_time_viscous/burgers_time_viscous.html.'''

    #Variables
    set_log_active(False)
    gridSize = 255
    viscosity = float(viscosity)
    viscosity = Constant(viscosity)
    control_variable = Control(viscosity)
    DT = Constant(0.01)
    t = 0.0
    t_final = 0.99
    result = []

    #Create grid
    mesh = IntervalMesh(gridSize, -1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    solution_comparison = Function(V)
    
    #Boundary condition
    def on_left(x, on_boundary):
        return on_boundary and near(x[0], -1.)

    def on_right(x, on_boundary):
        return on_boundary and near(x[0], 1.)

    bc_left = DirichletBC(V, 0, on_left)
    bc_right = DirichletBC(V, 0, on_right)
    bc = [bc_left, bc_right]

    #Initial condition
    u_init = Function(V)
    u_init.vector()[:] = initial_condition
    u_init.vector()[:] = u_init.vector().get_local(dof_to_vertex_map(V))

    #Trial and test functions
    u = Function(V)
    u_old = Function(V)
    v = TestFunction(V)

    #Weak formulation
    u = u_init
    u_old.assign(u)

    F = (
        dot(u - u_old, v) / DT
        + viscosity * inner(grad(u), grad(v))
        + inner(u * u.dx(0), v)
    ) * dx

    #Jacobian
    J = derivative(F, u)

    #Solving and saving for each time step
    while t < t_final:

        if gradient_mode:
            result.append(u.copy(deepcopy=True))
        else:
            result.append(u.vector().get_local(dof_to_vertex_map(V)))

        solve(F == 0, u, bc, J=J, solver_parameters={'newton_solver':  {'maximum_iterations': 50}})

        t = t + float(DT)

        u_old.assign(u)

    if gradient_mode:
        #Gradient mode is for when training
        result.append(u.copy(deepcopy=True))
    else:
        #Not gradient mode when just running a simulation.
        result.append(u.vector().get_local(dof_to_vertex_map(V)))

    if gradient_mode:
        return [result], control_variable, solution_comparison, dof_to_vertex_map(V), []
 
    return [result]
