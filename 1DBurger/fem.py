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
    dt = 0.01
    t = 0.0
    t_final = 0.99
    result = []

    #Create grid
    mesh = IntervalMesh(gridSize, -1, 1)
    velocity = FunctionSpace(mesh, "CG", 1)
    solution_comparison = Function(velocity)
    
    #Boundary condition
    def left_boundary(x, on_boundary):
        return on_boundary and near(x[0], -1.)

    def right_boundary(x, on_boundary):
        return on_boundary and near(x[0], 1.)

    left_dirichlet = DirichletBC(velocity, 0, left_boundary)
    right_dirichlet = DirichletBC(velocity, 0, right_boundary)
    boundary_conditions = [left_dirichlet, right_dirichlet]

    #Initial condition
    init_vel = Function(velocity)
    init_vel.vector()[:] = initial_condition
    init_vel.vector()[:] = init_vel.vector().get_local(dof_to_vertex_map(velocity))

    #Trial and test functions
    cur_vel = Function(velocity)
    old_vel = Function(velocity)
    test_vel = TestFunction(velocity)

    #Weak formulation
    cur_vel = init_vel
    old_vel.assign(cur_vel)

    F = (dot(cur_vel - old_vel, test_vel) / dt + viscosity * inner(grad(cur_vel), grad(test_vel)) + inner(cur_vel * cur_vel.dx(0), test_vel)) * dx

    #Jacobian
    jacobian = derivative(F, cur_vel)

    #Solving and saving for each time step
    while t < t_final:

        solve(F == 0, cur_vel, boundary_conditions, J=jacobian, solver_parameters={'newton_solver':  {'maximum_iterations': 50}})

        if gradient_mode:
            result.append(cur_vel.copy(deepcopy=True))
        else:
            result.append(cur_vel.vector().get_local(dof_to_vertex_map(velocity)))

        t = t + float(dt)

        old_vel.assign(cur_vel)

    if gradient_mode:
        return [result], control_variable, solution_comparison, dof_to_vertex_map(velocity), []
 
    return [result]
