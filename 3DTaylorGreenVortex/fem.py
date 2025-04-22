from fenics import *
from tqdm import *
import numpy as np
import time
import matplotlib.pyplot as plt

#Basic settings
set_log_level(1000)
nx, ny, nz = 4, 4, 4
Lz = 2 * np.pi
Lxy = np.pi
dt = 0.05
max_step = 50

class PeriodicBoundary(SubDomain):
    '''Periodic boundary condition.'''

    def inside(self, x, on_boundary):
        return bool ((near(x[0], -np.pi) or near(x[1], -np.pi) or near(x[2], 0)) and 
            (not ((near(x[0], np.pi) and near(x[2], 2*np.pi)) or near(x[0], np.pi)  or near(x[1], np.pi) or near(x[2], 2*np.pi) or
                  (near(x[0], np.pi) and near(x[1], np.pi)) or
                  (near(x[1], np.pi) and near(x[2], 2*np.pi)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], np.pi) and near(x[1], np.pi) and near(x[2],2*np.pi):
            y[0] = -np.pi
            y[1] = -np.pi
            y[2] = 0
        elif near(x[0], np.pi) and near(x[2], 2*np.pi):
            y[0] = -np.pi
            y[1] = x[1] 
            y[2] = 0    
        elif near(x[1], np.pi) and near(x[2], 2*np.pi):
            y[0] = x[0] 
            y[1] = -np.pi
            y[2] = 0
        elif near(x[0], np.pi) and near(x[1], np.pi):
            y[0] = -np.pi
            y[1] = -np.pi
            y[2] = x[2]
        elif near(x[0], np.pi):
            y[0] = -np.pi
            y[1] = x[1]
            y[2] = x[2]
        elif near(x[1], np.pi):
            y[0] = x[0]
            y[1] = -np.pi
            y[2] = x[2]   
        elif near(x[2], 2*np.pi):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = 0

def calculate_normalizer(p_next, mesh):
    '''Normalize pressure field for comparison with data.'''

    integral_p = assemble(p_next * dx(mesh))
    domain = assemble(Constant(1.0) * dx(mesh))
    mean_pressure = integral_p / domain

    return mean_pressure

def tgv_vortex(visc, slsqp=[], pinn=[]):
    #Inspired by https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/fenics/lid_driven_cavity.py.
    
    #Viscosity and start recording time
    print("Viscosity:", visc[0])
    start_time = time.time()
    visc = Constant(visc[0])

    #Mesh and space
    mesh = BoxMesh(Point(-np.pi, -np.pi, 0.0), Point(np.pi, np.pi, 2*np.pi), nx, ny, nz)
    pbc = PeriodicBoundary()
    velocity = VectorFunctionSpace(mesh, "Lagrange", 2, constrained_domain=pbc)
    pressure = FunctionSpace(mesh, "Lagrange", 1, constrained_domain=pbc)

    #Create all functions
    u_trial = TrialFunction(velocity)
    v_test = TestFunction(velocity)
    u_prev = Function(velocity)
    u_init = Expression(("sin(x[0])*cos(x[1])*cos(x[2])", 
                        "-cos(x[0])*sin(x[1])*cos(x[2])", "0"), degree=2)
    u_prev.assign(u_init)
    u_tent = Function(velocity)
    u_next = Function(velocity)

    p_trial = TrialFunction(pressure)
    q_test = TestFunction(pressure)
    p_next = Function(pressure)
    p_init = Expression(("0"), degree=2)
    p_next.interpolate(p_init)
    
    #Chorin's projection FEM formulations
    momentum = (((1/dt) * inner(u_trial - u_prev, v_test) * dx) + (inner(grad(u_prev) * u_prev, v_test) * dx) + (visc * inner(grad(u_trial), grad(v_test)) * dx))
    pressure = ((inner(grad(p_trial), grad(q_test)) * dx) + ((1/dt) * div(u_tent)*q_test*dx))
    velocity = ((inner(u_trial, v_test) * dx) - ((inner(u_tent,v_test) * dx) - (dt*inner(grad(p_next), v_test) * dx)))

    lhs_momentum = lhs(momentum)
    rhs_momentum = rhs(momentum)

    lhs_pressure = lhs(pressure)
    rhs_pressure = rhs(pressure)

    lhs_velocity = lhs(velocity)
    rhs_velocity = rhs(velocity)

    #For recording predictions
    predictions = []

    solver_parameters = {
    'linear_solver': 'mumps'
    }

    solver_parameters1 = {
    'linear_solver': 'gmres',
    'preconditioner': 'ilu',
    'krylov_solver': {
        'absolute_tolerance': 1e-12,
        'relative_tolerance': 1e-12,
        'maximum_iterations': 5000,
    }
    }


    for t in tqdm(range(max_step)):

        solve(
            lhs_momentum == rhs_momentum,
            u_tent,
            solver_parameters=solver_parameters1
        )
        
        solve(
            lhs_pressure == rhs_pressure,
            p_next,
            solver_parameters=solver_parameters
        )
        
        solve(
            lhs_velocity == rhs_velocity,
            u_next,
            solver_parameters=solver_parameters1
        )

        #Record relevant time steps
        if len(slsqp) > 0:
            #For FEM/SLSQP
            mean_pressure = calculate_normalizer(p_next, mesh)
            for point in slsqp[np.isclose(slsqp[:, 3] * 100, float((t+1)*5)), 0:3]:
                x_vel, y_vel, z_vel = u_next(point[0], point[1], point[2])
                pressure = p_next(point[0], point[1], point[2]) - mean_pressure
                predictions.append([x_vel, y_vel, z_vel, pressure])
        elif len(pinn) > 0:
            #For FEM/PINN
            mean_pressure = calculate_normalizer(p_next, mesh)
            for point in pinn[np.isclose(pinn[:, 3].detach().numpy() * 100, float((t+1)*5)), 0:3]:
                x_vel, y_vel, z_vel = u_next(point[0], point[1], point[2])
                pressure = p_next(point[0], point[1], point[2]) - mean_pressure
                predictions.append([x_vel, y_vel, z_vel, pressure])

        u_prev.assign(u_next)

    del mesh #Memory leak debugging
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    return predictions