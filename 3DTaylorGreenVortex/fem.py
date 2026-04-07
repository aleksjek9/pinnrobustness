from fenics import *
from tqdm import *
import numpy as np
import time
import matplotlib.pyplot as plt
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Basic settings
set_log_level(1000)
nx, ny, nz = 32,32,32
Lz = 2 * np.pi
Lxy = np.pi
dt = 0.05
time_scale = 100
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

    complete = False
    
    while not complete:
    
        try:
            predictions = tgv_vortex_go(visc, slsqp, pinn)
            complete = True
        except Exception as e:
            print("Probably parallelization error with initialization.")
            print("Actual error:", e)
            
    return predictions
    

def tgv_vortex_go(visc, slsqp=[], pinn=[]):
    """Inspired by:
    https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/fenics/lid_driven_cavity.py.
    
    Taylor-Green Vortex using Chorin's projection FEM formulations.
    
    """
    
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

    #Parallell safe initialization
    dof_coords = velocity.tabulate_dof_coordinates().reshape((-1, 3))
    local_dofs = velocity.dofmap().dofs()
    owned_dofs = velocity.dofmap().ownership_range()
    owned_dofs = range(owned_dofs[0], owned_dofs[1])
    
    local_values = np.zeros(len(local_dofs))
    
    for i, dof in enumerate(local_dofs):
        if dof in owned_dofs:
            x, y, z = dof_coords[i]
            comp = dof % 3
            if comp == 0:
                local_values[i] = np.sin(x) * np.cos(y) * np.cos(z)
            elif comp == 1:
                local_values[i] = -np.cos(x) * np.sin(y) * np.cos(z)
            elif comp == 2:
                local_values[i] = 0.0
                
    u_prev.vector().set_local(local_values)
    u_prev.vector().apply("insert")
    
    u_tent = Function(velocity)
    u_next = Function(velocity)

    p_trial = TrialFunction(pressure)
    q_test = TestFunction(pressure)
    
    p_next = Function(pressure)
    dofmap = pressure.dofmap()
    local_dofs = dofmap.dofs()
    owned_dofs = range(*dofmap.ownership_range())
    local_values = np.zeros(len(local_dofs))
    p_next.vector().set_local(local_values)
    p_next.vector().apply("insert")
    
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
    'preconditioner': 'amg',
    'krylov_solver': {
        'absolute_tolerance': 1e-12,
        'relative_tolerance': 1e-10,
        'maximum_iterations': 50000,
    }
    }

    # For recording local predictions
    local_predictions = []

    for t in tqdm(range(max_step)):

        solve(
            lhs_momentum == rhs_momentum,
            u_tent,
            solver_parameters=solver_parameters1
        )
        
        solve(
            lhs_pressure == rhs_pressure,
            p_next,
            solver_parameters=solver_parameters1
        )
        
        solve(
            lhs_velocity == rhs_velocity,
            u_next,
            solver_parameters=solver_parameters1
        )
        
        mean_pressure = calculate_normalizer(p_next, mesh)

        #Record relevant time steps
        if len(slsqp) > 0:
            # For parallellized FEM/SLSQP
            time_filtered = [
                (idx, point) for idx, point in enumerate(slsqp)
                if np.isclose(point[3] * time_scale, float((t+1)*5))
            ] 
            for idx, point in time_filtered:
                x, y, z = point[0], point[1], point[2]
                try:
                    x_vel, y_vel, z_vel = u_next(x, y, z)
                    p_val = p_next(x, y, z) - mean_pressure
                    local_predictions.append((idx, [x_vel, y_vel, z_vel, p_val]))
                except Exception as e:
                    #print(f"[Rank {rank}] Evaluation failed at ({x:.3f}, {y:.3f}) β†’ {e}", flush=True)
                    pass
        elif len(pinn) > 0:
            time_filtered = [
                (idx, point) for idx, point in enumerate(pinn)
                if np.isclose(point[3] * time_scale, float((t+1)*5))
            ]
            for idx, point in time_filtered:
                x, y, z = point[0], point[1], point[2]
                try:
                    x_vel, y_vel, z_vel = u_next(x, y, z)
                    p_val = p_next(x, y, z) - mean_pressure
                    local_predictions.append((idx, [x_vel, y_vel, z_vel, p_val]))
                except Exception as e:
                    #print(f"[Rank {rank}] Evaluation failed at ({x:.3f}, {y:.3f}) β†’ {e}", flush=True)
                    pass

        u_prev.assign(u_next)

    print(f"Rank {rank} done/saved predictions", flush=True)

    all_predictions = comm.gather(local_predictions, root=0)

    if (rank == 0):
        """
        Gather all local predictions across ranks and sort them by index.
        After removing duplicate entries, then broadcast the final result to all ranks.
        """
        combined = []

        for item in all_predictions:
            combined.extend(item)
        combined.sort(key=lambda x: x[0])
        unique = {}
        for index, vec in combined:
            unique[index] = vec
        cleaned = [unique[k] for k in sorted(unique)]
    else:
        cleaned = None

    predictions = comm.bcast(cleaned, root=0)

    del mesh #Memory leak debugging

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Elapsed time: {elapsed_time:.2f} seconds") if rank==0 else None

    return predictions
