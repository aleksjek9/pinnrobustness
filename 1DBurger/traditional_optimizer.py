import numpy as np
from scipy.optimize import minimize
from fem import burgers_1d, get_gradient, get_prediction_indexes
from sklearn.metrics import root_mean_squared_error


class Optimizer:
    """
    Algorithm which solves inverse problem using FEM.
    """


    def __init__(self, data, indexes, initial_condition, 
                burgers_1d=burgers_1d, get_gradient=get_gradient):
        # Initializes optimizer

        self.gradient = 0
        self.y_test = data[0]
        self.y_train = data[1]
        self.y_val = data[2]
        self.indexes = indexes[:-2]
        self.val_indexes = indexes[-2:]
        self.initial_condition = initial_condition
        self.l2_lambda = 0
        self.burgers_1d = burgers_1d
        self.get_gradient = get_gradient


    def grad(self, viscosity):
        """Returns the gradient for optimization."""

        print(viscosity)
        return np.array(self.gradient)


    def error(self, viscosity):
        """Solves 1D Burgers. Then calculates and returns the error on training set."""

        try:
            results, control_variable, solution_comparison, map, domain = self.burgers_1d(
                viscosity, self.initial_condition, gradient_mode=True
            )
        except:
            return np.nan

        error, self.gradient = self.get_gradient(
            self.l2_lambda, 
            self.indexes, 
            results, 
            self.y_train, 
            control_variable, 
            solution_comparison, 
            map
        )
        return error

    
    def error_include_val(self, viscosity):
        """Solves 1D Burgers. Then calculates and returns the error on training set. 
        Uses validation set for training as well."""

        try:
            results, control_variable, solution_comparison, map, domain = self.burgers_1d(
                viscosity, self.initial_condition, gradient_mode=True
            )
        except:
            return np.nan

        indexes = self.indexes + self.val_indexes

        error, self.gradient = self.get_gradient(
            self.l2_lambda, 
            indexes, 
            results, 
            self.y_train, 
            control_variable, 
            solution_comparison, 
            map
        )
        return error


    def validation(self):
        """Returns validation set error."""

        result = self.burgers_1d(self.viscosity, self.initial_condition)
        y_val = np.array(self.y_val).reshape(2, 256)
        result = get_prediction_indexes(result, self.val_indexes)
        rmse = root_mean_squared_error(y_val, result)
        return rmse


    def test(self):
        """Returns test set error."""

        inds = self.indexes + self.val_indexes
        result = self.burgers_1d(self.viscosity, self.initial_condition, excluded_indices=inds)
        y_test = np.array(self.y_test).reshape(-1)
        result = np.array(result).reshape(-1)
        rmse = root_mean_squared_error(y_test, result)
        return rmse


    def run(self):
        """Solves the inverse problem using Sequential Least Squares Programming optimizer."""

        options = {
            "ftol": 1e-16, 
            "maxiter": 100
        }
        
        result = minimize(
            #fun=self.error,
            fun=self.error_include_val,
            x0=[5],
            method="SLSQP",
            jac=self.grad,
            bounds=[(-5, 5)],
            callback=None,
            options = options,
        )
        return result
