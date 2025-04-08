import numpy as np
from scipy.optimize import minimize
from fem import burgers_1d, get_gradient, get_prediction_indexes
from sklearn.metrics import root_mean_squared_error


class optimizer:
    #Algorithm which solves inverse problem using FEM

    def __init__(self, data, indexes, initial_condition, burgers_1d=burgers_1d, get_gradient=get_gradient):
        #Initializes optimizer

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
        #Returns the gradient

        print(viscosity)

        return np.array(self.gradient)

    def error(self, viscosity):
        #Returns the error on training set

        try:
            #Solves 1D Burgers
            results, control_variable, solution_comparison, map, domain = self.burgers_1d(viscosity, self.initial_condition, gradient_mode=True)
        except:
            return np.nan

        #Calculates the error
        error, self.gradient = self.get_gradient(self.l2_lambda, self.indexes, results, self.y_train, control_variable, solution_comparison, map)

        return error

    def validation(self):
        #Returns validation set error

        result = self.burgers_1d(self.viscosity, self.initial_condition)
        y_val = np.array(self.y_val).reshape(2, 256)
        result = get_prediction_indexes(result, self.val_indexes)

        rmse = root_mean_squared_error(y_val, result)

        return rmse


    def test(self):
        #Returns test set error

        result = self.burgers_1d(self.viscosity, self.initial_condition)
        y_test = np.array(self.y_test).reshape(25600)
        result = np.array(result).reshape(25600)
        
        rmse = root_mean_squared_error(y_test, result)

        return rmse


    def run(self):
        #Solves the inverse problem

        options = {"ftol": 1e-16, "maxiter": 100}

        result = minimize(
            fun=self.error,
            x0=[5],
            method="SLSQP",
            jac=self.grad,
            bounds=[(-5, 5)],
            callback=None,
            options = options,
        )

        result["x"]

        return result
