from fem import tgv_vortex
import numpy as np
from scipy.optimize import minimize
import importlib
import time

class Optimizer:
    """
    Algorithm which solves inverse problem using FEM.
    """


    def __init__(self, data):
        # Initializes optimizer

        self.x_test = np.array(data[0])
        self.y_test = np.array(data[1])
        self.x_train = np.array(data[2])
        self.y_train = data[3]
        self.y_val = data[4]
        self.x_val = np.array(data[5])
        self.l2_lambda = -1
        self.viscosity = [0]

    def error(self, viscosity):
        #Return error on training set

        predictions = tgv_vortex(viscosity, slsqp=self.x_train)
        #print(predictions[0:10])
        #print(self.y_train[0:10])
        rmse = float((self.l2_lambda*viscosity)**2) + np.sqrt(np.mean((np.array(predictions)[:, 0:3] - self.y_train[~np.isclose(self.x_train[:, 3], 0.0), 0:3]) ** 2))
        pressure_rmse = np.sqrt(np.mean((np.array(predictions)[:, 3] - self.y_train[~np.isclose(self.x_train[:, 3], 0.0), 3]) ** 2))
        #print(rmse)

        return rmse - float((self.l2_lambda*self.viscosity[0])**2)

    def error_include_val(self, viscosity):
        """Return error on training set."""

        x_train_val = np.concatenate((self.x_train, self.x_val))
        y_train_val = np.concatenate((self.y_train, self.y_val))

        sort_by_time = np.argsort(x_train_val[:, 3])
        x_train_val = x_train_val[sort_by_time]
        y_train_val = y_train_val[sort_by_time]

        predictions = np.array(tgv_vortex(viscosity, slsqp=x_train_val))
        rmse = float((self.l2_lambda*viscosity)**2) + np.sqrt(np.mean((np.array(predictions)[:, 0:3] - y_train_val[~np.isclose(x_train_val[:, 3], 0.0), 0:3]) ** 2))
        pressure_rmse = np.sqrt(np.mean((np.array(predictions)[:, 3] - y_train_val[~np.isclose(x_train_val[:, 3], 0.0), 3]) ** 2))
        print(rmse)

        return rmse - float((self.l2_lambda*self.viscosity[0])**2)

    def validation(self):
        #Return error on validation set

        if (self.viscosity == 0) or (self.l2_lambda == -1):
            return "You have to set viscosity and l2 lambda values before validation."

        predictions = tgv_vortex(self.viscosity, self.x_val)
        rmse = float((self.l2_lambda*self.viscosity[0])**2) + np.sqrt(np.mean((np.array(predictions)[:, 0:3] - self.y_val[~np.isclose(self.x_val[:, 3], 0.0), 0:3]) ** 2))
        pressure_rmse = np.sqrt(np.mean((np.array(predictions)[:, 3] - self.y_val[~np.isclose(self.x_val[:, 3], 0.0), 3]) ** 2))

        return rmse - float((self.l2_lambda*self.viscosity[0])**2)

    def test(self):
        #Returns test set error

        if (self.viscosity == 0) or (self.l2_lambda == -1):
            return "You have to set viscosity and l2 lambda values before test."

        predictions = tgv_vortex(self.viscosity, self.x_test)
        rmse = np.sqrt(np.mean((np.array(predictions)[:, 0:3] - self.y_test[~np.isclose(self.x_test[:, 3], 0.0), 0:3]) ** 2))
        pressure_rmse = np.sqrt(np.mean((np.array(predictions)[:, 3] - self.y_test[~np.isclose(self.x_test[:, 3], 0.0), 3]) ** 2))
        
        return rmse

    def run(self):
        #Solves the inverse problem

        options = {
            "ftol": 1e-16, 
            "maxiter": 100, 
            "disp": True
        }
        
        done = False
        
        while done == False:

            result = minimize(
                fun=self.error_include_val,
                x0=[5],
                method="SLSQP",
                jac="3-point",
                bounds=[(0.00314159265, 5)],
                options = options,
            )
            
            if (result.nit > 1) and result.success:
                done = True

        print("Final viscosity:", result["x"])

        return result
