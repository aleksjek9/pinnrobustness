import numpy as np
import time
import random
import secrets
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim


def gradient(output, input, create=True):
    """Get gradient output with regards to input."""

    grad = autograd.grad(
        outputs=output,
        inputs=input,
        grad_outputs=torch.ones_like(output),
        create_graph=create,
        retain_graph=True,
    )[0]

    return grad


class Model(nn.Module):
    
    def __init__(self, name):
        """Initialize the neural network model."""
        super().__init__()

        self.network = self.create_network()

        self.start_time = time.time()
        self.patience = 0
        self.epoch = 0
        self.weight = 1
        self.minutes = []
        self.phy_history = []
        self.data_history = []
        self.val_history = []
        self.parameter_history = []
        self.weights_history = []
        self.name = name
        self.cc = None
        self.ev = None
        self.pde = None
        self.visc = nn.Parameter(data=torch.tensor([4.99007631268])) # log(e^4.99685840735−1) to get 5 with offset softplus
        self.network.register_parameter("visc", self.visc)

        # Optimizers
        self.adam_optimizer = optim.AdamW(self.network.parameters(), weight_decay=0)
        self.lbfgs_optimizer = torch.optim.LBFGS(
            self.network.parameters(),
            lr=1,
            max_iter=50000,
            max_eval=None,
            tolerance_grad=1e-16,
            tolerance_change=1e-16,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        self.val_loss = None


    def create_network(self):
        """Create the neural network itself."""

        network = []

        network.append(nn.Linear(3, 20).double())
        network.append(nn.Tanh().double())
        network.append(nn.Linear(20, 20).double())
        network.append(nn.Tanh().double())
        network.append(nn.Linear(20, 20).double())
        network.append(nn.Tanh().double())
        network.append(nn.Linear(20, 20).double())
        network.append(nn.Tanh().double())
        network.append(nn.Linear(20, 20).double())
        network.append(nn.Tanh().double())
        network.append(nn.Linear(20, 20).double())
        network.append(nn.Tanh().double())
        network.append(nn.Linear(20, 20).double())
        network.append(nn.Tanh().double())
        network.append(nn.Linear(20, 20).double())
        network.append(nn.Tanh().double())
        network.append(nn.Linear(20, 20).double())
        network.append(nn.Tanh().double())
        network.append(nn.Linear(20, 2).double())

        return nn.Sequential(*network)


    def forward(self, inputs):
        """Run a forward pass through the neural network."""
        
        inputs = inputs[:, 0:3]

        output = self.network(inputs)
        return output


    def mse_loss(self, data):
        """Calculate data loss."""

        if len(data[0]) == 0:
            return torch.tensor([0])

        input = data[0]
        x, y, t = input[:, 0], input[:, 1], input[:, 2]
        x.requires_grad, y.requires_grad, t.requires_grad = True, True, True
        
        input = torch.stack((x, y, t), dim=1)
        output = self.forward(input)
        
        u_pred = gradient(output[:, 0], x)
        v_pred = -1 * gradient(output[:, 0], y)
        p_pred = output[:, 1]
        output = torch.stack((u_pred, v_pred, p_pred), dim=1)
        
        data_loss = torch.mean((output - data[1]) ** 2)
        return data_loss

    def root_mean_squared_error(self, y_true, y_pred):
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

    def save_history(self, elapsed_minutes, phy_loss, cc_loss, val, visc):
        """
        Saves various loss histories that can be
        plotted later for better understanding.
        """

        self.minutes.append(elapsed_minutes)
        self.phy_history.append(phy_loss)
        self.data_history.append(cc_loss)
        self.parameter_history.append(visc)
        
        x, y, t = self.x_test[:, 0], self.x_test[:, 1], self.x_test[:, 2]
        x.requires_grad, y.requires_grad, t.requires_grad = True, True, True
        x_test = torch.stack((x, y, t), dim=1)
        pred = self.forward(x_test)
        u_pred = gradient(pred[:, 0], x, create=False)
        v_pred = -1 *gradient(pred[:, 0], y, create=False)
        pred = torch.stack((u_pred, v_pred), dim=1)
        error = self.root_mean_squared_error(pred, self.y_test)
        self.total_history.append(error)


        if self.epoch in list(range(1, 300002, 10000)):

            with open('results/minutes_' + self.name + '.txt', 'a') as f:
                for item in self.minutes:
                    f.write(f"{item}, ")

            with open('results/phy_history_' + self.name + '.txt', 'a') as f:
                for item in self.phy_history:
                    f.write(f"{item}, ")

            with open('results/data_history_' + self.name + '.txt', 'a') as f:
                for item in self.data_history:
                    f.write(f"{item}, ")

            val_loss = [self.mse_loss(val)]

            with open('results/val_history_' + self.name + '.txt', 'a') as f:
                for item in val_loss:
                    f.write(f"{item}, ")

            with open('results/parameter_history_' + self.name + '.txt', 'a') as f:
                for item in self.parameter_history:
                    f.write(f"{item}, ")

            with open('results/weights_history_' + self.name + '.txt', 'a') as f:
                for item in self.weights_history:
                    f.write(f"{item}, ")
                    
            with open('results/total_history_' + self.name + '.txt', 'a') as f:
                for item in self.total_history:
                    f.write(f"{item}, ")
            
            # Empty lists until next writing
            self.minutes = []
            self.phy_history = []
            self.data_history = []
            self.val_history = []
            self.parameter_history = []
            self.weights_history = []
            self.total_history = [] 

    def phy_loss(self, pde):
        """
        Calculate the physics loss.
        Inspired by https://github.com/chen-yingfa/pinn-torch.
        """

        x, y, t = pde[:, 0], pde[:, 1], pde[:, 2]
        x.requires_grad, y.requires_grad, t.requires_grad = True, True, True
        pde = torch.stack((x, y, t), dim=1)

        output = self.forward(pde)

        viscosity = torch.nn.functional.softplus(self.visc) + 0.00314159265

        u_pred = gradient(output[:, 0], x)
        v_pred = -1 * gradient(output[:, 0], y)
        p_pred = output[:, 1]

        u_t = gradient(u_pred, t, create=False)
        u_x = gradient(u_pred, x)
        u_y = gradient(u_pred, y)

        u_xx = gradient(u_x, x, create=False)
        u_yy = gradient(u_y, y, create=False)

        v_t = gradient(v_pred, t, create=False)
        v_x = gradient(v_pred, x)
        v_y = gradient(v_pred, y)

        v_xx = gradient(v_x, x, create=False)
        v_yy = gradient(v_y, y, create=False)

        p_x = gradient(p_pred, x, create=False)
        p_y = gradient(p_pred, y, create=False)

        pde_loss2 = u_t + u_pred*u_x + v_pred*u_y + p_x - viscosity*(u_xx+u_yy)
        pde_loss3 = v_t + u_pred*v_x+v_pred*v_y+p_y-viscosity*(v_xx + v_yy)

        loss = torch.square(pde_loss2 + pde_loss3).mean()
        return loss
    
    def save_if_best(self, val_loss, phy_loss):
        """
        Saves the best model so far, based on validation step.
        Loading is disabled by default in train_model().
        """
        
        if (self.later_val_loss is None or self.later_val_loss > val_loss) and self.epoch > 10000:
            self.later_val_loss = val_loss
            torch.save(self.network.state_dict(), "results/best_later_val" + self.name + ".hdf5")
            with open("results/val_saved_epoch_" + self.name + ".txt", "w") as f:
                f.write(str(self.epoch))
            
        if (self.later_phy_loss is None or self.later_phy_loss > phy_loss) and self.epoch > 10000:
            self.later_phy_loss = phy_loss
            torch.save(self.network.state_dict(), "results/best_later_phy" + self.name + ".hdf5")
            with open("results/phy_saved_epoch_" + self.name + ".txt", "w") as f:
                f.write(str(self.epoch))

    def loss_fn(self, cc, val, pde, lbfgs=False):
        """Calculates the full loss function."""

        cc_loss = self.mse_loss(cc)
        phy_loss = self.phy_loss(pde)
        bc_loss = self.mse_loss(self.bc)
        ic_loss = self.mse_loss(self.ic)
        
        data_loss = cc_loss + bc_loss + ic_loss

        total_loss = data_loss * self.weight + phy_loss

        elapsed_time = time.time() - self.start_time
        elapsed_minutes = elapsed_time / 60

        print("Epoch: ", self.epoch, " Minutes: ", elapsed_minutes, " Phy_loss: ", phy_loss.item(), " Data_loss: ", data_loss.item(), " Parameter: ", torch.nn.functional.softplus(self.visc).item() + 0.00314159265)
        
        self.save_history(
            elapsed_minutes, 
            phy_loss.item(), 
            data_loss.item(), 
            val, 
            torch.nn.functional.softplus(self.visc).item() + 0.00314159265
        )
        
        val_loss = self.mse_loss(val)
        self.save_if_best(val_loss, phy_loss)
            
        return total_loss

    def closure(self):
        """Helper function necessary for L-BFGS."""

        self.epoch += 1
        self.lbfgs_optimizer.zero_grad()
        loss = self.loss_fn(self.cc, self.ev, self.pde, lbfgs=True)
        loss.backward()
        return loss

    def train_model(self, cc, val, pde, iterations, tests, bcic):
        """Trains the model with both optimizers."""
        
        self.x_test = tests[0]
        self.y_test = tests[1]
        self.bc = bcic[0]
        self.ic = bcic[1]

        # Training with ADAM
        for iter in range(iterations):
            self.epoch += 1
            loss = self.loss_fn(cc, val, pde)
            self.adam_optimizer.zero_grad()
            loss.backward()
            self.adam_optimizer.step()

            if self.patience > 60000:
                iter = iterations

            """Gradient pathologies adaptive weight from https://arxiv.org/abs/2001.04536."""
            if iter % 10 == 0:
                # Get max gradient of physics loss
                phy_loss = self.phy_loss(pde)
                self.adam_optimizer.zero_grad()
                phy_loss.backward()
                gradients = []

                for param in self.parameters():
                    if param.grad is not None:
                        gradients.append(param.grad.abs().max())

                max_gradient = torch.max(torch.stack(gradients))
                max_grad = max_gradient.item()

                # Get mean absolute gradient of data loss
                cc_loss = self.mse_loss(cc)
                bc_loss = self.mse_loss(self.bc)
                ic_loss = self.mse_loss(self.ic)
                data_loss = cc_loss + bc_loss + ic_loss
                self.adam_optimizer.zero_grad()
                data_loss.backward()
                gradients = []

                for param in self.parameters():
                    if param.grad is not None:
                        gradients.append(param.grad.abs())

                all_gradients = torch.cat([g.view(-1) for g in gradients])
                mean_gradient = all_gradients.mean()
                mean_grad = mean_gradient.item()

                # Calculate the new weight using alpha=0.9 |note that 1.0 - alpha = 0.1
                self.weights_history.append(self.weight)
                self.weight = 0.1 * self.weight + 0.9 * (max_grad / mean_grad)
                print("New weight:", self.weight)
        
        #self.network.load_state_dict(torch.load("best.hdf5"))
        self.patience = 0

        # Train with L-BFGS
        self.cc = cc
        self.ev = val
        self.pde = pde
        self.lbfgs_optimizer.step(self.closure)
        self.network.load_state_dict(torch.load("best.hdf5"))
