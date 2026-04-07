import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Model(nn.Module):

    def __init__(self, name):
        """Initialize the neural network model."""
        super().__init__()

        self.network = self.create_network()
        
        self.bc = None
        self.ic = None
        self.cc = None
        self.ev = None
        self.pde = None
        self.minutes = []
        self.phy_history = []
        self.data_history = []
        self.val_history = []
        self.parameter_history = []
        self.name = name
        self.weight = 1
        self.epoch = 0
        self.start_time = time.time()
        self.visc = nn.Parameter(data=torch.tensor([math.log10(5)/math.log10(10)]))
        self.network.register_parameter("visc", self.visc)

        # Optimizers
        self.adam_optimizer = optim.AdamW(self.network.parameters(), weight_decay=0)
        self.lbfgs_optimizer = optim.LBFGS(
            self.network.parameters(),
            lr=1,
            max_iter=4000,
            max_eval=None,
            tolerance_grad=1e-16,
            tolerance_change=1e-16,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        self.val_loss = None
        self.test_loss_history = []
        self.x_test = None
        self.y_test = None 


    def create_network(self):
        """Create the neural network itself."""

        network = []

        network.append(nn.Linear(2, 20).double())
        network.append(nn.Tanh().double())
        network.append(nn.Linear(20, 20).double())
        network.append(nn.Tanh().double())
        network.append(nn.Linear(20, 20).double())
        network.append(nn.Tanh().double())
        network.append(nn.Linear(20, 20).double())
        network.append(nn.Tanh().double())
        network.append(nn.Linear(20, 1).double())

        return nn.Sequential(*network)


    def forward(self, inputs):
        """Run a forward pass through the neural network."""
        
        inputs = inputs[:, 0:2]

        output = self.network(inputs)
        return output


    def mse_loss(self, data):
        """Calculate data loss."""

        if len(data[0]) == 0:
            return torch.tensor([0])

        output = self.forward(data[0])
        return torch.mean((output - data[1]) ** 2)

    def flush_histories(self):

        with open('results/minutes_' + self.name + '.txt', 'a') as f:
                for item in self.minutes:
                    f.write(f"{item}, ")

        with open('results/phy_history_' + self.name + '.txt', 'a') as f:
            for item in self.phy_history:
                f.write(f"{item}, ")

        with open('results/data_history_' + self.name + '.txt', 'a') as f:
            for item in self.data_history:
                f.write(f"{item}, ")

        with open('results/val_history_' + self.name + '.txt', 'a') as f:
            for item in self.val_history:
                f.write(f"{item}, ")

        with open('results/parameter_history_' + self.name + '.txt', 'a') as f:
            for item in self.parameter_history:
                f.write(f"{item}, ")

        with open('results/test_history_' + self.name + '.txt', 'a') as f:
            for item in self.test_loss_history:
                f.write(f"{item}, ") 
        
        # Empty lists
        self.minutes = []
        self.phy_history = []
        self.data_history = []
        self.val_history = []
        self.parameter_history = []
        self.test_loss_history = []


    def save_history(self, elapsed_minutes, phy_loss, cc_loss, val_loss, visc):
        """
        Saves various loss histories that can be
        plotted later for better understanding.
        """

        self.minutes.append(elapsed_minutes)
        self.phy_history.append(phy_loss)
        self.data_history.append(cc_loss)
        self.parameter_history.append(visc)
        self.val_history.append(val_loss)

        test_loss = self.mse_loss([self.x_test, self.y_test])
        self.test_loss_history.append(test_loss.item())

        if self.epoch in list(range(1, 10000, 100)):

            with open('results/minutes_' + self.name + '.txt', 'a') as f:
                for item in self.minutes:
                    f.write(f"{item}, ")

            with open('results/phy_history_' + self.name + '.txt', 'a') as f:
                for item in self.phy_history:
                    f.write(f"{item}, ")

            with open('results/data_history_' + self.name + '.txt', 'a') as f:
                for item in self.data_history:
                    f.write(f"{item}, ")

            with open('results/val_history_' + self.name + '.txt', 'a') as f:
                for item in self.val_history:
                    f.write(f"{item}, ")

            with open('results/parameter_history_' + self.name + '.txt', 'a') as f:
                for item in self.parameter_history:
                    f.write(f"{item}, ")

            with open('results/test_history_' + self.name + '.txt', 'a') as f:
                for item in self.test_loss_history:
                    f.write(f"{item}, ") 
            
            # Empty lists until next writing
            self.minutes = []
            self.phy_history = []
            self.data_history = []
            self.val_history = []
            self.parameter_history = []
            self.test_loss_history = []


    def phy_loss(self, pde):
        """Calculate the physics loss."""

        pde.requires_grad = True
        output = self.forward(pde)

        y_t = autograd.grad(
            output, pde,
            retain_graph=True,
            grad_outputs = torch.ones_like(output),
            create_graph=True,
        )

        y_t = y_t[0][:, 1]

        y_x = autograd.grad(
            output, pde,
            retain_graph=True,
            grad_outputs = torch.ones_like(output),
            create_graph=True,
        )

        y_x2 = autograd.grad(
            y_x[0], pde,
            retain_graph=True,
            grad_outputs = torch.ones_like(y_x[0]),
            create_graph=True,
        )

        y_x = y_x[0][:, 0]
        y_x2 = y_x2[0][:, 0]

        # Note logarithmic search
        pde_loss = y_t + (torch.flatten(output) * y_x) - ((10**self.visc) * y_x2)
        pde_loss = torch.square(pde_loss).mean()
        return pde_loss
    

    def save_if_best(self, val_loss):
        """
        Saves the best model so far, based on validation step.
        Loading is disabled by default in train_model().
        """

        if self.val_loss is None or self.val_loss > val_loss:
            self.val_loss = val_loss
            torch.save(self.network.state_dict(), "best.hdf5")
            print("Saved")


    def loss_fn(self, bc, ic, cc, val, pde):
        """Calculates the total loss function."""

        bc_loss = self.mse_loss(bc)
        ic_loss = self.mse_loss(ic)
        cc_loss = self.mse_loss(cc)
        phy_loss = self.phy_loss(pde)
        val_loss = self.mse_loss(val)

        data_loss = bc_loss + ic_loss + cc_loss

        #value = 1-(self.epoch/9000)
        #total_loss = max(data_loss, (data_loss * self.weight * value)) + phy_loss 
        total_loss = data_loss * self.weight + phy_loss

        elapsed_time = time.time() - self.start_time
        elapsed_minutes = elapsed_time / 60

        print("Epoch: ", self.epoch, " Minutes: ", elapsed_minutes, " Phy_loss: ", phy_loss.item(), " Data_loss: ", data_loss.item(), " Parameter: ", 10**self.visc.item())

        self.save_history(
            elapsed_minutes, 
            phy_loss.item(), 
            data_loss.item(), 
            val_loss.item(), 
            10**self.visc.item()
        )

        self.save_if_best(val_loss)
        return total_loss


    def closure(self):
        """Helper function necessary for L-BFGS."""

        self.epoch += 1
        self.lbfgs_optimizer.zero_grad()
        loss = self.loss_fn(self.bc, self.ic, self.cc, self.ev, self.pde)
        loss.backward()
        return loss

    def train_model(self, bc, ic, cc, val, pde, iterations, tests):
        """Trains the model with both optimizers."""

        self.x_test = tests[0].to(device)
        self.y_test = tests[1].to(device) 

        bc[0] = bc[0].to(device)
        bc[1] = bc[1].to(device)
        ic[0] = ic[0].to(device)
        ic[1] = ic[1].to(device)
        cc[0] = cc[0].to(device)
        cc[1] = cc[1].to(device)
        cc[0] = cc[0].to(device)
        val[1] = val[1].to(device)
        val[0] = val[0].to(device)
        pde = pde.to(device)

        # Training with ADAM
        for iter in range(iterations):
            self.epoch += 1
            loss = self.loss_fn(bc, ic, cc, val, pde)
            self.adam_optimizer.zero_grad()
            loss.backward()
            self.adam_optimizer.step()

            """Gradient pathologies adaptive weight from https://arxiv.org/abs/2001.04536."""
            '''if iter % 10 == 0:
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
                loss = self.mse_loss(cc) + self.mse_loss(ic) + self.mse_loss(bc)
                self.adam_optimizer.zero_grad()
                loss.backward()
                gradients = []

                for param in self.parameters():
                    if param.grad is not None:
                        gradients.append(param.grad.abs())

                all_gradients = torch.cat([g.view(-1) for g in gradients])
                mean_gradient = all_gradients.mean()
                mean_grad = mean_gradient.item()

                # Calculate the new weight using alpha=0.9 | note that 1.0 - alpha = 0.1
                self.weight = 0.1 * self.weight + 0.9 * (max_grad / mean_grad)
                print("New weight:", self.weight)'''
                
        #self.network.load_state_dict(torch.load("best.hdf5"))

        # Training with L-BFGS
        self.bc = bc
        self.ic = ic
        self.cc = cc
        self.ev = val
        self.pde = pde
        self.lbfgs_optimizer.step(self.closure)
        self.flush_histories()
        #self.network.load_state_dict(torch.load("best.hdf5"))
