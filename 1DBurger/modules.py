import numpy as np
import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim


class Model(nn.Module):

    def __init__(self):
        """Initialize the neural network model."""
        super().__init__()

        self.network = self.create_network()
        
        self.bc = None
        self.ic = None
        self.cc = None
        self.ev = None
        self.pde = None
        self.weight = 1
        self.visc = nn.Parameter(data=torch.tensor([math.log10(5)/math.log10(10)]))
        self.network.register_parameter("visc", self.visc)

        # Optimizers
        self.adam_optimizer = optim.AdamW(self.network.parameters(), weight_decay=0)
        self.lbfgs_optimizer = optim.LBFGS(
            self.network.parameters(),
            lr=1,
            max_iter=2000,
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


    def loss_fn(self, bc, ic, cc, val, pde):
        """Calculates the total loss function."""

        bc_loss = self.mse_loss(bc)
        ic_loss = self.mse_loss(ic)
        cc_loss = self.mse_loss(cc)
        phy_loss = self.phy_loss(pde)
        val_loss = self.mse_loss(val)

        total_loss = (bc_loss + ic_loss + cc_loss) * self.weight + phy_loss 
        # self.save_if_best(val_loss)
        return total_loss


    def closure(self):
        """Helper function necessary for L-BFGS."""

        self.lbfgs_optimizer.zero_grad()
        loss = self.loss_fn(self.bc, self.ic, self.cc, self.ev, self.pde)
        loss.backward()
        return loss

    def train_model(self, bc, ic, cc, val, pde, iterations):
        """Trains the model with both optimizers."""

        # Training with ADAM
        for iter in range(iterations):
            loss = self.loss_fn(bc, ic, cc, val, pde)
            self.adam_optimizer.zero_grad()
            loss.backward()
            self.adam_optimizer.step()

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
                # print("New weight:", self.weight)
                
        # self.network.load_state_dict(torch.load("best.hdf5"))

        # Training with L-BFGS
        self.bc = bc
        self.ic = ic
        self.cc = cc
        self.ev = val
        self.pde = pde
        self.lbfgs_optimizer.step(self.closure)
        # self.network.load_state_dict(torch.load("best.hdf5"))
