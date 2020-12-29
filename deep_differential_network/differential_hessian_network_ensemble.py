import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from deep_differential_network.activations import *


class DifferentialLayer(nn.Module):

    def __init__(self, n_network, input_size, output_size, activation="ReLu"):
        super(DifferentialLayer, self).__init__()

        # Create layer weights and biases:
        self.n_output = output_size
        self.weight = nn.Parameter(torch.empty(n_network, 1, output_size, input_size))
        self.bias = nn.Parameter(torch.empty(n_network, 1, output_size, 1))

        # Initialize activation function and its derivative:
        if activation == "ReLu":
            self.g = nn.ReLU()
            self.g_prime = ReLUDer()
            self.g_pprime = ReLUDer2()

        elif activation == "SoftPlus":
            self.softplus_beta = 1.0
            self.g = nn.Softplus(beta=self.softplus_beta)
            self.g_prime = SoftplusDer(beta=self.softplus_beta)
            self.g_pprime = SoftplusDer2()

        elif activation == "Cos":
            self.g = Cos()
            self.g_prime = CosDer()
            self.g_pprime = CosDer2()

        elif activation == "Linear":
            self.g = Linear()
            self.g_prime = LinearDer()
            self.g_pprime = LinearDer2()

        elif activation == "Tanh":
            self.g = Tanh()
            self.g_prime = TanhDer()
            self.g_pprime = TanhDer2()

        elif activation == "Quad":
            self.g = Quad()
            self.g_prime = QuadDer()
            self.g_pprime = QuadDer2()

        elif activation == "Cubic":
            self.g = Cubic()
            self.g_prime = CubicDer()
            self.g_pprime = CubicDer2()

        else:
            raise ValueError("Activation Type must be in ['Linear', 'ReLu', 'SoftPlus', 'Cos'] but is {0}".format(self.activation))

    def forward(self, h, dh_dx, d2h_d2x, hessian=False):
        # Apply Affine Transformation:
        a = torch.matmul(self.weight, h) + self.bias

        # Compute the output:
        hi = self.g(a)

        # Compute the jacobian:
        dhi_dh = self.g_prime(a) * self.weight
        dhi_dx = torch.matmul(dhi_dh, dh_dx)

        # Compute the hessian:
        if hessian:
            tmp = (self.weight.transpose(2, 3).unsqueeze(-1) * (self.g_pprime(a) * self.weight).unsqueeze(2))
            # p1 = torch.matmul(torch.matmul(tmp, dh_dx.unsqueeze(1)).transpose(1, 3), dh_dx.unsqueeze(1)).transpose(1, 3)
            p1 = torch.matmul(torch.matmul(tmp, dh_dx.unsqueeze(2)).transpose(2, 4), dh_dx.unsqueeze(2)).transpose(2, 4)
            p2 = torch.matmul(dhi_dh.unsqueeze(2), d2h_d2x)
            d2hi_d2x = p1 + p2

        else:
            d2hi_d2x = d2h_d2x

        return hi, dhi_dx, d2hi_d2x


class DifferentialNetwork(nn.Module):
    name = "Differential Network"

    def __init__(self, n_input, **kwargs):
        super(DifferentialNetwork, self).__init__()

        # Read optional arguments:
        self.n_input = n_input
        self.n_network = kwargs.get('n_network', 1)
        self.n_width = kwargs.get("n_width", 128)
        self.n_hidden = kwargs.get("n_depth", 1)
        self.n_output = kwargs.get("n_output", 1)
        non_linearity = kwargs.get("activation", "ReLu")

        # Initialization of the layers:
        self._w_init = kwargs.get("w_init", "xavier_normal")
        self._b_hidden = kwargs.get("b_hidden", 0.1)
        self._b_output = kwargs.get("b_output", 0.1)
        self._g_hidden = kwargs.get("g_hidden", np.sqrt(2.))
        self._g_output = kwargs.get("g_output", 1.0)
        self._p_sparse = kwargs.get("p_sparse", 0.2)

        # Construct Weight Initialization:
        if self._w_init == "xavier_normal":

            # Construct initialization function:
            def init_hidden(layer):

                # Set the Hidden Gain:
                if self._g_hidden <= 0.0: hidden_gain = torch.nn.init.calculate_gain('relu')
                else: hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b_hidden)
                torch.nn.init.xavier_normal_(layer.weight, hidden_gain)

                with torch.no_grad():
                    layer.weight = torch.nn.Parameter(layer.weight)

            def init_output(layer):
                # Set Output Gain:
                if self._g_output <= 0.0: output_gain = torch.nn.init.calculate_gain('linear')
                else: output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b_output)
                torch.nn.init.xavier_normal_(layer.weight, output_gain)

        elif self._w_init == "orthogonal":

            # Construct initialization function:
            def init_hidden(layer):
                # Set the Hidden Gain:
                if self._g_hidden <= 0.0: hidden_gain = torch.nn.init.calculate_gain('relu')
                else: hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b_hidden)
                torch.nn.init.orthogonal_(layer.weight, hidden_gain)

            def init_output(layer):
                # Set Output Gain:
                if self._g_output <= 0.0: output_gain = torch.nn.init.calculate_gain('linear')
                else: output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b_output)
                torch.nn.init.orthogonal_(layer.weight, output_gain)

        elif self._w_init == "sparse":
            assert self._p_sparse < 1. and self._p_sparse >= 0.0

            # Construct initialization function:
            def init_hidden(layer):
                p_non_zero = self._p_sparse
                hidden_std = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b_hidden)
                torch.nn.init.sparse_(layer.weight, p_non_zero, hidden_std)

            def init_output(layer):
                p_non_zero = self._p_sparse
                output_std = self._g_output

                torch.nn.init.constant_(layer.bias, self._b_output)
                torch.nn.init.sparse_(layer.weight, p_non_zero, output_std)

        else:
            raise ValueError("Weight Initialization Type must be in ['xavier_normal', 'orthogonal', 'sparse'] "
                             "but is {0}".format(self._w_init))

        # Create Network:
        self.layers = nn.ModuleList()

        # Create Input Layer:
        self.layers.append(DifferentialLayer(self.n_network, self.n_input, self.n_width, activation=non_linearity))
        init_hidden(self.layers[-1])

        # Create Hidden Layer:
        for _ in range(1, self.n_hidden):
            self.layers.append(DifferentialLayer(self.n_network, self.n_width, self.n_width, activation=non_linearity))
            init_hidden(self.layers[-1])

        # Create output Layer:
        self.layers.append(DifferentialLayer(self.n_network, self.n_width, self.n_output, activation='Linear'))
        init_output(self.layers[-1])

        self._eye = torch.eye(self.n_input).view(1, self.n_input, self.n_input)
        self._zeros = torch.zeros(1, self.n_input, self.n_input, self.n_input)
        self.device = self._eye.device

    def forward(self, x, hessian=False):
        x = x.view(1, -1, self.n_input, 1)

        # Create initial derivative of dq/ dq.
        dx_dx = self._eye.repeat(1, x.shape[1], 1, 1)
        d2x_d2x = self._zeros.repeat(1, x.shape[1], 1, 1, 1)

        # Compute the Network:
        x, dx_dx, d2x_d2x = self.layers[0](x, dx_dx, d2x_d2x, hessian=hessian)
        for i in range(1, len(self.layers)):
            x, dx_dx, d2x_d2x = self.layers[i](x, dx_dx, d2x_d2x, hessian=hessian)

        out = (x, dx_dx, d2x_d2x) if hessian else (x, dx_dx)
        return out

    def cuda(self, device=None):

        # Move the Network to the GPU:
        super(DifferentialNetwork, self).cuda(device=device)

        # Move the eye matrix to the GPU:
        self._eye = self._eye.cuda()
        self._zeros = self._zeros.cuda()

        self.device = self._eye.device
        return self

    def cpu(self):

        # Move the Network to the CPU:
        super(DifferentialNetwork, self).cpu()

        # Move the eye matrix to the CPU:
        self._eye = self._eye.cpu()
        self._zeros = self._zeros.cpu()
        self.device = self._eye.device
        return self

