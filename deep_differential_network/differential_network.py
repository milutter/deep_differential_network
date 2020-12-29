import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from deep_differential_network.activations import *


class DifferentialLayer(nn.Module):

    def __init__(self, input_size, output_size, activation="ReLu"):
        super(DifferentialLayer, self).__init__()

        # Create layer weights and biases:
        self.n_output = output_size
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))

        # Initialize activation function and its derivative:
        if activation == "ReLu":
            self.g = nn.ReLU()
            self.g_prime = ReLUDer()

        elif activation == "SoftPlus":
            self.softplus_beta = 1.0
            self.g = nn.Softplus(beta=self.softplus_beta)
            self.g_prime = SoftplusDer(beta=self.softplus_beta)

        elif activation == "Cos":
            self.g = Cos()
            self.g_prime = CosDer()

        elif activation == "Linear":
            self.g = Linear()
            self.g_prime = LinearDer()

        elif activation == "Tanh":
            self.g = Tanh()
            self.g_prime = TanhDer()

        else:
            raise ValueError("Activation Type must be in ['Linear', 'ReLu', 'SoftPlus', 'Cos'] but is {0}".format(self.activation))

    def forward(self, x, der_prev):
        # Apply Affine Transformation:
        a = F.linear(x, self.weight, self.bias)
        out = self.g(a)
        der = torch.matmul(self.g_prime(a).view(-1, self.n_output, 1) * self.weight, der_prev)
        return out, der


class DifferentialNetwork(nn.Module):

    def __init__(self, n_input, **kwargs):
        super(DifferentialNetwork, self).__init__()

        # Read optional arguments:
        self.n_input = n_input
        self.n_width = kwargs.get("n_width", 128)
        self.n_hidden = kwargs.get("n_depth", 1)
        self.n_output = kwargs.get("n_output", 1)
        non_linearity = kwargs.get("activation", "ReLu")

        # Initialization of the layers:
        self._w_init = kwargs.get("w_init", "xavier_normal")
        self._b0 = kwargs.get("b_init", 0.1)
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

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, hidden_gain)

                with torch.no_grad():
                    layer.weight = torch.nn.Parameter(layer.weight)

            def init_output(layer):
                # Set Output Gain:
                if self._g_output <= 0.0: output_gain = torch.nn.init.calculate_gain('linear')
                else: output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, output_gain)

        elif self._w_init == "orthogonal":

            # Construct initialization function:
            def init_hidden(layer):
                # Set the Hidden Gain:
                if self._g_hidden <= 0.0: hidden_gain = torch.nn.init.calculate_gain('relu')
                else: hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, hidden_gain)

            def init_output(layer):
                # Set Output Gain:
                if self._g_output <= 0.0: output_gain = torch.nn.init.calculate_gain('linear')
                else: output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, output_gain)

        elif self._w_init == "sparse":
            assert self._p_sparse < 1. and self._p_sparse >= 0.0

            # Construct initialization function:
            def init_hidden(layer):
                p_non_zero = self._p_sparse
                hidden_std = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, p_non_zero, hidden_std)

            def init_output(layer):
                p_non_zero = self._p_sparse
                output_std = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, p_non_zero, output_std)

        else:
            raise ValueError("Weight Initialization Type must be in ['xavier_normal', 'orthogonal', 'sparse'] "
                             "but is {0}".format(self._w_init))

        # Create Network:
        self.layers = nn.ModuleList()

        # Create Input Layer:
        self.layers.append(DifferentialLayer(self.n_input, self.n_width, activation=non_linearity))
        init_hidden(self.layers[-1])

        # Create Hidden Layer:
        for _ in range(1, self.n_hidden):
            self.layers.append(DifferentialLayer(self.n_width, self.n_width, activation=non_linearity))
            init_hidden(self.layers[-1])

        # Create output Layer:
        self.layers.append(DifferentialLayer(self.n_width, self.n_output, activation="Linear"))
        init_output(self.layers[-1])

        self._eye = torch.eye(self.n_input).view(1, self.n_input, self.n_input)
        self.device = self._eye.device

    def forward(self, q):
        # Create initial derivative of dq/ dq.
        # qd_dq = self._eye.repeat(q.shape[0], 1, 1).type_as(q)
        qd_dq = self._eye.repeat(q.shape[0], 1, 1)

        # Compute the Network:
        qd, qd_dq = self.layers[0](q, qd_dq)
        for i in range(1, len(self.layers)):
            qd, qd_dq = self.layers[i](qd, qd_dq)

        return qd, qd_dq

    def cuda(self, device=None):

        # Move the Network to the GPU:
        super(DifferentialNetwork, self).cuda(device=device)

        # Move the eye matrix to the GPU:
        self._eye = self._eye.cuda()
        self.device = self._eye.device
        return self

    def cpu(self):

        # Move the Network to the CPU:
        super(DifferentialNetwork, self).cpu()

        # Move the eye matrix to the CPU:
        self._eye = self._eye.cpu()
        self.device = self._eye.device
        return self

