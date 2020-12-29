import torch
import torch.nn as nn
import numpy as np

class SoftplusDer(nn.Module):
    def __init__(self, beta=1.):
        super(SoftplusDer, self).__init__()
        self._beta = beta

    def forward(self, x):
        cx = torch.clamp(x, -10., 10.)
        exp_x = torch.exp(self._beta * cx)
        out = exp_x / (exp_x + 1.0)

        if torch.isnan(out).any():
            print("SoftPlus Jacobian output is NaN.")

            if torch.isnan(cx).any():
                print("Input is already NaN.")

        return out


class SoftplusDer2(nn.Module):
    def __init__(self, beta=1.):
        super(SoftplusDer2, self).__init__()
        self._beta = beta

    def forward(self, x):
        cx = torch.clamp(x, -20., 20.)
        exp_x = torch.exp(self._beta * cx)
        out = exp_x / (exp_x + 1.0)**2

        if torch.isnan(out).any():
            print("SoftPlus Hessian output is NaN.")
        return out


class ReLUDer(nn.Module):
    def __init__(self):
        super(ReLUDer, self).__init__()

    def forward(self, x):
        return torch.ceil(torch.clamp(x, 0, 1))


class ReLUDer2(nn.Module):
    def __init__(self):
        super(ReLUDer2, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 0, 0)


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, x):
        return x


class LinearDer(nn.Module):
    def __init__(self):
        super(LinearDer, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 1, 1)


class LinearDer2(nn.Module):
    def __init__(self):
        super(LinearDer2, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 0, 0)

class Cos(nn.Module):
    def __init__(self):
        super(Cos, self).__init__()

    def forward(self, x):
        return torch.cos(2. * np.pi * x)


class CosDer(nn.Module):
    def __init__(self):
        super(CosDer, self).__init__()

    def forward(self, x):
        return -2. * np.pi * torch.sin(2. * np.pi * x)


class CosDer2(nn.Module):
    def __init__(self):
        super(CosDer2, self).__init__()

    def forward(self, x):
        return -4. * np.pi**2 *torch.cos(2. * np.pi * x)


class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return torch.tanh(x)


class TanhDer(nn.Module):
    def __init__(self):
        super(TanhDer, self).__init__()

    def forward(self, x):
        return 1. - torch.tanh(x)**2


class TanhDer2(nn.Module):
    def __init__(self):
        super(TanhDer2, self).__init__()

    def forward(self, x):
        tanh = torch.tanh(x)
        return -2. * tanh * (1. - tanh**2)


class Quad(nn.Module):
    def __init__(self):
        super(Quad, self).__init__()

    def forward(self, x):
        return 0.5 * torch.clamp(x, min=0)**2


class QuadDer(nn.Module):
    def __init__(self):
        super(QuadDer, self).__init__()

    def forward(self, x):
        return torch.clamp(x, min=0)


class QuadDer2(nn.Module):
    def __init__(self):
        super(QuadDer2, self).__init__()

    def forward(self, x):
        return torch.ceil(torch.clamp(x, 0, 1))


class Cubic(nn.Module):
    def __init__(self):
        super(Cubic, self).__init__()

    def forward(self, x):
        return 1./6. * torch.clamp(x, min=0)**3


class CubicDer(nn.Module):
    def __init__(self):
        super(CubicDer, self).__init__()

    def forward(self, x):
        return 0.5 * torch.clamp(x, min=0)**2


class CubicDer2(nn.Module):
    def __init__(self):
        super(CubicDer2, self).__init__()

    def forward(self, x):
        return torch.clamp(x, min=0)