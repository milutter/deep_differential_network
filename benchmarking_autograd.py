import time
import torch
from torch.autograd.functional import jacobian as torch_jacobian
from torch.autograd.functional import hessian as torch_hessian
from deep_differential_network.utils import jacobian, hessian


if __name__ == "__main__":
    torch.set_num_threads(1)

    n_samples, n_dim, n_iter = 100, 3, 50
    x = torch.empty(n_samples, n_dim, 1).uniform_(-5, +5)
    A = torch.empty(1, n_dim, n_dim).uniform_(-1, +1)

    def fun(input):
        input = input.unsqueeze(0) if input.ndim == 2 else input
        out = torch.matmul(input.transpose(dim0=1, dim1=2), torch.matmul(A, input))
        return out

    def jacobian_fun(input):
        input = input.unsqueeze(0) if input.ndim == 2 else input
        out = torch.matmul(A + A.transpose(dim0=1, dim1=2), input)
        return out

    def hessian_fun(input):
        out = (A + A.transpose(dim0=1, dim1=2)).repeat(input.shape[0], 1, 1)
        return out

    y, dydx, d2yd2x = fun(x), jacobian_fun(x), hessian_fun(x)

    t0 = time.perf_counter()
    for i in range(n_iter):
        auto_dydx = jacobian(fun, x)

    t_jacobian = (time.perf_counter() - t0) / n_iter

    t0 = time.perf_counter()
    for i in range(n_iter):
        auto_d2yd2x = jacobian(jacobian_fun, dydx)
    t_hessian = (time.perf_counter() - t0) / n_iter

    t0 = time.perf_counter()
    for i in range(n_iter):
        auto_d2yd2x = hessian(fun, dydx)
    t_auto_hessian = (time.perf_counter() - t0) / n_iter

    t0 = time.perf_counter()
    for i in range(n_iter):
        idx = torch.arange(0, x.shape[0], 1, dtype=int)
        torch_dydx = torch_jacobian(fun, x, create_graph=False)[idx, :, 0, idx, :, 0].transpose(1, 2)
    t_torch_jacobian = (time.perf_counter() - t0) / n_iter

    t0 = time.perf_counter()
    for i in range(n_iter):
        idx = torch.arange(0, x.shape[0], 1, dtype=int)
        torch_d2yd2x = torch_jacobian(jacobian_fun, x, create_graph=False)[idx, :, 0, idx, :, 0].transpose(1, 2)
    t_torch_hessian = (time.perf_counter() - t0) / n_iter

    print("\nDifference:")
    print(f"Diff     dy/dx = {torch.sum((dydx-auto_dydx)**2):.3e}/{torch.sum((dydx-torch_dydx)**2):.3e} Jacobian Time = {t_jacobian:.3e}s Torch Time = {t_torch_jacobian:.3e}s")
    print(f"Diff d^2y/d^2x = {torch.sum((d2yd2x - auto_d2yd2x) ** 2):.3e}/{torch.sum((d2yd2x-torch_d2yd2x)**2):.3e} Jacobian Time = {t_hessian:.3e}s Torch Time = {t_torch_hessian:.3e}s Hessian Time = {t_auto_hessian:.3e}s ")
