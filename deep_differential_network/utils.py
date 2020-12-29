import torch
from torch.autograd.functional import jacobian as torch_jacobian
from torch.autograd.functional import hessian as torch_hessian


def evaluate(f, x, n_minibatch=1024):
    idx = torch.cat([torch.arange(0, x.shape[0], n_minibatch, dtype=torch.int),
                     torch.tensor([x.shape[0]], dtype=torch.int)], 0)

    n = idx.shape[0] - 1
    y = [f(x[idx[i]:idx[i+1]]) for i in range(n)]

    if isinstance(y[0], (list, tuple)) and y[0][0].ndim == 3:
        out = [torch.cat(yi) for yi in zip(*y)]

    elif isinstance(y[0], (list, tuple)) and y[0][0].ndim == 4:
        out = [torch.cat(yi, dim=1) for yi in zip(*y)]

    elif isinstance(y[0], (torch.Tensor, )) and y[0].ndim in (4, 5):
        out = torch.cat(y, dim=1)
    else:
        out = torch.cat(y)

    return out


def jacobian(func, inputs, create_graph=False, strict=False, v1=True):
    n_batch, n_dim = inputs.shape[0], inputs.shape[1]

    if v1:
        # Version 1: (This version is faster for computing the Jacobian)
        idx = torch.arange(0, n_batch, 1, dtype=int)
        out = torch_jacobian(func, inputs, create_graph=create_graph, strict=strict).view(n_batch, -1, n_batch, n_dim)
        out = out[idx, :, idx, :]

    else:
        # Version 2: (This version is faster for computing the Hessian using the Jacobian)
        inputs = inputs.view(inputs.shape[0], 1, inputs.shape[1]) if inputs.ndim == 2 else inputs.transpose(2, 1)
        out = [torch_jacobian(func, x.view(1, n_dim, 1), create_graph=create_graph, strict=strict).squeeze().unsqueeze(0) for x in inputs]
        out = torch.stack(out, 0)

    return out.transpose(dim0=1, dim1=2)


def hessian(func, inputs, create_graph=False, strict=False):

    # Version 1:
    out = [torch_hessian(func, x, create_graph=create_graph, strict=strict)[:, 0, :, 0] for x in inputs]
    out = torch.stack(out, 0)

    # Version 2: (super bad runtime)
    # idx = torch.arange(0, inputs.shape[0], 1, dtype=int)
    #
    # def jac_fun(x):
    #     out = torch_jacobian(func, x, create_graph=True, strict=strict)[idx, :, 0, idx, :, 0]
    #     return out.transpose(dim0=1, dim1=2)
    #
    # out = torch_jacobian(jac_fun, inputs, create_graph=create_graph, strict=strict)[idx, :, 0, idx, :, 0]

    return out.transpose(dim0=1, dim1=2)


def jacobian_auto(func, x, create_graph=True):
    with torch.set_grad_enabled(True):
        x = x.requires_grad_(True)
        jac = torch.autograd.grad(func(x).sum(), x, allow_unused=False, create_graph=create_graph)[0]

    return jac
