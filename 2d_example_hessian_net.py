import argparse
import torch
import numpy as np
import time
from matplotlib import cm
import matplotlib as mp
try: mp.use("Qt5Agg")
except: pass

mp.rc('text', usetex=True)
mp.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from deep_differential_network.differential_hessian_network import DifferentialNetwork
from deep_differential_network.replay_memory import PyTorchReplayMemory
from deep_differential_network.utils import jacobian, evaluate

LOAD_MODEL = False
RENDER = True
SAVE_MODEL = True
SAVE_PLOT = True


# Define the Function and the Jacobian:
def f(x):
    return np.cos(x[:, 0:1]) * np.sin(x[:, 1:2])


def jacobian_f(x):
    return np.hstack([-np.sin(x[:, 0:1]) * np.sin(x[:, 1:2]), np.cos(x[:, 0:1]) * np.cos(x[:, 1:2])])


def hessian_f(x):
    out = np.dstack([np.hstack([-np.cos(x[:, 0:1]) * np.sin(x[:, 1:2]), -np.sin(x[:, 0:1]) * np.cos(x[:, 1:2])]),
                     np.hstack([-np.sin(x[:, 0:1]) * np.cos(x[:, 1:2]), -np.cos(x[:, 0:1]) * np.sin(x[:, 1:2])])])

    return out

def shape_axis(ax):
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_ylim(lim)
    ax.set_xlim(lim)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_label)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_label)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    return ax


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=500, formatter={'float_kind': lambda x: "{0:+08.4f}".format(x)})

    # Read Command Line Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, required=False, default=[True, ], help="Training using CUDA.")
    parser.add_argument("-i", nargs=1, type=int, required=False, default=[0, ], help="Specifies the CUDA id.")
    parser.add_argument("-s", nargs=1, type=int, required=False, default=[42, ], help="Specifies the random seed")
    args = parser.parse_args()

    seed = args.s[0]
    cuda_flag = args.c[0] and torch.cuda.is_available()
    cuda_id = args.i[0]

    # Set the seed:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set the number of threads:
    torch.set_num_threads(12)

    # Construct Hyperparameters:
    # Activation must be in ['ReLu', 'SoftPlus']
    hyper = {'n_width': 128,
             'n_depth': 2,
             'n_minibatch': 128,
             'n_minibatch_eval': 300,
             'learning_rate': 1.0e-03,
             'weight_decay': 1.e-6,
             'activation': "Quad"}

    filename = f"2d_{hyper['activation']}_diff_hessian_net"

    # Parameters:
    n_input = 2
    max_epoch = 500
    n_train_samples = hyper["n_minibatch"] * 100
    n_test_samples = 50
    std_noise = 0.001
    lim = [-np.pi, +np.pi]

    print("\n################################################")
    print("Model:\n")
    print(f"    Dimension = {n_input:d} x [{hyper['n_width']:d}x{hyper['n_depth']}] x 1")
    print(f"Non-Linearity = {hyper['activation']}")

    print("\n################################################")
    print("Data:\n")

    x_train = np.random.uniform(lim[0], lim[1], (n_train_samples, n_input))

    di = np.linspace(lim[0], lim[1], n_test_samples, endpoint=True)
    x0_mat, x1_mat = np.meshgrid(di, di)

    x_test = np.vstack([x0_mat.reshape(np.prod(x0_mat.shape)), x1_mat.reshape(np.prod(x1_mat.shape))]).transpose()

    y_train, y_test = f(x_train), f(x_test)
    dydx_train, dydx_test = jacobian_f(x_train), jacobian_f(x_test)
    d2yd2x_train, d2yd2x_test = hessian_f(x_train), hessian_f(x_test)

    x_train_noisy = x_train + np.random.normal(0.0, std_noise, (n_train_samples, n_input))
    y_train_noisy = y_train + np.random.normal(0.0, std_noise, (n_train_samples, 1))
    dydx_train_noisy = dydx_train + np.random.normal(0.0, std_noise, (n_train_samples, n_input))

    # Set CUDA Device:
    if cuda_flag and torch.cuda.device_count() > 1:
        assert cuda_id < torch.cuda.device_count()
        torch.cuda.set_device(cuda_id)

    plot_y_test = [None, None]
    plot_dydx_test = [None, None]
    plot_d2yd2x_test = [None, None]
    dydx_autograd = [None, None]
    d2yd2x_autograd = [None, None]

    # Test different cost functions:
    for i in range(2):
        print("\n################################################")
        print("Creating & Training Differential Network:", end="\n")

        # Compute the loss of the inverse model:
        if i == 0: print("Using the both the f(x) and df(x)/dx as supervising feedback\n")
        if i == 1: print("Using the only f(x) as supervising feedback\n")

        # Construct Training Network:
        t0_net = time.perf_counter()

        # Load existing model parameters:
        if LOAD_MODEL:
            load_file = f"./models/{filename}_loss_{i:01d}.torch"
            state = torch.load(load_file, map_location='cpu')

            diff_net = DifferentialNetwork(n_input, **state['hyper'])
            diff_net.load_state_dict(state['state_dict'])

        else:
            diff_net = DifferentialNetwork(n_input, **hyper)

        if cuda_flag:
            diff_net.cuda()

        print("{0:30}: {1:05.2f}s".format("Initialize Network", time.perf_counter() - t0_net))

        # Generate & Initialize the Optimizer:
        t0_opt = time.perf_counter()
        optimizer = torch.optim.Adam(diff_net.parameters(),
                                     lr=hyper["learning_rate"],
                                     weight_decay=hyper["weight_decay"],
                                     amsgrad=True)

        print("{0:30}: {1:05.2f}s".format("Initialize Optimizer", time.perf_counter() - t0_opt))

        # Generate Replay Memory:
        t0_replay = time.perf_counter()

        mem = PyTorchReplayMemory(int(1e6), hyper["n_minibatch"], ((n_input,), (1,), (n_input,)), cuda_flag)
        mem.add_samples([x_train_noisy, y_train_noisy, dydx_train_noisy])
        print("{0:30}: {1:05.2f}s".format("Initialize Replay Memory", time.perf_counter() - t0_opt))

        # Start Training Loop:
        print("")
        alpha = 0.8
        epoch_i, t_opt = 0, 0.0

        t0_start = time.perf_counter()
        while epoch_i < max_epoch and not LOAD_MODEL:

            # Train network for an Epoch:
            t0_epoch = time.perf_counter()
            l_mem_mean, l_mem_var, n_batches = 0.0, 0.0, 0.0

            for x_i, y_i, dydx_i in mem:
                t0_batch = time.perf_counter()

                # Reset gradients:
                optimizer.zero_grad()

                # Compute the network output:
                y_i_hat, dydx_i_hat = diff_net(x_i)

                # Compute the loss of the inverse model:
                if i == 0:
                    # Using the both the f(x) and df(x)/dx as supervising feedback:
                    l2_err = torch.sum((y_i_hat[:, :, 0] - y_i) ** 2, dim=1) + \
                             torch.sum((dydx_i_hat[:, 0, :] - dydx_i) ** 2, dim=1)

                if i == 1:
                    # Using the only f(x) as supervising feedback:
                    l2_err = torch.sum((y_i_hat[:, :, 0] - y_i) ** 2, dim=1)

                l_mean, l_var = torch.mean(l2_err), torch.var(l2_err)

                # Compute gradients & update the weights:
                l_mean.backward()
                optimizer.step()

                # Update internal data:
                n_batches += 1
                l_mem_mean += l_mean.item()
                l_mem_var += l_var.item()
                t_batch = time.perf_counter() - t0_batch

            # Update Epoch Loss & Computation Time:
            epoch_i += 1
            l_mem_mean /= float(max(n_batches, 1))
            l_mem_var /= float(max(n_batches, 1))

            if epoch_i == 1: t_opt = (time.perf_counter() - t0_epoch)
            else: t_opt = alpha * t_opt + (1. - alpha) * (time.perf_counter() - t0_epoch)

            if epoch_i == 1 or np.mod(epoch_i, 10) == 0:
                print("Epoch {0:04d}: ".format(epoch_i), end="")
                print("\tComp Time = {0:08.3f}s".format(time.perf_counter() - t0_start), end="")
                print("\tTrain Loss = {0:.3e} \u00B1 {1:.3e}".format(l_mem_mean, 1.96 * np.sqrt(l_mem_var)))

        # Save the Model:
        if SAVE_MODEL and not LOAD_MODEL:
            torch.save({"epoch": epoch_i,
                        "hyper": hyper,
                        "loss": (l_mem_mean, l_mem_var),
                        "state_dict": diff_net.state_dict()},
                       f"./models/{filename}_loss_{i:01d}.torch")

        print("\n################################################")
        print("Evaluating Performance:\n")
        t0_batch = time.perf_counter()

        with torch.no_grad():
            # Convert NumPy samples to torch:
            x_torch = torch.from_numpy(x_test).float().to(diff_net.device)

            fun = lambda x: diff_net(x, hessian=True)
            y_hat, dydx_hat, d2yd2x_hat = evaluate(fun, x_torch, n_minibatch=hyper['n_minibatch_eval'])

            y_hat = y_hat.cpu().numpy()
            dydx_hat = dydx_hat.transpose(dim0=1, dim1=2).cpu().numpy().squeeze()
            d2yd2x_hat = d2yd2x_hat.cpu().numpy().squeeze()

        t_batch = (time.perf_counter() - t0_batch) / (float(x_test.shape[0]))

        # Compute Errors:
        err_y = 1. / float(x_test.shape[0]) * np.sum((y_hat[:, :, 0] - y_test) ** 2)
        err_dydx = 1. / float(x_test.shape[0]) * np.sum((dydx_hat - dydx_test) ** 2)
        err_d2yd2x = 1. / float(x_test.shape[0]) * np.sum((d2yd2x_hat - d2yd2x_test) ** 2)

        print(f"Performance:")
        print(f"       y MSE = {err_y:.3e}")
        print(f"   dy/dx MSE = {err_dydx:.3e}")
        print(f" d2y/d2x MSE = {err_d2yd2x:.3e}")

        plot_y_test[i] = y_hat
        plot_dydx_test[i] = dydx_hat
        plot_d2yd2x_test[i] = d2yd2x_hat

        print("\n################################################")
        print("Autograd Performance:\n")

        ## Autograd Test:
        f_diff_net = lambda x: diff_net(x)[0]
        dfdx_diff_net = lambda x: diff_net(x)[1]
        d2fd2x_diff_net = lambda x: diff_net(x, hessian=True)[2]

        with torch.no_grad():
            t0_jac = time.perf_counter()
            dydx_hat = dfdx_diff_net(x_torch).detach().cpu().numpy().squeeze()
            t_for_jac = time.perf_counter() - t0_jac

            t0_hes = time.perf_counter()
            d2yd2x_hat = evaluate(d2fd2x_diff_net, x_torch, n_minibatch=hyper['n_minibatch_eval']).detach().cpu().numpy().squeeze()
            t_for_hes = time.perf_counter() - t0_hes

        t0_jac = time.perf_counter()
        dydx_autograd[i] = jacobian(f_diff_net, x_torch, create_graph=False, v1=False).detach().cpu().numpy().squeeze()
        t_rev_jac = time.perf_counter() - t0_jac

        t0_hes = time.perf_counter()
        d2yd2x_autograd[i] = jacobian(dfdx_diff_net, x_torch, create_graph=False, v1=False).detach().cpu().numpy().squeeze()
        t_rev_hes = time.perf_counter() - t0_hes

        err_dydx = 1. / float(x_test.shape[0]) * np.sum((dydx_test - plot_dydx_test[i]) ** 2)
        err_d2yd2x = 1. / float(x_test.shape[0]) * np.sum((d2yd2x_test - d2yd2x_hat) ** 2)
        err_dydx_autograd = 1. / float(x_test.shape[0]) * np.sum((dydx_hat - dydx_autograd[i]) ** 2)
        err_d2yd2x_autograd = 1. / float(x_test.shape[0]) * np.sum((d2yd2x_hat - d2yd2x_autograd[i]) ** 2)

        print(f"Jacobian:\n"
              f"          Approximation MSE = {err_dydx:.3e}\n"
              f"              Diff Mode MSE = {err_dydx_autograd:.3e}\n"
              f"               Reverse Diff = {t_rev_jac:.3e}s\n"
              f"               Forward Diff = {t_for_jac:.3e}s\n"
              f"                   Speed Up = {t_rev_jac/t_for_jac:.2f}x")

        print("")
        print(f"Hessian:\n"
              f"          Approximation MSE = {err_d2yd2x:.3e}\n"
              f"              Diff Mode MSE = {err_d2yd2x_autograd:.3e}\n"
              f"               Reverse Diff = {t_rev_hes:.3e}s\n"
              f"               Forward Diff = {t_for_hes:.3e}s\n"
              f"                   Speed Up = {t_rev_hes/t_for_hes:.2f}x")

    print("\n################################################")
    print("Plotting Performance:")

    # Plot the performance:
    plt.rc('text', usetex=True)
    n_levels = 100
    cmap = cm.Spectral
    norm = cm.colors.Normalize(vmax=+1.0, vmin=-1.0)

    fig = plt.figure(figsize=(30.0/1.54, 12.0/1.54))
    left, bottom, right, top = 0.05, 0.06, 0.935, 0.95
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=0.3, hspace=0.15)
    ticks = [-np.pi, -np.pi/2., 0.0, np.pi/2., np.pi]
    tick_label = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$+\pi/2$", r"$+\pi$"]

    ###################################################################################################################
    # Plot the Model trained on f(x) & df(x)/dx:
    ax0 = shape_axis(fig.add_subplot(3, 6, 1))
    cset0 = ax0.contourf(x0_mat, x1_mat, y_test.reshape(x0_mat.shape),
                         levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    ax1 = shape_axis(fig.add_subplot(3, 6, 2))
    cset1 = ax1.contourf(x0_mat, x1_mat, dydx_test[:, 0].reshape(x0_mat.shape),
                         levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    ax2 = shape_axis(fig.add_subplot(3, 6, 3))
    cset2 = ax2.contourf(x0_mat, x1_mat, dydx_test[:, 1].reshape(x0_mat.shape),
                         levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    ax3 = shape_axis(fig.add_subplot(3, 6, 4))
    cset3 = ax3.contourf(x0_mat, x1_mat, d2yd2x_test[:, 0, 0].reshape(x0_mat.shape),
                         levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    ax4 = shape_axis(fig.add_subplot(3, 6, 5))
    cset4 = ax4.contourf(x0_mat, x1_mat, d2yd2x_test[:, 1, 1].reshape(x0_mat.shape),
                         levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    ax5 = shape_axis(fig.add_subplot(3, 6, 6))
    cset5 = ax5.contourf(x0_mat, x1_mat, d2yd2x_test[:, 1, 0].reshape(x0_mat.shape),
                         levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    # Add colorbar:
    cbaxes = fig.add_axes([right + 0.02, ax0.get_position().bounds[1], 0.01, ax0.get_position().bounds[3]])
    plt.colorbar(cset0, cax=cbaxes, ticks=[-1.0, -0.5, 0.0, +0.5, +1.0])

    # Add title:
    ax0.set_title(r"$f(x, y) = \text{cos}(x)\text{sin}(y)$")
    ax1.set_title(r"$\partial f(x, y) / \partial x = -\text{sin}(x)\text{sin}(y)$")
    ax2.set_title(r"$\partial f(x, y) / \partial y = \text{cos}(x)\text{cos}(y)$")
    ax3.set_title(r"$\partial^2 f(x, y) / \partial^2 x = -\text{cos}(x)\text{sin}(y)$")
    ax4.set_title(r"$\partial^2 f(x, y) / \partial^2 y = -\text{cos}(x)\text{sin}(y)$")
    ax5.set_title(r"$\partial^2 f(x, y) / \partial x \partial y = -\text{sin}(x)\text{cos}(y)$")

    ax0.text(-0.32, 0.5, "Ground Truth",
             size=12, transform=ax0.transAxes, ha='center', va="center", rotation="vertical")

    ###################################################################################################################
    # Plot the Model trained on f(x) & df(x)/dx:
    ax0 = shape_axis(fig.add_subplot(3, 6, 7))
    cset0 = ax0.contourf(x0_mat, x1_mat, plot_y_test[0].reshape(x0_mat.shape),
                        levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    ax1 = shape_axis(fig.add_subplot(3, 6, 8))
    cset1 = ax1.contourf(x0_mat, x1_mat, plot_dydx_test[0][:, 0].reshape(x0_mat.shape),
                        levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    ax2 = shape_axis(fig.add_subplot(3, 6, 9))
    cset2 = ax2.contourf(x0_mat, x1_mat, plot_dydx_test[0][:, 1].reshape(x0_mat.shape),
                        levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    ax3 = shape_axis(fig.add_subplot(3, 6, 10))
    cset3 = ax3.contourf(x0_mat, x1_mat, d2yd2x_autograd[0][:, 0, 0].reshape(x0_mat.shape),
                         levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    ax4 = shape_axis(fig.add_subplot(3, 6, 11))
    cset4 = ax4.contourf(x0_mat, x1_mat, d2yd2x_autograd[0][:, 1, 1].reshape(x0_mat.shape),
                         levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    ax5 = shape_axis(fig.add_subplot(3, 6, 12))
    cset5 = ax5.contourf(x0_mat, x1_mat, d2yd2x_autograd[0][:, 1, 0].reshape(x0_mat.shape),
                         levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    # Add colorbar:
    cbaxes = fig.add_axes([right + 0.02, ax2.get_position().bounds[1], 0.01, ax2.get_position().bounds[3]])
    plt.colorbar(cset0, cax=cbaxes, ticks=[-1.0, -0.5, 0.0, +0.5, +1.0])

    # Add Title:
    ax0.text(-0.32, 0.5, r"$l_2$-loss" + "\n" + r"with $y$ \& $\partial y/\partial x$",
            size=12, transform=ax0.transAxes, ha='center', va="center", rotation="vertical")

    ###################################################################################################################
    # Plot the Model trained on f(x):
    ax0 = shape_axis(fig.add_subplot(3, 6, 13))
    cset0 = ax0.contourf(x0_mat, x1_mat, plot_y_test[1].reshape(x0_mat.shape),
                        levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    ax1 = shape_axis(fig.add_subplot(3, 6, 14))
    cset1 = ax1.contourf(x0_mat, x1_mat, plot_dydx_test[1][:, 0].reshape(x0_mat.shape),
                         levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    ax2 = shape_axis(fig.add_subplot(3, 6, 15))
    cset2 = ax2.contourf(x0_mat, x1_mat, plot_dydx_test[1][:, 1].reshape(x0_mat.shape),
                        levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    ax3 = shape_axis(fig.add_subplot(3, 6, 16))
    cset3 = ax3.contourf(x0_mat, x1_mat, d2yd2x_autograd[1][:, 0, 0].reshape(x0_mat.shape),
                         levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    ax4 = shape_axis(fig.add_subplot(3, 6, 17))
    cset4 = ax4.contourf(x0_mat, x1_mat, d2yd2x_autograd[1][:, 1, 1].reshape(x0_mat.shape),
                         levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    ax5 = shape_axis(fig.add_subplot(3, 6, 18))
    cset5 = ax5.contourf(x0_mat, x1_mat, d2yd2x_autograd[1][:, 1, 0].reshape(x0_mat.shape),
                        levels=n_levels, norm=norm, cmap=cm.get_cmap(cmap, n_levels))

    # Add Colorbar:
    cbaxes = fig.add_axes([right + 0.02, ax2.get_position().bounds[1], 0.01, ax2.get_position().bounds[3]])
    plt.colorbar(cset0, cax=cbaxes, ticks=[-1.0, -0.5, 0.0, +0.5, +1.0])

    # Add Title:
    ax0.text(-0.32, 0.5, r"$l_2$-loss" + "\n" + r"with $y$",
             size=12, transform=ax0.transAxes, ha='center', va="center", rotation="vertical")

    if RENDER:
        plt.show()

    if SAVE_PLOT:
        fig.savefig(f"./figures/{filename}.pdf", format="pdf")
        fig.savefig(f"./figures/{filename}.png", format="png")

    print("\n################################################\n\n")
