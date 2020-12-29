import argparse
import torch
import numpy as np
import time

import matplotlib as mp
try: mp.use("Qt5Agg")
except: pass

mp.rc('text', usetex=True)
mp.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch.nn as nn

from deep_differential_network.differential_hessian_network_ensemble import DifferentialNetwork
from deep_differential_network.replay_memory import PyTorchReplayMemory
from deep_differential_network.utils import jacobian, hessian, jacobian_auto

LOAD_MODEL = False
RENDER = True
SAVE_MODEL = True
SAVE_PLOT = False

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=2, linewidth=500, formatter={'float_kind': lambda x: "{0:+08.4f}".format(x)})

    # Read Command Line Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, required=False, default=[True, ], help="Training using CUDA.")
    parser.add_argument("-i", nargs=1, type=int, required=False, default=[0, ], help="Specifies the CUDA id.")
    parser.add_argument("-s", nargs=1, type=int, required=False, default=[0, ], help="Specifies the random seed")
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
             'n_depth': 1,
             'n_minibatch': 256,
             'n_network': 20,
             'learning_rate': 1.0e-03,
             'weight_decay': 1.e-6,
             'activation': "Tanh"}

    filename = f"1d_{hyper['activation']}_diff_hessian_net_ensemble"

    # Parameters:
    n_dof = 1
    max_epoch = 50
    n_train_samples = hyper["n_minibatch"] * 100
    n_test_samples = 100
    std_x = 0.001
    std_y = 0.001

    print("\n\n################################################")
    print("Data:")
    print("")

    x_train = np.random.uniform(-0.75 * np.pi, 0.75 * np.pi, n_train_samples)[:, np.newaxis]
    x_test = np.linspace(-np.pi, np.pi, n_test_samples, endpoint=True)[:, np.newaxis]

    y_train = np.sin(x_train)
    y_test = np.sin(x_test)

    dydx_train = np.cos(x_train)
    dydx_test = np.cos(x_test)
    d2yd2x_train = - np.sin(x_train)
    d2yd2x_test = - np.sin(x_test)

    x_train_noisy = x_train + np.random.normal(0.0, std_x, (n_train_samples, 1))
    y_train_noisy = y_train + np.random.normal(0.0, std_y, (n_train_samples, 1))
    dydx_train_noisy = dydx_train + np.random.normal(0.0, std_y, (n_train_samples, 1))

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
        if i == 0: print("Using the both the f(x) and df(x)/dx as supervising feedback\n")
        if i == 1: print("Using the only f(x) as supervising feedback\n")

        # Construct Training Network:
        t0_net = time.perf_counter()

        # Load existing model parameters:
        if LOAD_MODEL:
            load_file = f"./models/{filename}_loss_{i:01d}.torch"
            state = torch.load(load_file, map_location='cpu')

            diff_net = DifferentialNetwork(n_dof, **state['hyper'])
            diff_net.load_state_dict(state['state_dict'])

        else:
            diff_net = DifferentialNetwork(n_dof, **hyper)

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

        mem = PyTorchReplayMemory(int(1e6), hyper["n_minibatch"], ((1, ), (1, ), (1, )), cuda_flag)
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
            l_mem_mean = 0.0
            l_mem_var = 0.0
            n_batches = 0.0

            for x_i, y_i, dydx_i in mem:
                t0_batch = time.perf_counter()

                # Reset gradients:
                optimizer.zero_grad()

                # Compute f(x)
                y_i_hat, dydx_i_hat = diff_net(x_i)

                # Compute the loss:
                if i == 0:
                    # Using the both the f(x) and df(x)/dx as supervising feedback:
                    l2_err = torch.sum((y_i_hat - y_i.view(1, -1, y_i.shape[1], 1)) ** 2, dim=[0, 2]) + \
                             torch.sum((dydx_i_hat - dydx_i.view(1, -1, dydx_i.shape[1], 1)) ** 2, dim=[0, 2])

                elif i == 1:
                    # Using the only f(x) as supervising feedback:
                    l2_err = torch.sum((y_i_hat - y_i.view(1, -1, y_i.shape[1], 1)) ** 2, dim=[0, 2])

                else:
                    raise RuntimeError

                l_mean = torch.mean(1./hyper['n_ensemble'] * l2_err)
                l_var = torch.var(1./hyper['n_ensemble'] * l2_err)

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

        time.sleep(1.)
        print("\n################################################")
        print("Evaluating Derivative:")
        t0_batch = time.perf_counter()

        with torch.no_grad():
            # Convert NumPy samples to torch:
            x_torch = torch.from_numpy(x_test).float().to(diff_net.device)
            y_hat, dydx_hat, d2yd2x_hat = diff_net(x_torch, hessian=True)
            y_hat = y_hat.cpu().numpy()[:, :, 0]
            dydx_hat = dydx_hat.cpu().numpy()[:, :, 0]
            d2yd2x_hat = d2yd2x_hat.squeeze(-2).cpu().numpy()[:, :, 0]

        t_batch = (time.perf_counter() - t0_batch) / (float(x_test.shape[0]))

        # Compute Errors:
        err_y = 1. / float(x_test.shape[0]) * np.sum((y_hat - y_test) ** 2)
        err_dydx = 1. / float(x_test.shape[0]) * np.sum((dydx_hat - dydx_test) ** 2)
        err_d2yd2x = 1. / float(x_test.shape[0]) * np.sum((d2yd2x_hat - d2yd2x_test) ** 2)

        print("\nPerformance:")
        print("        y MSE = {0:.3e}".format(err_y))
        print("    dy/dx MSE = {0:.3e}".format(err_dydx))
        print("d^2y/d^2x MSE = {0:.3e}".format(err_d2yd2x))

        plot_y_test[i] = y_hat
        plot_dydx_test[i] = dydx_hat
        plot_d2yd2x_test[i] = d2yd2x_hat

        print("\n################################################")
        print("Autograd Performance:")

        ## Autograd Test:
        f_diff_net = lambda x: diff_net(x)[0]
        dfdx_diff_net = lambda x: diff_net(x)[1]
        d2fd2x_diff_net = lambda x: diff_net(x, hessian=True)[2]

        with torch.no_grad():
            t0_jac = time.perf_counter()
            dydx_hat = dfdx_diff_net(x_torch)
            t_for_jac = time.perf_counter() - t0_jac
            dydx_hat = dydx_hat.detach().cpu().numpy().squeeze()

            t0_jac = time.perf_counter()
            d2yd2x_hat = d2fd2x_diff_net(x_torch)
            t_for_hes = time.perf_counter() - t0_jac
            d2yd2x_hat = d2yd2x_hat.detach().cpu().numpy().squeeze()

        t0_jac = time.perf_counter()
        dydx_autograd[i] = jacobian(f_diff_net, x_torch, create_graph=False, v1=False).transpose(0, 1).view(hyper['n_ensemble'], -1)
        t_rev_jac = time.perf_counter() - t0_jac
        dydx_autograd[i] = dydx_autograd[i].detach().cpu().numpy().squeeze()

        t0_hes = time.perf_counter()
        d2yd2x_autograd[i] = jacobian(dfdx_diff_net, x_torch, create_graph=False, v1=False).transpose(0, 1).view(hyper['n_ensemble'], -1)
        t_rev_hes = time.perf_counter() - t0_hes
        d2yd2x_autograd[i] = d2yd2x_autograd[i].detach().cpu().numpy().squeeze()

        seq_net = nn.Sequential(nn.Linear(1, hyper['n_width']), nn.Tanh(), nn.Linear(hyper['n_width'], 1)).cuda()
        t0_auto = time.perf_counter()
        dydx_auto_test = jacobian_auto(seq_net, x_torch, create_graph=True)
        t_test = time.perf_counter() - t0_auto
        dydx_auto_test = dydx_auto_test.detach().cpu().numpy().squeeze()

        n_norm = (float(x_test.shape[0]) * hyper['n_ensemble'])
        err_dydx = 1. / n_norm * np.sum((dydx_test - plot_dydx_test[i][0]) ** 2)
        err_d2yd2x = 1. / n_norm * np.sum((d2yd2x_test - plot_d2yd2x_test[i]) ** 2)
        err_dydx_autograd = 1. / n_norm * np.sum((dydx_hat - dydx_autograd[i]) ** 2)
        err_d2yd2x_autograd = 1. / n_norm * np.sum((d2yd2x_hat - d2yd2x_autograd[i]) ** 2)

        print(f"Jacobian:\n"
              f"          Approximation MSE = {err_dydx:.3e}\n"
              f"              Diff Mode MSE = {err_dydx_autograd:.3e}\n"
              f"               Reverse Diff = {t_rev_jac:.3e}s\n"
              f"               Forward Diff = {t_for_jac:.3e}s\n"
              f"               Test    Diff = {t_test:.3e}s\n"
              f"                   Speed Up = {t_rev_jac / t_for_jac:.2f}x")

        print("")
        print(f"Hessian:\n"
              f"          Approximation MSE = {err_d2yd2x:.3e}\n"
              f"              Diff Mode MSE = {err_d2yd2x_autograd:.3e}\n"
              f"               Reverse Diff = {t_rev_hes:.3e}s\n"
              f"               Forward Diff = {t_for_hes:.3e}s\n"
              f"                   Speed Up = {t_rev_hes / t_for_hes:.2f}x")

    print("\n################################################")
    print("Plotting Performance:")

    # Plot the performance:
    plt.rc('text', usetex=True)

    fig = plt.figure(figsize=(15.0/1.54, 12.0/1.54), dpi=100)
    left, bottom, right, top = 0.1, 0.1, 0.98, 0.95
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=0.3, hspace=0.25)
    # fig.subplots_adjust(left=0.07, bottom=0.15, right=0.75, top=0.95, wspace=0.3, hspace=0.2)

    legend = [mp.patches.Patch(color="k", label="Ground Truth"),
              mp.patches.Patch(color="silver", label="Noisy Training Samples"),
              mp.patches.Patch(color="r", label=r"DiffNet, $l_2$-loss with $y$ \& $dy/dx$"),
              mp.patches.Patch(color="m", label=r"DiffNet, $l_2$-loss with $y$")]

    ticks = [-np.pi, -3.*np.pi/4, -np.pi/2., -np.pi/4., 0.0, np.pi/4., np.pi/2., 3.*np.pi / 4., np.pi]
    tick_label = [r"$-\pi$", r"$-3\pi /4$", r"$-\pi/2$", r"$-\pi/4$", r"$0$", r"$+\pi/4$", r"$+\pi/2$", r"$+3\pi/4$", r"$+\pi$"]

    ax0 = fig.add_subplot(3, 1, 1)
    ax0.set_ylabel(r"$y = \text{sin}(x)$", fontsize=12)
    ax0.yaxis.set_label_coords(-0.075, 0.5)
    ax0.set_ylim(-1.5, +1.5)
    ax0.set_xlim(-np.pi, np.pi)
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(tick_label)

    ax1 = fig.add_subplot(3, 1, 2)
    ax1.set_ylabel(r"$\partial y / \partial x = \text{cos}(x)$", fontsize=12)
    ax1.yaxis.set_label_coords(-0.075, 0.5)
    ax1.set_xlabel(r"$x$")
    ax1.set_ylim(-1.5, +1.5)
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(tick_label)

    ax2 = fig.add_subplot(3, 1, 3)
    ax2.set_ylabel(r"$\partial^2 y / \partial^2 x = -\text{sin}(x)$", fontsize=12)
    ax2.yaxis.set_label_coords(-0.075, 0.5)
    ax2.set_xlabel(r"$x$")
    ax2.set_ylim(-1.5, +1.5)
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(tick_label)

    ax2.legend(handles=legend, bbox_to_anchor=(0.5, -0.20), loc='upper center', ncol=4, framealpha=0.)

    # Plot Ground Truth:
    ax0.plot(x_test, y_test, color="k")
    ax1.plot(x_test, dydx_test, color="k")
    ax2.plot(x_test, d2yd2x_test, color="k")

    ax0.scatter(x_train_noisy, y_train_noisy, color="silver")
    ax1.scatter(x_train_noisy, dydx_train_noisy, color="silver")

    # Plot Differential Network Prediction:
    ax0.plot(x_test, plot_y_test[0][:, :, 0].transpose(), color="r")
    ax0.plot(x_test, plot_y_test[1][:, :, 0].transpose(), color="m")

    ax1.plot(x_test, plot_dydx_test[0][:, :, 0].transpose(), color="r")
    ax1.plot(x_test, plot_dydx_test[1][:, :, 0].transpose(), color="m")

    ax2.plot(x_test, plot_d2yd2x_test[0][:, :, 0].transpose(), color="r")
    ax2.plot(x_test, plot_d2yd2x_test[1][:, :, 0].transpose(), color="m")

    if RENDER:
        plt.show()

    if SAVE_PLOT:
        fig.savefig(f"./figures/{filename}.pdf", format="pdf")
        fig.savefig(f"./figures/{filename}.png", format="png")

    print("\n################################################\n\n")
