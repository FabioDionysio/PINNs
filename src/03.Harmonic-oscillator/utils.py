import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt


def oscillator(d, w0, x):
    """Analytical solution to the 1D underdamped harmonic oscillator. 

    Args:
        d (int): delta
        w0 (float): frequency
        x (torch.tensor): input
    """
    assert d < w0
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d / w)
    A = 1 / (2 * np.cos(phi))
    cos = torch.cos(phi + w*x)
    sin = torch.sin(phi + w*x)
    exp = torch.exp(-d*x)
    y = exp * 2 * A * cos
    return y


def oscillator2d(d, w0, t, x1_0, v1_0, x2_0, v2_0):
    """Analytical solution to the 2D underdamped harmonic oscillator.

    Args:
        d (float): damping coefficient (delta)
        w0 (float): natural frequency
        t (torch.tensor): time input
        x1_0 (float): initial displacement in x1 direction
        v1_0 (float): initial velocity in x1 direction
        x2_0 (float): initial displacement in x2 direction
        v2_0 (float): initial velocity in x2 direction
    """
    assert d < w0

    # Calculate the damped angular frequency
    w = np.sqrt(w0**2 - d**2)

    # Calculate phase and amplitude for x1
    phi1 = np.arctan((v1_0 + d * x1_0) / (w * x1_0))
    phi1 = torch.tensor(phi1)
    A1 = x1_0 / torch.cos(phi1)

    # Calculate phase and amplitude for x2
    phi2 = np.arctan((v2_0 + d * x2_0) / (w * x2_0))
    phi2 = torch.tensor(phi2)
    A2 = x2_0 / torch.cos(phi2)

    # Calculate the solutions for x1 and x2
    x1 = A1 * torch.exp(-d * t) * torch.cos(w * t - phi1)
    x2 = A2 * torch.exp(-d * t) * torch.cos(w * t - phi2)

    return x1, x2


class OscillatorFCN(nn.Module):
    """Defines a fully connected network for the 2D damped harmonic oscillator"""

    def __init__(self, n_input, n_output, n_hidden, n_layers):
        """Initializes the network
        Args:
            n_input (int): number of inputs
            n_output (int): number of outputs
            n_hidden (int): number of hidden neurons
            n_layers (int): number of layers
        """
        super(OscillatorFCN, self).__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[nn.Linear(n_input, n_hidden), activation()])
        self.fch = nn.Sequential(
            *[nn.Sequential(*[nn.Linear(n_hidden, n_hidden), activation()]) for _ in range(n_layers-1)])
        self.fce = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


class fcn(nn.Module):
    """Defines a connected network
    """

    def __init__(self, n_input, n_output, n_hidden, n_layers):
        """Initializes the network
        Args:
            n_input (int): number of inputs
            n_output (int): number of outputs
            n_hidden (int): number of hidden neurons
            n_layers (int): number of layers
        """
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[nn.Linear(n_input, n_hidden), activation()])
        self.fch = nn.Sequential(
            *[nn.Sequential(*[nn.Linear(n_hidden, n_hidden), activation()]) for _ in range(n_layers-1)])
        self.fce = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


def save_gif_PIL(outfile, files, fps=5, loop=0):
    """Function for saving GIFs

    Args:
        outfile (gif): output file
        files (png): images to be saved
        fps (int, optional): frame rate. Defaults to 5.
        loop (int, optional): Defaults to 0.
    """
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:],
                 save_all=True, duration=int(1000/fps), loop=loop)


def plot_result(x, y, x_data, y_data, yh, xp=None, step=None):
    """Pretty plot training results

    Args:
        x (torch.tensor): input
        y (torch.tensor): output
        x_data (torch.tensor): slice of input
        y_data (torch.tensor): slice of output
        yh (torch.tensor): neural network prediction
        xp (torch.tensor, optional): optional tensor for physics loss training locations. Defaults to None.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label='Analytical solution',
             linestyle='--', color='tab:grey', alpha=0.8)
    plt.plot(x, yh, label='Neural network prediction',
             linewidth=4, color='tab:blue', alpha=0.8)
    plt.scatter(x_data, y_data, label='Training data',
                s=60, alpha=0.4, color='tab:red')
    if xp is not None:
        plt.scatter(xp, -0*torch.ones_like(xp),
                    label='Physics loss training locations', s=60, alpha=0.4, color="tab:green")
    l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.5, 1.5)
    plt.text(
        1.065, 0.7, f'Training step: {step}', fontsize='xx-large', color='k')
    plt.axis('off')


def plot_result2d(t, x1, x2, t_data, x1_data, x2_data, x1_pred, x2_pred, xp=None, step=None):
    """Pretty plot training results for 2D oscillator

    Args:
        t (torch.tensor): time input
        x1 (torch.tensor): analytical solution for x1
        x2 (torch.tensor): analytical solution for x2
        t_data (torch.tensor): slice of time input
        x1_data (torch.tensor): slice of output for x1
        x2_data (torch.tensor): slice of output for x2
        x1_pred (torch.tensor): neural network prediction for x1
        x2_pred (torch.tensor): neural network prediction for x2
        xp (torch.tensor, optional): optional tensor for physics loss training locations. Defaults to None.
        step (int, optional): current training step. Defaults to None.
    """
    plt.figure(figsize=(10, 4))
    # x1
    plt.plot(t.numpy(), x1.numpy(), label='Analytical solution $x_1(t)$',
             linestyle='--', color='tab:blue', alpha=0.5)
    plt.scatter(t_data.numpy(), x1_data.numpy(),
                label='Training data $x_1(t)$', s=60, alpha=0.4, color='tab:blue')
    plt.plot(t.numpy(), x1_pred.numpy(), label='Neural network prediction $x_1(t)$',
             linewidth=4, color='tab:blue', alpha=0.8)
    # x2
    plt.plot(t.numpy(), x2.numpy(), label='Analytical solution $x_2(t)$',
             linestyle='--', color='tab:green', alpha=0.5)
    plt.scatter(t_data.numpy(), x2_data.numpy(
    ), label='Training data $x_2(t)$', s=60, alpha=0.4, color='tab:green')
    plt.plot(t.numpy(), x2_pred.numpy(), label='Neural network prediction $x_2(t)$',
             linewidth=4, color='tab:green', alpha=0.8)

    if xp is not None:
        plt.scatter(xp.numpy(), -0 * torch.ones_like(xp).numpy(),
                    label='Physics loss training locations', s=60, alpha=0.4, color="tab:red")

    l = plt.legend(loc=(1.01, 0.22), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k", fontsize="medium")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.5, 1.5)
    plt.text(
        1.065, 1.15, f'Training step: {step}', fontsize='xx-large', color='k')
    plt.axis('off')
