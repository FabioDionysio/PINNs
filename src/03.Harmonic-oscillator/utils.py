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


class fcn(nn.Module):
    """Defines a connected network
    """
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        """Initializes the network
        Args:
            n_input (_int_): number of inputs
            n_output (_int_): number of outputs
            n_hidden (_int_): number of hidden neurons
            n_layers (_int_): number of layers
        """
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[nn.Linear(n_input, n_hidden), activation()])
        self.fch = nn.Sequential(*[nn.Sequential(*[nn.Linear(n_hidden, n_hidden), activation()]) for _ in range(n_layers-1)])
        self.fce = nn.Linear(n_hidden, n_output)

    def forward (self, x):
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
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)
