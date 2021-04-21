"""
Created on Mon Apr 19 14:41:00 2021

@author: alexandre-rio
"""

__all__ = ['plot_save_grid', 'pairwise_cosine_distance', 'CReLU', 'sinkhorn_divergence']

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# PLOTS


def plot_save_grid(output, img_size=32, size=8, save=False):
    """
    Plot 8x8 grid to visualize results from the output of the model
    """
    # Process output
    if output.is_cuda:
        output = output.cpu()
    output = output.detach()
    output = output.view(-1, 1, img_size, img_size)
    output = output.numpy().transpose((0, 2, 3, 1))
    output = np.clip(output, 0, 1)
    # Plot figures
    fig = plt.figure(figsize=(size, size))
    fig.suptitle("Generated digits")
    gridspec = fig.add_gridspec(size, size)
    for idx in range(size ** 2):
        ax = fig.add_subplot(gridspec[idx])
        ax.imshow(output[idx], cmap='gray')
        ax.set_axis_off()
    if save:
        fig.savefig('images/digits_mnist.png')


# MATHS


def pairwise_cosine_distance(x, y):
    """
    Compute the pairwise batch cosine distance between two batches x and y.
    :param x, y: batches of samples (n_samples x d)
    :return: the pairwise cosine distance matrix (n_samples, n_samples)
    """
    # L2 normalize batches x and y
    x_norm, y_norm = torch.norm(x, dim=1, keepdim=True), torch.norm(y, dim=1, keepdim=True)
    x, y = x / x_norm, y / y_norm

    # Compute pairwise cosine distance
    cost = 1 - torch.matmul(x, y.transpose(0, 1))

    return cost


# ACTIVATIONS


class CReLU(nn.Module):
    """"
    Implementation of the Concatenated Rectified Linear Unit (CReLU).
    Source: https://github.com/pytorch/pytorch/issues/1327
    """
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), 1)


# OPTIMAL TRANSPORT


def sinkhorn_divergence(x, y, distance, parameters, device, critic=None, flat_size=1024):
    """
    Compute the Sinkhorn divergence between batches x and y, where x and y are assumed to have the same size,
    using Sinkhorn algorithm.
    :param x, y: two batches of samples (n_samples x d)
    :param distance: transport distance to use ('euclidean' or 'cosine') (str)
    :param parameters: a parser containing the number of iterations for the Sinkhorn algorithm and
    the entropy regularization value
    :param critic: a learnable cost with NN representation. None if fixed L2 cost.
    :param flat_size: flat size of the input if critic is None.
    :return: the Sinkhorn divergence between x and y
    """
    if critic is not None:
        x, y = critic(x), critic(y)
    else:
        x, y = x.view(-1, flat_size), y.view(-1, flat_size)

    # Compute pairwise transport cost matrix
    if distance == 'cosine':
        cost = pairwise_cosine_distance(x, y)
    elif distance == 'euclidean':
        cost = torch.cdist(x, y)
    else:
        cost = None

    kernel = torch.exp(- cost / parameters.entropy_regularization)

    n = x.shape[0]
    a = None
    b = torch.ones(n).to(device)
    ones = torch.ones(n).to(device) / n

    for iteration in range(parameters.sinkhorn_iterations):
        a = ones / torch.matmul(kernel, b)
        b = ones / torch.matmul(kernel, a)

    w = torch.dot(torch.matmul(kernel * cost, b), a)

    return w
