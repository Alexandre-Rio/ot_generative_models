"""
Created on Mon Apr 19 14:41:00 2021

@author: alexandre-rio
"""

__all__ = ['minibatch_energy_distance', 'train_ot_gan']

import argparse
import os
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
import torchvision

from torch.utils.data import DataLoader
from early_stopping_pytorch.pytorchtools import EarlyStopping

from architectures import *
from data_preprocessing import mnist_transforms
from utils import *
from simulated_data import GaussianToy


def minibatch_energy_distance(x, x_prime, y, y_prime, parameters, device, critic):
    """
    Compute the Sinkhorn loss between a batch of real data and a batch of fake data, using the Sinkhorn divergence.
    :param x, x_prime: independent batches of real samples (n_samples x d)
    :param y, y_prime: independent batches of generated samples (n_samples x d)
    :return: the Sinkhorn loss between the two batches
    """
    if parameters.distance == 'default':
        distance = 'cosine'
    else:
        distance = parameters.distance

    x_y = sinkhorn_divergence(x, y, distance, parameters, device, critic)
    x_xp = sinkhorn_divergence(x, x_prime, distance, parameters, device, critic)
    y_yp = sinkhorn_divergence(y, y_prime, distance, parameters, device, critic)
    x_yp = sinkhorn_divergence(x, y_prime, distance, parameters, device, critic)
    xp_y = sinkhorn_divergence(x_prime, y, distance, parameters, device, critic)
    xp_yp = sinkhorn_divergence(x_prime, y_prime, distance, parameters, device, critic)

    loss = x_y + x_yp + xp_y + xp_yp - 2 * x_xp - 2 * y_yp

    return loss


def train_ot_gan(data_loader, generator, critic, optimizer_g, optimizer_c, parameters, device):

    losses = []

    # Initialize Early Stopping callback
    early_stopping = EarlyStopping(patience=parameters.patience, verbose=True)

    generator.to(device)
    critic.to(device)

    for epoch in range(parameters.n_epochs):

        epoch_loss = 0

        for batch_id, (real_data, _) in enumerate(data_loader):

            real_data = real_data.to(device)

            # Split batch to have two independent samples
            x, x_prime = torch.split(real_data, parameters.batch_size)

            # Generate two independent samples of fake data from sampled random noise
            if parameters.latent_space == 'gaussian':
                z = torch.randn(parameters.batch_size, parameters.latent_dim)
                z_prime = torch.randn(parameters.batch_size, parameters.latent_dim)
            else:
                z = 2 * torch.rand(parameters.batch_size, parameters.latent_dim) - 1
                z_prime = 2 * torch.rand(parameters.batch_size, parameters.latent_dim) - 1
            z, z_prime = z.to(device), z_prime.to(device)
            y, y_prime = generator(z), generator(z_prime)

            if batch_id % (parameters.generator_steps + 1) == 0:
                # Update critic (cost function)
                optimizer_c.zero_grad()

                # Compute Minibatch Energy Distance
                critic_loss = - minibatch_energy_distance(x, x_prime, y, y_prime, parameters, device, critic)
                critic_loss.backward()
                epoch_loss += critic_loss.item()

                # Take an optimization step
                optimizer_c.step()

            else:
                # Update generator
                optimizer_g.zero_grad()

                # Compute Minibatch Energy Distance
                generator_loss = minibatch_energy_distance(x, x_prime, y, y_prime, parameters, device, critic)
                generator_loss.backward()
                epoch_loss += generator_loss.item()

                # Take an optimization step
                optimizer_g.step()

        # Compute and store epoch losses
        epoch_loss /= len(data_loader.dataset)
        losses.append(epoch_loss)
        if parameters.display:
            print(
                "[Epoch %d/%d] [Loss: %f]" % (epoch + 1, parameters.n_epochs, float(epoch_loss))
            )

        # Early stopping if validation loss increases
        early_stopping(epoch_loss, generator)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if epoch in parameters.checkpoints:
            torch.save(generator.state_dict(), os.path.join(parameters.output_path,
                                                            'sinkhorn_gan_generator_cp' + str(epoch) + 'epochs.pth'))

    # load the last checkpoint with the best model
    generator.load_state_dict(torch.load('checkpoint.pt'))

    # Save models
    torch.save(generator.state_dict(), os.path.join(parameters.output_path, 'ot_gan_generator.pth'))
    if critic is not None:
        torch.save(critic.state_dict(), os.path.join(parameters.output_path, 'ot_gan_critic.pth'))

    return generator, critic, losses
