"""
Created on Mon Apr 19 14:41:00 2021

@author: alexandre-rio
"""

__all__ = ['sinkhorn_loss', 'train_sinkhorn_gan']

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


def sinkhorn_loss(fake, real, parameters, device, critic):
    """
    Compute the Sinkhorn loss between a batch of real data and a batch of fake data, using the Sinkhorn divergence.
    :param fake: a batch of generated samples (n_samples x d)
    :param real: a batch of real samples (n_samples x d)
    :return: the Sinkhorn loss between the two batches
    """
    if parameters.distance == 'default':
        distance = 'euclidean'
    else:
        distance = parameters.distance

    fake_real = sinkhorn_divergence(fake, real, distance, parameters, device, critic)
    real_real = sinkhorn_divergence(real, real, distance, parameters, device, critic)
    fake_fake = sinkhorn_divergence(fake, fake, distance, parameters, device, critic)

    loss = 2 * fake_real - real_real - fake_fake

    return loss


def train_sinkhorn_gan(data_loader, generator, critic, optimizer_g, optimizer_c, parameters, device):

    losses = []

    # Initialize Early Stopping callback
    early_stopping = EarlyStopping(patience=parameters.patience, verbose=True)

    generator.to(device)
    if critic is not None:
        critic.to(device)

    for epoch in range(parameters.n_epochs):

        epoch_loss = 0

        for batch_id, (real_data, _) in enumerate(data_loader):

            real_data = real_data.to(device)

            # Generate fake data from sampled random noise
            if parameters.latent_space == 'gaussian':
                z = torch.randn(parameters.batch_size, parameters.latent_dim).to(device)
            else:
                z = torch.rand(parameters.batch_size, parameters.latent_dim).to(device)
            fake_data = generator(z)

            if batch_id % (parameters.critic_steps + 1) != 0:
                # Update critic (cost function)
                optimizer_c.zero_grad()

                # Compute Sinkhorn loss
                critic_loss = - sinkhorn_loss(fake_data, real_data, parameters, device, critic)
                critic_loss.backward()
                epoch_loss += critic_loss.item()

                # Take an optimization step
                nn.utils.clip_grad_norm_(critic.parameters(), parameters.clipping_value)  # Clip gradient norm
                optimizer_c.step()

            else:
                # Update generator
                optimizer_g.zero_grad()

                # Compute Sinkhorn loss
                generator_loss = sinkhorn_loss(fake_data, real_data, parameters, device, critic)
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
        if parameters.checkpoints is not None:
            if epoch in parameters.checkpoints:
                torch.save(generator.state_dict(), os.path.join(parameters.output_path,
                                                            'sinkhorn_gan_generator_cp' + str(epoch) + 'epochs.pth'))

    # load the last checkpoint with the best model
    generator.load_state_dict(torch.load('checkpoint.pt'))

    # Save models
    torch.save(generator.state_dict(), os.path.join(parameters.output_path, 'sinkhorn_gan_generator.pth'))
    if critic is not None:
        torch.save(critic.state_dict(), os.path.join(parameters.output_path, 'sinkhorn_gan_critic.pth'))

    return generator, critic, losses
