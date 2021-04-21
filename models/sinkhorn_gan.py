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
    fake_real = sinkhorn_divergence(fake, real, 'euclidean', parameters, device, critic)
    real_real = sinkhorn_divergence(real, real, 'euclidean', parameters, device, critic)
    fake_fake = sinkhorn_divergence(fake, fake, 'euclidean', parameters, device, critic)

    loss = 2 * fake_real - real_real - fake_fake

    return loss


def train_sinkhorn_gan(data_loader, generator, critic, optimizer_g, optimizer_c, parameters, device):

    generator_losses = []
    critic_losses = []

    generator.to(device)
    if critic is not None:
        critic.to(device)

    for epoch in range(parameters.n_epochs):

        epoch_generator_loss = 0
        epoch_critic_loss = 0

        for batch_id, (real_data, _) in enumerate(data_loader):

            real_data = real_data.to(device)

            # Generate fake data from sampled random noise
            z = torch.rand(parameters.batch_size, parameters.latent_dim).to(device)
            fake_data = generator(z)

            if batch_id % (parameters.critic_steps + 1) != 0:
                # Update critic (cost function)
                optimizer_c.zero_grad()

                # Compute Sinkhorn loss
                critic_loss = - sinkhorn_loss(fake_data, real_data, parameters, device, critic)
                critic_loss.backward()
                epoch_critic_loss += critic_loss.item()

                # Take an optimization step
                nn.utils.clip_grad_norm_(critic.parameters(), parameters.clipping_value)  # Clip gradient norm
                optimizer_c.step()

            else:
                # Update generator
                optimizer_g.zero_grad()

                # Compute Sinkhorn loss
                generator_loss = sinkhorn_loss(fake_data, real_data, parameters, device, critic)
                generator_loss.backward()
                epoch_generator_loss += generator_loss.item()

                # Take an optimization step
                optimizer_g.step()

        # Compute and store epoch losses
        epoch_generator_loss /= len(data_loader.dataset)
        epoch_critic_loss /= len(data_loader.dataset)
        generator_losses.append(epoch_generator_loss)
        critic_losses.append(epoch_critic_loss)
        if parameters.display:
            print(
                "[Epoch %d/%d] [Generator loss: %f] [Critic loss: %f]" % (epoch + 1, parameters.n_epochs,
                                                                                float(epoch_generator_loss),
                                                                                float(epoch_critic_loss))
            )

    # Save models
    torch.save(generator.state_dict(), os.path.join(parameters.output_path, 'sinkhorn_gan_generator.pth'))
    if critic is not None:
        torch.save(critic.state_dict(), os.path.join(parameters.output_path, 'sinkhorn_gan_critic.pth'))

    return generator, critic, generator_losses, critic_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--display", type=bool, default=True, help="true to display training results")
    parser.add_argument("--output_path", type=str, default='saved_models', help="path where the trained models"
                                                                                       "will be saved")

    # Network parameters

    # Model parameters
    parser.add_argument("--entropy_regularization", type=float, default=1, help="entropy regularization parameter")
    parser.add_argument("--sinkhorn_iterations", type=int, default=10, help="number of Sinkhorn iterations")
    parser.add_argument("--latent_dim", type=int, default=2, help="dimension of the latent space")

    # Training parameters
    parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--critic_steps", type=float, default=5, help="number of critic optimization steps")
    parser.add_argument("--clipping_value", type=float, default=5, help="clipping value for the critic gradient")

    # Build parser
    params = parser.parse_args()

    # Create data loader
    mnist = torchvision.datasets.MNIST(os.path.join(os.path.dirname(os.getcwd()), 'data'), train=True,
                                       transform=mnist_transforms)
    data_loader = DataLoader(mnist, batch_size=params.batch_size, shuffle=True)

    # Instantiate model and optimizer
    generator = Generator(params.latent_dim)
    generator.load_state_dict(torch.load('./saved_models/sinkhorn_gan_generator_100_epochs.pth'))
    critic = None
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=params.learning_rate)
    optimizer_c = None
    if critic is not None:
        optimizer_c = torch.optim.Adam(critic.parameters(), lr=params.learning_rate)
    else:
        params.critic_steps = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train model
    training_results = train_sinkhorn_gan(data_loader, generator, critic, optimizer_g, optimizer_c, params, device)

    end = True