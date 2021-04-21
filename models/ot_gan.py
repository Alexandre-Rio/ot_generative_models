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
    x_y = sinkhorn_divergence(x, y, 'cosine', parameters, device, critic)
    x_xp = sinkhorn_divergence(x, x_prime, 'cosine', parameters, device, critic)
    y_yp = sinkhorn_divergence(y, y_prime, 'cosine', parameters, device, critic)
    x_yp = sinkhorn_divergence(x, y_prime, 'cosine', parameters, device, critic)
    xp_y = sinkhorn_divergence(x_prime, y, 'cosine', parameters, device, critic)
    xp_yp = sinkhorn_divergence(x_prime, y_prime, 'cosine', parameters, device, critic)

    loss = x_y + x_yp + xp_y + xp_yp - 2 * x_xp - 2 * y_yp

    return loss


def train_ot_gan(data_loader, generator, critic, optimizer_g, optimizer_c, parameters, device):

    generator_losses = []
    critic_losses = []

    generator.to(device)
    critic.to(device)

    for epoch in range(parameters.n_epochs):

        epoch_generator_loss = 0
        epoch_critic_loss = 0

        for batch_id, (real_data, _) in enumerate(data_loader):

            real_data = real_data.to(device)

            # Split batch to have two independent samples
            x, x_prime = torch.split(real_data, parameters.batch_size)

            # Generate two independent samples of fake data from sampled random noise
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
                epoch_critic_loss += critic_loss.item()

                # Take an optimization step
                optimizer_c.step()

            else:
                # Update generator
                optimizer_g.zero_grad()

                # Compute Minibatch Energy Distance
                generator_loss = minibatch_energy_distance(x, x_prime, y, y_prime, parameters, device, critic)
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
    torch.save(generator.state_dict(), os.path.join(parameters.output_path, 'ot_gan_generator.pth'))
    if critic is not None:
        torch.save(critic.state_dict(), os.path.join(parameters.output_path, 'ot_gan_critic.pth'))

    return generator, critic, generator_losses, critic_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--display", type=bool, default=True, help="true to display training results")
    parser.add_argument("--output_path", type=str, default='saved_models', help="path where the trained models"
                                                                                       "will be saved")
    parser.add_argument("--dataset", type=str, default='gaussian', help="dataset to use (mnist or gaussian)")

    # Network parameters

    # Model parameters
    parser.add_argument("--entropy_regularization", type=float, default=1, help="entropy regularization parameter")
    parser.add_argument("--sinkhorn_iterations", type=int, default=10, help="number of Sinkhorn iterations")
    parser.add_argument("--latent_dim", type=int, default=2, help="dimension of the latent space")

    # Training parameters
    parser.add_argument("--n_epochs", type=int, default=30, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--beta_1", type=float, default=0.5, help="1st beta coefficient for Adam optimizer")
    parser.add_argument("--beta_2", type=float, default=0.999, help="2nd beta coefficient for Adam optimizer")
    parser.add_argument("--generator_steps", type=float, default=3, help="number of generator optimization steps")

    # Build parser
    params = parser.parse_args()

    # Create data loader. Batch size is multiplied by 2 to simulate the sampling of 2 independent batches by splitting
    # one batch at each step.
    data_loader = None
    if params.dataset == 'mnist':
        mnist = torchvision.datasets.MNIST(os.path.join(os.path.dirname(os.getcwd()), 'data'), train=True,
                                        transform=mnist_transforms)
        data_loader = DataLoader(mnist, batch_size=2*params.batch_size, shuffle=True)
    elif params.dataset == 'gaussian':
        gaussian_toy = GaussianToy()
        toy_dataset = gaussian_toy.build()
        data_loader = DataLoader(toy_dataset, batch_size=2*params.batch_size, shuffle=True)

    # Instantiate model and optimizer
    generator, critic = None, None
    if params.dataset == 'mnist':
        generator = ConvGenerator(params.latent_dim)
        critic = ConvCritic()
    elif params.dataset == 'gaussian':
        generator = Generator(input_dim=params.latent_dim, hidden_dim=256, output_dim=2)
        critic = Critic(input_dim=2, hidden_dim=256, output_dim=1)
    #generator.load_state_dict(torch.load('./saved_models/ot_gan_generator_15_epochs.pth'))
    #critic.load_state_dict(torch.load('./saved_models/ot_gan_critic_15_epochs.pth'))
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=params.learning_rate,
                                   betas=(params.beta_1, params.beta_2))
    optimizer_c = torch.optim.Adam(critic.parameters(), lr=params.learning_rate,
                                   betas=(params.beta_1, params.beta_2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train model
    training_results = train_ot_gan(data_loader, generator, critic, optimizer_g, optimizer_c, params, device)

    end = True