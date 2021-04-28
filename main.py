"""
Created on Mon Apr 19 14:41:00 2021

@author: alexandre-rio
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import argparse
import numpy as np
import matplotlib.pyplot as plt

from data_preprocessing import mnist_transforms

from architectures import *
from models.sinkhorn_gan import *
from models.ot_gan import *
from simulated_data import GaussianToy

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--model", type=str, default="ot_gan", help="model to use (sinkhorn gan or ot_gan)")
parser.add_argument("--architecture", type=str, default="conv", help="architecture to use (simple or conv)")
parser.add_argument("--use_critic", type=bool, default=True, help="True if a learnable critic is used")
parser.add_argument("--trained_generator", type=str, default='', help="path to trained generator")
parser.add_argument("--trained_critic", type=str, default='', help="path to trained critic")
parser.add_argument("--seed", type=int, default=0, help="seed")
parser.add_argument("--display", type=bool, default=True, help="true to display training results")
parser.add_argument("--output_path", type=str, default='models/saved_models', help="path where the trained models"
                                                                                        "will be saved")
parser.add_argument("--dataset", type=str, default='mnist', help="dataset to use (mnist, cifar or gaussian)")
parser.add_argument("--patience", type=int, default=5, help="patience for early stopping")
parser.add_argument("--checkpoints", type=int, nargs='*', help="epochs at which to make a checkpoint")

# Network parameters
parser.add_argument("--hidden_dim", type=int, default=500, help="number of nodes in the MLPs hidden layer")
parser.add_argument("--critic_out_dim", type=int, default=2, help="dimension of the MLP critic out features")

# Model parameters
parser.add_argument("--entropy_regularization", type=float, default=1, help="entropy regularization parameter")
parser.add_argument("--sinkhorn_iterations", type=int, default=10, help="number of Sinkhorn iterations")
parser.add_argument("--latent_dim", type=int, default=50, help="dimension of the latent space")
parser.add_argument("--latent_space", type=str, default='uniform', help="type of latent space (uniform or gaussian)")
parser.add_argument("--data_dim", type=int, help="dimension of the data (flattened)")
parser.add_argument("--distance", type=str, default='default', help="distance to use for the critic "
                                                                    "(default, cosine or euclidean)")

# Training parameters
parser.add_argument("--n_epochs", type=int, default=200, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=200, help="batch size")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--beta_1", type=float, default=0.5, help="1st beta coefficient for Adam optimizer")
parser.add_argument("--beta_2", type=float, default=0.999, help="2nd beta coefficient for Adam optimizer")
parser.add_argument("--critic_steps", type=float, default=3, help="number of critic optimization steps")
parser.add_argument("--generator_steps", type=float, default=3, help="number of generator optimization steps")
parser.add_argument("--clipping_value", type=float, default=3, help="clipping value for the critic gradient")

# Build parser
params = parser.parse_args()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(params, device):
    """
    Create and train an OT-based model from given parameters, and returns the trained model and losses.
    """
    # Print information
    print('__________________\nMODEL PARAMETERS\n__________________')
    print('Dataset: {}'.format(params.dataset))
    print('Model: {}'.format(params.model))
    print('Use Critic: {}'.format(params.use_critic))
    print('Architecture: {}'.format(params.architecture))
    print('____________________\nTRAINING PARAMETERS\n____________________')
    print('Number of epochs: {}'.format(params.n_epochs))
    print('Batch size: {}'.format(params.batch_size))
    print('Learning rate: {}'.format(params.learning_rate))
    print('Adam Beta: ({}, {})'.format(params.beta_1, params.beta_2))
    print('Sinkhorn iterations: {}'.format(params.sinkhorn_iterations))
    print('Entropy regularization: {}'.format(params.entropy_regularization))

    data_loader = None
    actual_batch_size = 2 * params.batch_size if params.model == 'ot_gan' else params.batch_size
    if params.dataset == 'mnist':
        mnist = torchvision.datasets.MNIST('./data', train=True, transform=mnist_transforms, download=True)
        data_loader = DataLoader(mnist, batch_size=actual_batch_size, shuffle=True)
        params.data_dim = 1024
    elif params.dataset == 'gaussian':
        gaussian_toy = GaussianToy()
        toy_dataset = gaussian_toy.build()
        data_loader = DataLoader(toy_dataset, batch_size=actual_batch_size, shuffle=True)
        params.data_dim = 2
    elif params.dataset == 'cifar':
        cifar = torchvision.datasets.CIFAR10('./data', train=True, transform=mnist_transforms, download=True)
        data_loader = DataLoader(cifar, batch_size=actual_batch_size, shuffle=True)
        params.data_dim = 1024

    # Instantiate model
    generator, critic = None, None
    if params.architecture == 'conv':
        generator = ConvGenerator(params.latent_dim, mode=params.dataset)
        if params.use_critic:
            critic = ConvCritic(mode=params.dataset)
    elif params.architecture == 'simple':
        generator = Generator(input_dim=params.latent_dim, hidden_dim=params.hidden_dim, output_dim=params.data_dim,
                              mode=params.dataset)
        if params.use_critic:
            critic = Critic(input_dim=params.data_dim, hidden_dim=params.hidden_dim, output_dim=params.critic_out_dim,
                            mode=params.dataset)

    # Load model if stated
    if params.trained_generator != '':
        generator.load_state_dict(torch.load(params.trained_generator))
    if params.trained_critic != '':
        critic.load_state_dict(torch.load(params.trained_critic))

    # Instantiate optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=params.learning_rate,
                                   betas=(params.beta_1, params.beta_2))
    if critic is not None:
        optimizer_c = torch.optim.Adam(critic.parameters(), lr=params.learning_rate,
                                       betas=(params.beta_1, params.beta_2))
    else:
        params.critic_steps = 0
        optimizer_c = None

    print('_____________________\nMODEL CREATION: SUCCESS\n_____________________')

    print('____________________\nSTARTING TRAINING\n____________________')
    # Train model
    if params.model == 'ot_gan':
        training_results = train_ot_gan(data_loader, generator, critic, optimizer_g, optimizer_c, params, device)
    elif params.model == 'sinkhorn_gan':
        training_results = train_sinkhorn_gan(data_loader, generator, critic, optimizer_g, optimizer_c, params, device)
    else:
        training_results = None

    return training_results


if __name__ == '__main__':

    # Create and train model
    training_results = main(params, device)

    # Plot loss
    loss = training_results[2]
    plt.title("Training loss")
    plt.plot(np.arange(len(training_results[2])) + 1, np.array(loss))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    end = True