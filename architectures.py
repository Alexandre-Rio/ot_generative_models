"""
Created on Mon Apr 19 14:41:00 2021

@author: alexandre-rio
"""

__all__ = ['Generator', 'ConvGenerator', 'Critic', 'ConvCritic']

import torch.nn as nn
import torch
from utils import CReLU


class Generator(nn.Module):
    """
    A simple MLP architecture with one hidden layer.
    """
    def __init__(self, input_dim, hidden_dim=500, output_dim=1024, mode='mnist'):
        super(Generator, self).__init__()

        self.mode = mode

        if self.mode == 'mnist':
            # Define linear layers
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

            # Initialize linear layers with Xavier method
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)

            # Define activations
            self.activation1 = nn.LeakyReLU()
            self.activation2 = nn.Tanh()

        elif self.mode == 'gaussian':
            # Define linear layers
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, output_dim)

            # Initialize linear layers with Xavier method
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
            nn.init.xavier_normal_(self.fc3.weight)
            nn.init.xavier_normal_(self.fc4.weight)

            # Define activations
            self.activation1 = nn.ReLU()

    def forward(self, z):
        if self.mode == 'mnist':
            x = self.activation1(self.fc1(z))
            x = self.activation2(self.fc2(x))

        elif self.mode == 'gaussian':
            x = self.activation1(self.fc1(z))
            x = self.activation1(self.fc2(x))
            x = self.activation1(self.fc3(x))
            x = self.fc4(x)

        return x


class Critic(nn.Module):
    """
    A simple MLP architecture with one hidden layer.
    """
    def __init__(self, input_dim=1024, hidden_dim=500, output_dim=256, mode='mnist'):
        super(Critic, self).__init__()

        self.mode = mode
        self.input_dim = input_dim

        if self.mode == 'mnist':

            # Define linear layers
            self.fc1 = nn.Linear(self.input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

            # Initialize linear layers with Xavier method
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)

            # Define activations
            self.activation = nn.LeakyReLU()

        if self.mode == 'gaussian':
            hidden_dim = 32

            # Define linear layers
            self.fc1 = nn.Linear(self.input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)

            # Initialize linear layers with Xavier method
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
            nn.init.xavier_normal_(self.fc3.weight)

            # Define activations
            self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Flatten x

        if self.mode == 'mnist':
            x = self.activation(self.fc1(x))
            x = self.fc2(x)

        elif self.mode == 'gaussian':
            x = x / 4
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.fc3(x)

        return x


class ConvGenerator(nn.Module):
    """
    A convolutional network for image generation, inspired from DCGAN. Adapted from the architecture described in the
    OT-GAN paper to adapt to the simpler MNIST images.
    Mode: 'mnist' or 'cifar'. Defines the complexity of the architecture.
    """
    def __init__(self, input_dim, output_channels=None, mode='mnist'):
        super(ConvGenerator, self).__init__()

        self.mode = mode
        if mode == 'mnist':
            output_channels = 1
            self.dim1, self.dim2, self.dim3, self.dim4, self.dim5 = 8192, 256, 128, 64, 32
            self.kernel_size = 3
        elif mode == 'cifar':
            output_channels = 3
            self.dim1, self.dim2, self.dim3, self.dim4, self.dim5 = 32768, 1024, 512, 256, 128
            self.kernel_size = 5

        self.fc1 = nn.Linear(input_dim, self.dim1)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(self.dim2, self.dim2, kernel_size=self.kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.dim3, self.dim3, kernel_size=self.kernel_size, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.dim4, self.dim4, kernel_size=self.kernel_size, stride=1, padding=1)
        self.conv4 = nn.Conv2d(self.dim5, output_channels, kernel_size=self.kernel_size, stride=1, padding=1)

        self.activation1 = nn.GLU(dim=1)
        self.activation2 = nn.Tanh()

    def forward(self, z):

        x = self.fc1(z)
        x = self.activation1(x)
        x = x.view(-1, self.dim2, 4, 4)

        x = self.upsample(x)
        x = self.activation1(self.conv1(x))
        x = self.upsample(x)
        x = self.activation1(self.conv2(x))
        x = self.upsample(x)
        x = self.activation1(self.conv3(x))
        x = self.activation2(self.conv4(x))

        return x


class ConvCritic(nn.Module):
    """
    A convolutional network for image generation, inspired from DCGAN. Adapted from the architecture described in the
    OT-GAN paper to adapt to the simpler MNIST images.
    Mode: 'mnist' or 'cifar'. Defines the complexity of the architecture.
    """
    def __init__(self, input_channels=None, mode='mnist'):
        super(ConvCritic, self).__init__()

        self.mode = mode
        if mode == 'mnist':
            input_channels = 1
            self.dim1, self.dim2, self.dim3, self.dim4, self.dim5 = 32, 64, 128, 256, 8192
            self.kernel_size = 3
        elif mode == 'cifar':
            input_channels = 3
            self.dim1, self.dim2, self.dim3, self.dim4, self.dim5 = 256, 512, 1024, 2048, 32768
            self.kernel_size = 5

        self.conv1 = nn.Conv2d(input_channels, self.dim1, kernel_size=self.kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.dim2, self.dim2, kernel_size=self.kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(self.dim3, self.dim3, kernel_size=self.kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(self.dim4, self.dim4, kernel_size=self.kernel_size, stride=2, padding=1)

        self.activation = CReLU()

    def forward(self, x):

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.view(-1, self.dim5)
        x = x / torch.norm(x, dim=1, keepdim=True)  # L2 normalization

        return x
