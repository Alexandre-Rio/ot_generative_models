"""
Created on Mon Apr 19 14:41:00 2021

@author: alexandre-rio
"""

__all__ = ['Generator', 'ConvGenerator', 'Critic', 'ConvCritic']

import torch.nn as nn
import torch
from utils import CReLU


class Generator(nn.Module):

    def __init__(self, input_dim, hidden_dim=500, output_dim=1024):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation1 = nn.LeakyReLU()
        self.activation2 = nn.Tanh()

    def forward(self, z):
        x = self.activation1(self.fc1(z))
        x = self.activation2(self.fc2(x))
        return x


class Critic(nn.Module):

    def __init__(self, input_dim=1024, hidden_dim=500, output_dim=256):
        super(Critic, self).__init__()

        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Flatten x
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvGenerator(nn.Module):

    def __init__(self, input_dim, output_channels=1):
        super(ConvGenerator, self).__init__()

        self.fc1 = nn.Linear(input_dim, 8192)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1)

        self.activation1 = nn.GLU(dim=1)
        self.activation2 = nn.Tanh()

    def forward(self, z):

        x = self.fc1(z)
        x = self.activation1(x)
        x = x.view(-1, 256, 4, 4)

        x = self.upsample(x)
        x = self.activation1(self.conv1(x))
        x = self.upsample(x)
        x = self.activation1(self.conv2(x))
        x = self.upsample(x)
        x = self.activation1(self.conv3(x))
        x = self.activation2(self.conv4(x))

        return x


class ConvCritic(nn.Module):

    def __init__(self, input_channels=1):
        super(ConvCritic, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.activation = CReLU()

    def forward(self, x):

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.view(-1, 8192)
        x = x / torch.norm(x, dim=1, keepdim=True)  # L2 normalization

        return x