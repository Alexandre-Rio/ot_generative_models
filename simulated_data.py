"""
Created on Mon Apr 19 14:41:00 2021

@author: alexandre-rio
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset

cd = np.sqrt(2) / 2


class GaussianToy:

    def __init__(self, samples_per_mode=10000):

        self.samples_per_mode = samples_per_mode
        self.centers = np.array([[0, 1], [1, 0], [-1, 0], [0, -1], [cd, cd], [-cd, -cd], [-cd, cd], [cd, -cd]])
        self.std = 0.05
        self.data = np.zeros((len(self.centers) * self.samples_per_mode, 2))
        self.labels = np.zeros(len(self.centers) * self.samples_per_mode)
        self.dataset = None

    def build(self):

        for i in range(len(self.centers)):
            center = self.centers[i]
            samples = center + self.std * np.random.randn(self.samples_per_mode, 2)

            self.data[np.arange(i * self.samples_per_mode, (i + 1) * self.samples_per_mode), :] = samples
            self.labels[np.arange(i * self.samples_per_mode, (i + 1) * self.samples_per_mode)] = i

        self.dataset = TensorDataset(torch.Tensor(self.data), torch.Tensor(self.labels))

        return self.dataset

if __name__ == '__main__':

    gaussian_toy = GaussianToy()
    toy_dataset = gaussian_toy.build()

    plt.scatter(toy_dataset[:, 0], toy_dataset[:, 1])
    plt.show()

    end=True