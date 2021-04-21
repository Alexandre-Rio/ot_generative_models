"""
Created on Mon Apr 19 14:41:00 2021

@author: alexandre-rio
"""

import torchvision.transforms as transforms

mnist_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32))
])