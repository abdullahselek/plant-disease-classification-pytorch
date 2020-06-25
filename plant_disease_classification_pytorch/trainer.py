#!/usr/bin/env python

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from random import seed
from random import randint
from plant_disease_classification_pytorch import data_generator


parent_directory_path = os.path.dirname('.')
training_set_path = os.path.join(parent_directory_path, 'datasets/train')
test_set_path = os.path.join(parent_directory_path, 'datasets/test')
classes = os.listdir(training_set_path)
image_size = 128


def create_dataloaders():
    train_dataset, validation_dataset = data_generator.read_datasets(training_set_path,
        image_size, classes, 0.2)
    test_dataset = data_generator.read_test_dataset(test_set_path, image_size)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                            shuffle=False, num_workers=2)
    return trainloader, testloader


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
