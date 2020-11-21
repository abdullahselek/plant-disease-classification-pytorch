#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from plant_disease_classification_pytorch import trainer, constant


class CNN(nn.Module):
    """Convolutional Neural Network which does the raining."""

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20 * 30 * 30, 1024)
        self.fc2 = nn.Linear(1024, constant.NUMBER_OF_CLASSES)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 20 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
