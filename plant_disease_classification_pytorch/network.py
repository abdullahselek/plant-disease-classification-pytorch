#!/usr/bin/env python

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # self.conv1 = nn.Conv1d(128, 3, 128)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(3, 128, 128)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=3, kernel_size=128)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=5)

        self.pool = nn.MaxPool2d(kernel_size=5, stride=2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        print("x:")
        print(x)
        print(len(x))
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
