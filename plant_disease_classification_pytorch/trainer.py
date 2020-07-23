#!/usr/bin/env python

import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from random import seed
from random import randint
from plant_disease_classification_pytorch import data_generator
from plant_disease_classification_pytorch.network import Net


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


def train():
    trainloader, testloader = create_dataloaders()

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(3):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


train()
