#!/usr/bin/env python

import os
import torch
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
