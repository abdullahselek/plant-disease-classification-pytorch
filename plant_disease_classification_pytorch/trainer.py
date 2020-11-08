#!/usr/bin/env python

import os
import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from plant_disease_classification_pytorch import data_generator
from plant_disease_classification_pytorch.network import CNN


PARENT_DIRECTORY_PATH = os.path.dirname(".")
TRAINING_SET_PATH = os.path.join(PARENT_DIRECTORY_PATH, "datasets/train")
TEST_SET_PATH = os.path.join(PARENT_DIRECTORY_PATH, "datasets/test")
CLASSES = os.listdir(TRAINING_SET_PATH)
NUMBER_OF_CLASSES = len(CLASSES)
IMAGE_SIZE = 128
BATCH_SIZE = 25
LEARNING_RATE = 0.001
EPOCHS = 35

# CPU or GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_dataloaders():
    train_dataset, validation_dataset = data_generator.read_datasets(
        TRAINING_SET_PATH, IMAGE_SIZE, CLASSES, 0.2
    )
    test_dataset = data_generator.read_test_dataset(TEST_SET_PATH, IMAGE_SIZE)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    return train_loader, valid_loader, test_loader


def train():
    train_loader, valid_loader, testloader = create_dataloaders()

    model = CNN().to(DEVICE)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    valid_losses = []

    for epoch in range(1, EPOCHS + 1):
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # training the model
        model.train()
        # for data, target in train_loader:
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0]
            target = data[1]
            # move tensors to GPU
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            print("output shape:")
            print(output.shape)
            print("target shape:")
            print(target.shape)
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss wrt model parameters
            loss.backward()
            # perform a ingle optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        # validate the model
        model.eval()
        for data, target in valid_loader:

            data = data.to(DEVICE)
            target = target.to(DEVICE)

            output = model(data)

            loss = criterion(output, target)

            # update average validation loss
            valid_loss += loss.item() * data.size(0)

        # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

train()
