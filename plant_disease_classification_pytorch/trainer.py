#!/usr/bin/env python

import os
import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from plant_disease_classification_pytorch import data_generator, constant
from plant_disease_classification_pytorch.network import CNN

"""Module that does the model traning."""

IMAGE_SIZE = 128
BATCH_SIZE = 25
LEARNING_RATE = 0.001
EPOCHS = 35

# CPU or GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)


def create_dataloaders():
    train_dataset, validation_dataset = data_generator.read_datasets(
        constant.TRAINING_SET_PATH, IMAGE_SIZE, constant.classes(), 0.2
    )
    # test_dataset = data_generator.read_test_dataset(constant.TEST_SET_PATH, IMAGE_SIZE)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    # test_loader = DataLoader(
    #     dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    # )

    return train_loader, valid_loader


def check_accuracy(valid_loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in valid_loader:
            data = data.to(device=device)
            labels = labels.to(device=device)

            outputs = model(data)
            _, predicted = torch.max(outputs.data, 0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        "Accuracy of the network on the %d test images: %d %%"
        % (len(valid_loader.dataset), 100 * correct / total)
    )


def train():
    train_loader, valid_loader = create_dataloaders()

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
        for data, target in train_loader:
            # move tensors to GPU
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            target = torch.max(target, 1)[1]
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

            target = torch.max(target, 1)[1]
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

    # test the model
    check_accuracy(valid_loader=valid_loader, model=model, device=DEVICE)

    # save
    torch.save(model.state_dict(), "model.pt")
