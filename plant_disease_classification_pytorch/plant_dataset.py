#!/usr/bin/env python

import torch

from torch.utils.data import Dataset


class PlantDataset(Dataset):
    def __init__(self, images, labels, img_names, classes, transform=None):
        """Returns a PlantDataset instance.
        Args:
          images (np.array):
            Array of data images.
          labels (np.array):
            Array of data labels.
          img_names (np.array):
            Array of names of images.
          classes (np.array):
            Array of image classes.
          transform (transforms):
            Transform used in normalization and creating tensor.
        Returns:
          PlantDataset"""

        self.images = images
        self.labels = labels
        self.img_names = img_names
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_data = self.images[idx]
        label = None
        if self.labels is not None:
            label = self.labels[idx]

        if self.transform:
            image_data = self.transform(image_data)
        return image_data, label
