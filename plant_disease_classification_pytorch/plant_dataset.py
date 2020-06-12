#!/usr/bin/env python

from torch.utils.data import Dataset


class PlantDataset(Dataset):

    def __init__(self, images, labels, img_names, classes):
        self.images = images
        self.labels = labels
        self.img_names = img_names
        self.classes = classes


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        return self.images[idx]
