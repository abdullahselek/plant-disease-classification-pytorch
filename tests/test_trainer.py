#!/usr/bin/env python

import os
import unittest

from plant_disease_classification_pytorch import trainer


class TrainerTest(unittest.TestCase):

    image_size = 128

    def test_create_dataloaders(self):
        trainloader, testloader = trainer.create_dataloaders()
        self.assertEqual(trainloader.dataset.images.shape[0], 17534)
        self.assertEqual(testloader.dataset.images.shape[0], 32388)
