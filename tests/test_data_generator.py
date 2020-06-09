#!/usr/bin/env python

import os
import unittest

from plant_disease_classification_pytorch import data_generator


class DataGeneratorTest(unittest.TestCase):

    def test_load_train_data(self):
        parent_directory_path = os.path.dirname('.')
        training_set_path = os.path.join(parent_directory_path, 'datasets/train')
        classes = os.listdir(training_set_path)
        images, labels, img_names, class_array = data_generator.load_train_data(training_set_path, 128, classes)
        self.assertEqual(len(images), 21917)
        self.assertEqual(len(labels), 21917)
        self.assertEqual(len(img_names), 21917)
        self.assertEqual(len(class_array), 21917)
