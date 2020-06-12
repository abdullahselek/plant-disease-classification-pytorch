#!/usr/bin/env python

import os
import unittest
import pytest

from plant_disease_classification_pytorch import data_generator


class DataGeneratorTest(unittest.TestCase):

    def test_read_train_sets(self):
        parent_directory_path = os.path.dirname('.')
        training_set_path = os.path.join(parent_directory_path, 'datasets/train')
        classes = os.listdir(training_set_path)
        image_size = 128
        train_dataset, validation_dataset = data_generator.read_datasets(training_set_path,
            image_size, classes, 0.2)
        self.assertEqual(train_dataset.images.shape[0], 17534)
        self.assertEqual(validation_dataset.images.shape[0], 4383)


    def main(self):
        self.test_read_train_sets()


if __name__ == "__main__":
    data_generator_tests = DataGeneratorTest()
    data_generator_tests.main()
