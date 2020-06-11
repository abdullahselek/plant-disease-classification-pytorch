#!/usr/bin/env python

import os
import unittest
import pytest

from plant_disease_classification_pytorch import data_generator


class DataGeneratorTest(unittest.TestCase):

    parent_directory_path = os.path.dirname('.')
    training_set_path = os.path.join(parent_directory_path, 'datasets/train')
    classes = os.listdir(training_set_path)
    image_size = 128

    def test_load_train_data(self):
        images, labels, img_names, class_array = data_generator.load_train_data(self.training_set_path,
            self.image_size, self.classes)
        self.assertEqual(len(images), 21917)
        self.assertEqual(len(labels), 21917)
        self.assertEqual(len(img_names), 21917)
        self.assertEqual(len(class_array), 21917)


    def test_read_train_sets(self):
        dataset = data_generator.read_train_sets(self.training_set_path,
            self.image_size, self.classes, 0.2)
        self.assertEqual(dataset.train.images.shape[0], 17534)
        self.assertEqual(dataset.validation.images.shape[0], 4383)


    def main(self):
        self.test_load_train_data()
        self.test_read_train_sets()


if __name__ == "__main__":
    data_generator_tests = DataGeneratorTest()
    data_generator_tests.main()
