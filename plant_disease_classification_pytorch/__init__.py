#!/usr/bin/env python

"""Disease classification on different plants with using Machine Learning and Convolutional Neural Networks."""

from __future__ import absolute_import

__author__ = "Abdullah Selek"
__email__ = "abdullahselek@gmail.com"
__copyright__ = "Copyright (c) 2020 Abdullah Selek"
__license__ = "MIT License"
__version__ = "0.1"
__url__ = "https://github.com/abdullahselek/plant-disease-classification-pytorch"
__download_url__ = (
    "https://github.com/abdullahselek/plant-disease-classification-pytorch"
)
__description__ = "Disease classification on different plants with using Machine Learning and Convolutional Neural Networks."


from plant_disease_classification_pytorch import data_generator
from plant_disease_classification_pytorch.plant_dataset import PlantDataset
from plant_disease_classification_pytorch import trainer
from plant_disease_classification_pytorch.network import CNN
from plant_disease_classification_pytorch import constant
