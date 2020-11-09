#!/usr/bin/env python

import os

PARENT_DIRECTORY_PATH = os.path.dirname(".")
TRAINING_SET_PATH = os.path.join(PARENT_DIRECTORY_PATH, "datasets/train")
TEST_SET_PATH = os.path.join(PARENT_DIRECTORY_PATH, "datasets/test")
CLASSES = os.listdir(TRAINING_SET_PATH)
NUMBER_OF_CLASSES = 38 # len(CLASSES) .DS_Store
