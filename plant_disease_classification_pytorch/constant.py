#!/usr/bin/env python

import os

"""Constant values used in the project."""

PARENT_DIRECTORY_PATH = os.path.dirname(".")
TRAINING_SET_PATH = os.path.join(PARENT_DIRECTORY_PATH, "datasets/train")
TEST_SET_PATH = os.path.join(PARENT_DIRECTORY_PATH, "datasets/test")


def classes():
    """Returns the list of classes available in training set.
    Returns:
      Lis of strings."""

    folder_list = os.listdir(TRAINING_SET_PATH)
    return [x for x in folder_list if not (x.startswith("."))]


NUMBER_OF_CLASSES = len(classes())
