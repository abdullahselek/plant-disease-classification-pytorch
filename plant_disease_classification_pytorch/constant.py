#!/usr/bin/env python

import os

PARENT_DIRECTORY_PATH = os.path.dirname(".")
TRAINING_SET_PATH = os.path.join(PARENT_DIRECTORY_PATH, "datasets/train")
TEST_SET_PATH = os.path.join(PARENT_DIRECTORY_PATH, "datasets/test")


def classes():
    folder_list = os.listdir(TRAINING_SET_PATH)
    return [x for x in folder_list if not (x.startswith("."))]


NUMBER_OF_CLASSES = len(classes())
