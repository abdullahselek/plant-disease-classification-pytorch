#!/usr/bin/env python

import sys
import argparse

from plant_disease_classification_pytorch import trainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Disease classification on different plants with using Machine Learning and Convolutional Neural Networks."
    )
    parser.add_argument("-t", "--train", type=int, help="Number of epochs for training")
    args = parser.parse_args()

    if len(sys.argv) < 2:
        print("Specify a key to use")
        sys.exit(1)

    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()
    if args.train != None:
        trainer.EPOCHS = args.train
        trainer.train()
