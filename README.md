# plant-disease-classification-pytorch

Disease classification on different plants with using Machine Learning and Convolutional Neural Networks.

## List of contents

  - [Introduction](#introduction)
  - [Datasets](#datasets)
  - [Installation](#installation)
  - [Running](#running)
  - [License](#license)

## Introduction

One of the important field of AI in agriculture is the disease detection in plants. Having a disease in plants are quite usual and if proper care is not taken in this area then it can cause serious effects on plants and due to which respective product quality, quantity or productivity is also affected. Plant diseases cause a periodic outbreak of diseases which leads to large-scale death. These problems need to be solved at the initial stage, to save life and money of people. Automatic detection of plant diseases is an important research topic as it may prove benefits in monitoring large fields of crops, and at a very early stage itself it detects the symptoms of diseases means when they appear on plant leaves. Farm landowners and plant caretakers could be benefited a lot with an early disease detection, in order to prevent the worse to come to their plants and let the human know what has to be done beforehand for the same to work accordingly, in order to prevent the worse.

The project involves the use of self-designed image processing algorithms and techniques designed using Python and PyTorch to segment the disease from the leaf while using the concepts of machine learning to categorise the plant leaves as healthy or infected. This enables to be identify the diseases at the initial stage and the pest and infection control tools can be used to solve pest problems while minimizing risks to people and the environment.

## Datasets

Datasets are available at [here](https://github.com/abdullahselek/plant-disease-classification-pytorch/tree/master/datasets) and in another [repo](https://github.com/abdullahselek/plant-disease-classification-datasets) as compressed file.

## Installation

The code is hosted at https://github.com/abdullahselek/plant-disease-classification-pytorch

Check out the latest development version anonymously with

    git clone git://github.com/abdullahselek/plant-disease-classification-pytorch.git
    cd plant-disease-classification-pytorch

To install dependencies

    pip3 install -r requirements.txt

To install the plant-disease-classification-pytorch module

    pip3 install -e .

## Running

After installing module to your computer you can run commands below.

```
Disease classification on different plants with using Machine Learning and
Convolutional Neural Networks.

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN, --train TRAIN
                        Number of epochs for training
  -m MODEL, --model MODEL
                        Path of the model will be used in classification
  -i IMAGE, --image IMAGE
                        Path of the image file you want to classify
```

Traing your model with 35 epochs

    python -m plant_disease_classification_pytorch -t 35

Classify your image with trained model

    python -m plant_disease_classification_pytorch -m "model.pt" -i datasets/test/0a1e21dd9c2ddaf1ce1db1706d411649.jpg


## License

[MIT License](https://github.com/abdullahselek/plant-disease-classification-pytorch/blob/master/LICENSE)
