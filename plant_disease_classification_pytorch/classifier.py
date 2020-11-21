#!/usr/bin/env python

import torch
import torchvision.transforms as transforms

from torch.autograd import Variable
from PIL import Image
from plant_disease_classification_pytorch.network import CNN
from plant_disease_classification_pytorch import constant


# CPU or GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Classifier(object):
    """An interface which classifies the given image."""

    def __init__(self, model_path: str = None):
        """Returns a Classifier instance.
        Args:
          model_path (str):
            Model path that is going to be loaded for classification.
        Returns:
          Classifier"""

        self.img_size = 128
        self.tranform = transforms.Compose(
            [transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()]
        )
        if model_path:
            self.__load_model(path=model_path)
        else:
            print("Provide a path to trained model!")

    def __load_model(self, path: str):
        self.model = CNN()
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        self.model.eval()

    def __load_image(self, image_path: str):
        image = Image.open(image_path)
        tensor = self.tranform(image).float()
        tensor = Variable(tensor, requires_grad=True)
        tensor = tensor.to(DEVICE)
        return tensor

    def __batch_data(self, tensor):
        return tensor[None, ...]

    def classify(self, image_path: str):
        """Returns a prediction class.
        Args:
          image_path (str):
            Image path that is going to be used in classification.
        Returns:
          Prediction class (str)"""

        tensor = self.__load_image(image_path)
        output = self.model(self.__batch_data(tensor))
        predicted = torch.argmax(output)
        classes = constant.classes()
        prediction_class = classes[int(predicted.item())]
        return prediction_class
