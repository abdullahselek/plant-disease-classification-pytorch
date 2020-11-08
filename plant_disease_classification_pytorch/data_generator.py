#!/usr/bin/env python

import os
import glob
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from sklearn.utils import shuffle
from plant_disease_classification_pytorch.plant_dataset import PlantDataset


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
)

test_transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
)

valid_transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
)


def load_train_data(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    class_array = []
    extension_list = ("*.jpg", "*.JPG")

    print("Going to read training images")
    for fields in classes:
        index = classes.index(fields)
        print("Now going to read {} files (Index: {})".format(fields, index))
        for extension in extension_list:
            path = os.path.join(train_path, fields, extension)
            files = glob.glob(path)
            for file_path in files:
                image = Image.open(file_path)
                image = image.resize((image_size, image_size))
                pixels = np.array(image)
                pixels = pixels.astype(np.float32)
                pixels = np.multiply(pixels, 1.0 / 255.0)
                images.append(pixels)
                label = np.zeros(len(classes))
                label[index] = 1.0
                labels.append(label)
                file_base = os.path.basename(file_path)
                img_names.append(file_base)
                class_array.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    class_array = np.array(class_array)
    return images, labels, img_names, class_array


def read_datasets(train_path, image_size, classes, validation_size):
    images, labels, img_names, class_array = load_train_data(
        train_path, image_size, classes
    )
    # images, labels, img_names, class_array = shuffle(images, labels, img_names, class_array)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = class_array[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = class_array[validation_size:]

    train_dataset = PlantDataset(
        images=train_images,
        labels=train_labels,
        img_names=train_img_names,
        classes=train_cls,
        transform=train_transform,
    )
    validation_dataset = PlantDataset(
        images=validation_images,
        labels=validation_labels,
        img_names=validation_img_names,
        classes=validation_cls,
        transform=valid_transform,
    )

    return train_dataset, validation_dataset


def read_test_dataset(test_path, image_size):
    images = np.empty((32388))
    img_names = []

    print("Going to read test images")
    files = os.listdir(test_path)
    count = 0
    for f in files:
        file_path = os.path.join(test_path, f)
        image = Image.open(file_path)
        image = image.resize((image_size, image_size))
        pixels = np.array(image)
        pixels = pixels.astype(np.float32)
        pixels = np.multiply(pixels, 1.0 / 255.0)
        # images.append(pixels)
        np.append(images, pixels)
        file_base = os.path.basename(file_path)
        img_names.append(file_base)
        count += 1
        if count % 5000 == 0:
            print(str.format("Read {0} test images", count))
    print(str.format("Completed reading {0} images of test dataset", count))
    img_names = np.array(img_names)

    test_dataset = PlantDataset(
        images=images,
        labels=None,
        img_names=img_names,
        classes=None,
        transform=test_transform,
    )
    return test_dataset
