#!/bin/env python3.8

"""
Homework 3
Author: Youngwoong Cho
The Cooper Union Class of 2023
"""
import os
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from absl import app
from tqdm import trange


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


class MNISTDataset(Dataset):
    """
    MNIST dataset for train dataset
    https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_test.csv
    """

    def __init__(self, config, csv_fname, dataset_type="train"):
        self.data_root = config["data_root"]
        self.dataset_type = dataset_type
        self.image, self.label = self.parse_csv(csv_fname)

    def parse_csv(self, csv_fname):
        data = pd.read_csv(f"{self.data_root}/{csv_fname}").to_numpy()
        image = data[:, 1:].reshape(-1, 28, 28)
        label = data[:, 0]
        assert self.dataset_type in [
            "train",
            "test",
        ], "dataset_type must be either train or test"
        if self.dataset_type == "train":
            image = image[:50000]
            label = label[:50000]
        else:
            image = image[50000:]
            label = label[50000:]

        return image, label

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.image[idx]).float()
        image = (image - image.mean()) / image.std()
        image = image[None, :]

        label = torch.from_numpy(self.label)[idx].type(torch.LongTensor)

        sample = {"image": image, "label": label}
        return sample


class MNISTDataLoader:
    """
    MNIST dataset for train dataset
    """

    def __init__(self, config):
        self.data_root = config["data_root"]

        # Split mnist_train.csv into train and test
        # because it has 60,000 samples
        self.train_dataset = MNISTDataset(config, "mnist_train.csv", "train")
        self.test_dataset = MNISTDataset(config, "mnist_train.csv", "test")

        train_size = int(config["train_val_split"] * len(self.train_dataset))
        test_size = len(self.train_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset, [train_size, test_size]
        )

        self._train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=config["train"]["shuffle"],
        )
        self._val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=config["validation"]["batch_size"],
            shuffle=config["validation"]["shuffle"],
        )
        self._test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=len(self.test_dataset),
            shuffle=config["test"]["shuffle"],
        )

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def val_dataloader(self):
        return self._val_dataloader

    @property
    def test_dataloader(self):
        return self._test_dataloader


class MNISTModel(nn.Module):
    """
    MNIST CNN model for image classification
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        self.conv_drop = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.maxpool(self.conv1(x)))
        x = self.relu(self.maxpool(self.conv2(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.log_softmax(self.fc2(x))
        return x


class Trainer(object):
    """
    Trainer class for MNIST dataset classification
    """

    def __init__(self, config, dataloader: MNISTDataLoader, model: MNISTModel):
        self.config = config
        self.dataloader = dataloader
        self.model = model
        self.criterion = getattr(nn, config["train"]["criterion"])()
        self.optimizer = getattr(optim, config["train"]["optimizer"])(
            model.parameters(),
            lr=config["train"]["learning_rate"],
            weight_decay=config["train"]["l2_coeff"],
        )
        # Weight decay is not the same as L2 regularization for Adam.
        # However pytorch implementation of L2 regularization of Adam
        # is done via weight_decay parameter.
        # See https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        # https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch

        self.writer = SummaryWriter(config["log_dir"])

    def train(self):
        self.model.train()
        bar = trange(self.config["train"]["epoch"])
        val_accuracy = 0.0
        for i in bar:
            train_data = next(iter(self.dataloader.train_dataloader))
            image = train_data["image"]
            label = train_data["label"]

            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()

            bar.set_description(
                f"Loss @ {i} => {loss.detach().numpy():0.6f} Val accuracy @ {i} => {val_accuracy:0.6f}"
            )
            bar.refresh()

            # Run accuracy on validation set
            if i % 50 == 0:
                val_accuracy = self.validate()

                self.writer.add_scalar(
                    "NLL Loss",
                    loss.detach().numpy(),
                    i,
                )
                self.writer.add_scalar(
                    "Validation accuracy",
                    val_accuracy,
                    i,
                )

        test_accuracy = self.test()
        print(test_accuracy)
        self.save_model()

    def validate(self):
        self.model.eval()
        validate_data = next(iter(self.dataloader.val_dataloader))
        image = validate_data["image"]
        label = validate_data["label"]

        output = self.model(image)
        metric = accuracy(output, label)
        return metric

    def test(self):
        self.model.eval()
        test_data = next(iter(self.dataloader.test_dataloader))
        image = test_data["image"]
        label = test_data["label"]

        output = self.model(image)
        metric = accuracy(output, label)
        return metric

    def save_model(self):
        torch.save(self.model.state_dict(), self.config["save_dir"])


CONFIG = {
    "data_root": "./hw3/dataset",
    "train_val_split": 0.8,
    "train": {
        "batch_size": 32,
        "epoch": 1000,
        "shuffle": True,
        "criterion": "NLLLoss",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "l2_coeff": 0.001,
    },
    "validation": {
        "batch_size": 128,
        "shuffle": True,
    },
    "test": {
        "shuffle": True,
    },
    "log_dir": os.path.join(os.path.dirname(os.path.realpath(__file__)), "log"),
    "save_dir": os.path.join(os.path.dirname(os.path.realpath(__file__)), "save"),
}


def main(a):
    # Generate GT data
    dataloader = MNISTDataLoader(CONFIG)

    # Create a MLP model
    model = MNISTModel()

    # Prepare trainer
    trainer = Trainer(CONFIG, dataloader, model)
    trainer.train()


if __name__ == "__main__":
    app.run(main)
