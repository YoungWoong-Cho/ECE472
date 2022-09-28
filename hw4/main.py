#!/bin/env python3.8

"""
Homework 3
Author: Youngwoong Cho
The Cooper Union Class of 2023
"""
import os
import pandas as pd
import numpy as np
import pickle

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from absl import app
from tqdm import trange
from pdb import set_trace as bp


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


class CIFARDataset(Dataset):
    """
    MNIST dataset for train dataset
    https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_test.csv
    """

    def __init__(self, config, dataset_type="train"):
        self.config = config
        self.data_root = config["data_root"]
        self.dataset_type = dataset_type
        self.image, self.label = self._parse_data()

    def _parse_data(self):
        """
        fpath (list) : all data file paths
        """
        if self.config["dataset_name"] == "CIFAR-10":
            if self.dataset_type == "train":
                fpath = [
                    f"{self.data_root}/cifar-10-batches-py/data_batch_{i+1}"
                    for i in range(5)
                ]
            else:
                fpath = [f"{self.data_root}/cifar-10-batches-py/test_batch"]

            images = []
            labels = []
            for path in fpath:
                with open(path, "rb") as f:
                    data = pickle.load(f, encoding="bytes")
                image = data[b"data"].reshape(-1, 3, 32, 32)
                label = data[b"labels"]
                images.append(image)
                labels += label
            images = np.vstack(images)

        elif self.config["dataset_name"] == "CIFAR-100":
            if self.dataset_type == "train":
                fpath = f"{self.data_root}/cifar-100-python/train"
            else:
                fpath = f"{self.data_root}/cifar-100-python/test"
            with open(fpath, "rb") as f:
                data = pickle.load(f, encoding="bytes")
            images = data[b"data"].reshape(-1, 3, 32, 32)
            labels = data[b"fine_labels"]
        else:
            raise Exception("dataset_name not understood.")

        return images, labels

    def _save_img(self, data, idx):
        from PIL import Image

        images = data[b"data"]
        img = np.transpose(images.reshape(-1, 3, 32, 32), (0, 2, 3, 1))[idx]
        img = Image.fromarray(img)
        img.save("img.png")

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.image[idx]).float()
        image = (image - image.mean()) / image.std()

        label = torch.Tensor(self.label)[idx].type(torch.LongTensor)

        sample = {"image": image, "label": label}
        return sample


class CIFARDataLoader:
    """
    MNIST dataset for train dataset
    """

    def __init__(self, config):
        self.data_root = config["data_root"]

        self.train_dataset = CIFARDataset(config, "train")
        self.test_dataset = CIFARDataset(config, "test")

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


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(
#             planes, planes, kernel_size=3, stride=1, padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_planes,
#                     self.expansion * planes,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(self.expansion * planes),
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(
#             planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(
#             planes, self.expansion * planes, kernel_size=1, bias=False
#         )
#         self.bn3 = nn.BatchNorm2d(self.expansion * planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_planes,
#                     self.expansion * planes,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(self.expansion * planes),
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


# def ResNet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2])


# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])


# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])


# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])


# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CIFARModel(nn.Module):
    """
    CIFAR CNN model for image classification
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        self.conv_drop = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(32 * 32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.maxpool(self.conv1(x)))
        x = self.relu(self.maxpool(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.log_softmax(self.fc2(x))
        return x


class Trainer(object):
    """
    Trainer class for MNIST dataset classification
    """

    def __init__(self, config, dataloader: CIFARDataLoader, model):
        self.config = config
        self.dataloader = dataloader
        self.model = model
        self.criterion = getattr(nn, config["train"]["criterion"])()
        self.optimizer = getattr(optim, config["train"]["optimizer"])(
            self.model.parameters(),
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
                    "Cross Entropy Loss",
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
    "data_root": "./hw4/dataset",
    "dataset_name": "CIFAR-10",  # CIFAR-10 or CIFAR-100
    "train_val_split": 0.8,
    "train": {
        "batch_size": 128,
        "epoch": 1000,
        "shuffle": True,
        "criterion": "CrossEntropyLoss",
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
    dataloader = CIFARDataLoader(CONFIG)

    # Create a MLP model
    model = ResNet(ResidualBlock, [2, 2, 2])

    # Prepare trainer
    trainer = Trainer(CONFIG, dataloader, model)
    trainer.train()


if __name__ == "__main__":
    app.run(main)
