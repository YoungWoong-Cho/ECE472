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

def top_k_accuracy(output, target):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    top1 = 0.0
    top5 = 0.0
    for idx, label in enumerate(target):
        class_prob = output[idx]
        top_values = (-class_prob).argsort()[:5]
        if top_values[0] == label:
            top1 += 1.0
        if np.isin(np.array([label]), top_values):
            top5 += 1.0
    top1 = top1 / len(target)
    top5 = top5 / len(target)
    return {'top1': top1, 'top5': top5}


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

        if self.config['mps']:
            image = image.to('mps')
            label = label.to('mps')

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


# # 3x3 convolution
# def conv3x3(in_channels, out_channels, stride=1):
#     return nn.Conv2d(
#         in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
#     )


# # Residual block
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = conv3x3(in_channels, out_channels, stride)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(out_channels, out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out


# # ResNet
# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_channels = 16
#         self.conv = conv3x3(3, 16)
#         self.bn = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self.make_layer(block, 16, layers[0])
#         self.layer2 = self.make_layer(block, 32, layers[1], 2)
#         self.layer3 = self.make_layer(block, 64, layers[2], 2)
#         self.avg_pool = nn.AvgPool2d(8)
#         self.fc = nn.Linear(64, num_classes)

#     def make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         if (stride != 1) or (self.in_channels != out_channels):
#             downsample = nn.Sequential(
#                 conv3x3(self.in_channels, out_channels, stride=stride),
#                 nn.BatchNorm2d(out_channels),
#             )
#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride, downsample))
#         self.in_channels = out_channels
#         for i in range(1, blocks):
#             layers.append(block(out_channels, out_channels))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.bn(out)
#         out = self.relu(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.avg_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out


class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_layers, block, image_channels, num_classes):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)

def ResNet18(img_channels=3, num_classes=1000):
    return ResNet(18, Block, img_channels, num_classes)


def ResNet34(img_channels=3, num_classes=1000):
    return ResNet(34, Block, img_channels, num_classes)


def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(50, Block, img_channels, num_classes)


def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(101, Block, img_channels, num_classes)


def ResNet152(img_channels=3, num_classes=1000):
    return ResNet(152, Block, img_channels, num_classes)


class Trainer(object):
    """
    Trainer class for MNIST dataset classification
    """

    def __init__(self, config, dataloader: CIFARDataLoader, model):
        self.config = config
        self.dataloader = dataloader
        self.model = model
        if self.config['mps']:
            self.model.to('mps')
        self.criterion = getattr(nn, config["train"]["criterion"])()
        self.optimizer = getattr(optim, config["train"]["optimizer"])(
            self.model.parameters(),
            lr=config["train"]["learning_rate"],
            weight_decay=config["train"]["l2_coeff"],
        )

        self.writer = SummaryWriter(config["log_dir"])

    def train(self):
        self.model.train()
        val_accuracy = {'top1': 0.0, 'top5': 0.0}
        global_i = 0
        for epoch in range(self.config['train']['epoch']):
            bar = trange(len(self.dataloader.train_dataloader))
            for _ in bar:
                train_data = next(iter(self.dataloader.train_dataloader))
                image = train_data["image"]
                label = train_data["label"]

                self.optimizer.zero_grad()
                output = self.model(image)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                bar.set_description(
                    f"Epoch: {epoch} [Loss: {loss.cpu().detach().numpy():0.6f}] [Top1: {val_accuracy['top1']:0.6f}] [Top5: {val_accuracy['top5']:0.6f}]"
                )
                bar.refresh()

                # Run accuracy on validation set
                if global_i % 50 == 0:
                    val_accuracy = self.validate()

                    self.writer.add_scalar(
                        "Cross Entropy Loss",
                        loss.cpu().detach().numpy(),
                        global_i,
                    )
                    self.writer.add_scalar(
                        "Top 1 accuracy",
                        val_accuracy['top1'],
                        global_i,
                    )
                    self.writer.add_scalar(
                        "Top 5 accuracy",
                        val_accuracy["top5"],
                        global_i,
                    )

                global_i += 1

        test_accuracy = self.test()
        print(test_accuracy)
        self.save_model()

    def validate(self):
        self.model.eval()
        validate_data = next(iter(self.dataloader.val_dataloader))
        image = validate_data["image"]
        label = validate_data["label"]

        output = self.model(image)
        metric = top_k_accuracy(output, label)
        return metric

    def test(self):
        self.model.eval()
        test_data = next(iter(self.dataloader.test_dataloader))
        image = test_data["image"]
        label = test_data["label"]

        output = self.model(image)
        metric = top_k_accuracy(output, label)
        return metric

    def save_model(self):
        torch.save(self.model.state_dict(), self.config["save_dir"])


CONFIG = {
    "data_root": "./hw4/dataset",
    "dataset_name": "CIFAR-10",  # CIFAR-10 or CIFAR-100
    "train_val_split": 0.8,
    "mps": torch.backends.mps.is_available(),
    "train": {
        "batch_size": 128,
        "epoch": 100,
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
    "log_dir": os.path.join(os.path.dirname(os.path.realpath(__file__)), "log/ResNet50"),
    "save_dir": os.path.join(os.path.dirname(os.path.realpath(__file__)), "save/ResNet50"),
}


def main(a):
    # Generate GT data
    dataloader = CIFARDataLoader(CONFIG)

    # Create a MLP model
    # model = ResNet(ResidualBlock, [2, 2, 2])
    model = ResNet50(num_classes=10)

    # Prepare trainer
    trainer = Trainer(CONFIG, dataloader, model)
    trainer.train()


if __name__ == "__main__":
    app.run(main)
