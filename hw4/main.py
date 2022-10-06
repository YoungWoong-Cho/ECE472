#!/bin/env python3.8

"""
Homework 4
Author: Youngwoong Cho
The Cooper Union Class of 2023
"""
import os
import torch

from dataset import CIFARDataLoader
from model.seresnet import seresnet34
from trainer import Trainer

from absl import app

"""
This python project implements a multi-class classification task
on CIFAR-10 and CIFAR-100 dataset. It uses a SEResnet(Squeeze-and-Excitation Networks).

I conducted several experiments on the optimizer, learning rate, batch size, and
weight decay. SGD turns out to be the best optimizer together with the learning
rate of 0.1, warm-up learning rate for a single epoch, and a scheduler that
decreases the lr by the factor of 5 on 60th, 120th, and 160th epoch. The training
loss and accuracy increases when the learning rate is deacreased on each scheduled
milestone.
Large batch size (i.e. greater than 128) tends to decrease the training loss more
rapidly, but suffers from generalization error. Small batch size (i.e. batch size
of 64) required much time for the training loss to converge. The batch size of
128 was the best choice that both converges the training loss in reasonable time
and doesn't show the generalization error on the test set.
Large L2 normalization coefficient (higher than 1e-3) negatively affected both
training loss and test loss. Setting L2 coefficient to be zero also resulted in
poor generalization error. L2 coefficient of 5e-4 was the best choice for the
CIFAR dataset.
Lastly, data augmentation is implemented in order to increase the generalization
capability of the model. Normalization, random crop, random horizontal flip, and
random rotation transformations were imposed on the images.
"""

CONFIG = {
    "data_root": "./hw4/dataset",
    "dataset_name": "CIFAR-100",  # CIFAR-10 or CIFAR-100
    "train_val_split": 0.8,
    "cuda": torch.cuda.is_available(),
    "train": {
        "batch_size": 128,
        "epoch": 200,
        "shuffle": True,
        "criterion": "CrossEntropyLoss",
        "optimizer": "SGD",
        "learning_rate": float(1e-1),
        "momentum": 0.9,
        "weight_decay": float(5e-4),
        "scheduler": "MultiStepLR",
        "milestones": [60, 120, 160],
        "gamma": 0.2,
        "warm": 1,
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
    if not os.path.exists(CONFIG["log_dir"]):
        os.mkdir(CONFIG["log_dir"])
    if not os.path.exists(CONFIG["save_dir"]):
        os.mkdir(CONFIG["save_dir"])

    # Generate GT data
    dataloader = CIFARDataLoader(CONFIG)

    # Create model
    model = seresnet34()

    # Prepare trainer
    trainer = Trainer(CONFIG, dataloader, model)
    trainer.train()


if __name__ == "__main__":
    app.run(main)
