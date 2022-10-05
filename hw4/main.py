#!/bin/env python3.8

"""
Homework 4
Author: Youngwoong Cho
The Cooper Union Class of 2023
"""
import os
import torch

from dataset import CIFARDataLoader
# from model.ViT import ViT
# from model.resnet import ResNet50, ResNet101
# from model.dla import DLA
# from model.dpn import DPN26
# from model.googlenet import GoogleNet
from model.seresnet import seresnet34
from trainer import Trainer

from absl import app
from pdb import set_trace as bp


CONFIG = {
    "data_root": "./hw4/dataset",
    "dataset_name": "CIFAR-100",  # CIFAR-10 or CIFAR-100
    "train_val_split": 0.8,
    "cuda": torch.cuda.is_available(),
    "train": {
        "batch_size": 64,
        "epoch": 200,
        "shuffle": True,
        "criterion": "CrossEntropyLoss",
        "optimizer": "SGD",
        "learning_rate": float(1e-1),
        'warm': 1,
        "l2_coeff": float(5e-4),
        'scheduler': "MultiStepLR",
        'milestones': [60, 120, 160],
        'gamma': 0.2,
        'momentum': 0.9
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
    if not os.path.exists(CONFIG['log_dir']):
        os.mkdir(CONFIG['log_dir'])
    if not os.path.exists(CONFIG['save_dir']):
        os.mkdir(CONFIG['save_dir'])
    
    # Generate GT data
    dataloader = CIFARDataLoader(CONFIG)

    # Create model
    model = seresnet34()

    # Prepare trainer
    trainer = Trainer(CONFIG, dataloader, model)
    trainer.train()


if __name__ == "__main__":
    app.run(main)
