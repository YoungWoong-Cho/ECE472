#!/bin/env python3.8

"""
Homework 4
Author: Youngwoong Cho
The Cooper Union Class of 2023
"""
import os
import torch

from dataset import CIFARDataLoader
from model.ViT import ViT
from model.resnet import ResNet50
from trainer import Trainer

from absl import app
from pdb import set_trace as bp


CONFIG = {
    "data_root": "./hw4/dataset",
    "dataset_name": "CIFAR-10",  # CIFAR-10 or CIFAR-100
    "train_val_split": 0.8,
    "cuda": torch.cuda.is_available(),
    "train": {
        "batch_size": 128,
        "epoch": 200,
        "shuffle": True,
        "criterion": "CrossEntropyLoss",
        "optimizer": "Adam",
        "learning_rate": int(3e-5),
        "l2_coeff": 0.0,
        'gamma': 0.7    
    },
    "validation": {
        "batch_size": 128,
        "shuffle": True,
    },
    "test": {
        "shuffle": True,
    },
    "log_dir": os.path.join(os.path.dirname(os.path.realpath(__file__)), "log/ViT"),
    "save_dir": os.path.join(os.path.dirname(os.path.realpath(__file__)), "save/ViT"),
}

def main(a):
    # Generate GT data
    dataloader = CIFARDataLoader(CONFIG)

    # Create a ViT model
    model = ViT(image_size = 32,
                patch_size = 4,
                num_classes = 10,
                dim = 512,
                depth = 6,
                heads = 16,
                mlp_dim = 512,
                dropout = 0.1,
                emb_dropout = 0.1)
    # model = ResNet50()

    # Prepare trainer
    trainer = Trainer(CONFIG, dataloader, model)
    trainer.train()


if __name__ == "__main__":
    app.run(main)
