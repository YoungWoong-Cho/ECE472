#!/bin/env python3.8

"""
ECE472 midterm
Author: Youngwoong Cho, Rosemary Cho
The Cooper Union Class of 2023
"""
import os
import torch

from dataset import CIFARDataLoader
from trainer import Trainer
from model.PiT import PiT
from model.ViT import ViT

from absl import app

CONFIG = {
    "data_root": "./midterm/dataset",
    "dataset_name": "CIFAR-100",  # CIFAR-10 or CIFAR-100
    "train_val_split": 0.8,
    "cuda": torch.cuda.is_available(),
    "model": {
        "img_size": 112,
        "pretrained_weight": "/content/ECE472/midterm/pretrained_weights/pit_b_820.pth"
    },
    "train": {
        "batch_size": 512,
        "epoch": 20,
        "shuffle": True,
        "criterion": "CrossEntropyLoss",
        "optimizer": "SGD",
        "learning_rate": float(3e-2),
        "momentum": 0.9,
        # "weight_decay": float(5e-4),
        "weight_decay": 0.0,
        "scheduler": "CosineAnnealingLR",
        # "milestones": [60, 120, 160],
        # "gamma": 0.2,
        "grad_clip": 1.0,
        "warm": 5,
        "log_iter": 100,
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
    # model = PiT(img_size=CONFIG['model']['img_size'],
    #             num_classes=100)
    # model = PiT(img_size=CONFIG['model']['img_size'],
    #             num_classes=100,
    #             patch_size=16,
    #             stride=8,
    #             base_dims=[48, 48, 48],
    #             depth=[2, 6, 4],
    #             heads=[3, 6, 12],
    #             mlp_ratio=4,)
    # model = ViT('B_16', pretrained=False, image_size=CONFIG['model']['img_size'],
    #             num_classes=100)
    model = ViT('B_16', image_size=CONFIG['model']['img_size'], num_classes=100,
                dim=384, num_layers=12, num_heads=6)

    # Prepare trainer
    trainer = Trainer(CONFIG, dataloader, model)
    trainer.train()


if __name__ == "__main__":
    app.run(main)
