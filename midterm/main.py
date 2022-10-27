#!/bin/env python3.8

"""
ECE472 midterm
Author: Youngwoong Cho, Rosemary Cho
The Cooper Union Class of 2023
"""
import os

from config import CONFIG
from dataset import CIFARDataLoader
from trainer import Trainer
from model.PiT import PiT
from model.ViT import ViT

from absl import app


def main(a):
    if not os.path.exists(CONFIG["log_dir"]):
        os.mkdir(CONFIG["log_dir"])
    if not os.path.exists(CONFIG["save_dir"]):
        os.mkdir(CONFIG["save_dir"])

    # Generate GT data
    dataloader = CIFARDataLoader(CONFIG)

    # Create model    
    model = PiT(img_size=CONFIG['model']['img_size'], num_classes=100)
    # PiT_S16 = PiT(iimg_size=CONFIG['model']['img_size'],
    #               num_classes=100,
    #               patch_size=16,
    #               stride=8,
    #               base_dims=[48, 48, 48],
    #               depth=[2, 6, 4],
    #               heads=[3, 6, 12],
    #               mlp_ratio=4,)

    # ViT_B16 = ViT('B_16', image_size=CONFIG['model']['img_size'],
    #               num_classes=100)
    # ViT_S16 = ViT('B_16', image_size=CONFIG['model']['img_size'],
    #               num_classes=100,
    #               dim=384,
    #               num_layers=12, num_heads=6)



    # Prepare trainer
    trainer = Trainer(CONFIG, dataloader, model)
    trainer.train()


if __name__ == "__main__":
    app.run(main)
