#!/bin/env python3.8

"""
ECE472 MIDTERM
Author: Youngwoong Cho, Rosemary Cho
The Cooper Union Class of 2023
"""
import os

from config import CONFIG
from dataset import CIFARDataLoader
from model.PiT import get_PiT
from model.ViT import get_ViT
from trainer import Trainer
from utils.helpers import compute_flops

if __name__ == "__main__":
    if not os.path.exists(CONFIG["log_dir"]):
        os.mkdir(CONFIG["log_dir"])
    if not os.path.exists(CONFIG["save_dir"]):
        os.mkdir(CONFIG["save_dir"])

    # Prepare dataloader and model
    dataloader = CIFARDataLoader()
    model = get_PiT("Ti16")

    # Run GFLOPs analysis
    model.eval()
    flops = compute_flops(model)
    print(f"Model: {model.__class__.__name__}-{model.name}", end=" ")
    print(f"{flops / 1e9} GFLOPS")

    # Train
    trainer = Trainer(dataloader, model)
    trainer.train()
