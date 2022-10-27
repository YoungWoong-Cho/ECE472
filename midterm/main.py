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
from model.ViT import get_ViT
from model.PiT import get_PiT
from utils.helpers import compute_flops

if __name__ == "__main__":
    if not os.path.exists(CONFIG["log_dir"]):
        os.mkdir(CONFIG["log_dir"])
    if not os.path.exists(CONFIG["save_dir"]):
        os.mkdir(CONFIG["save_dir"])

    # Prepare dataloader and model
    dataloader = CIFARDataLoader()
    model = get_PiT('Ti16')

    # Run GFLOPs analysis
    model.eval()
    flops = compute_flops(model)
    print(f'{flops.total() / 1e9} GFLOPS')

    trainer = Trainer(dataloader, model)
    trainer.train()
