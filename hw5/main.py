#!/bin/env python3.8

"""
Homework 5
Author: Youngwoong Cho
The Cooper Union Class of 2023
"""
from dataset import get_dataset
from model import BidirectionalLSTM
from trainer import train_with_cross_validate

from absl import app


CONFIG = {
    "train": {
        "batch_size": 32,
        "epoch": 10,
        "shuffle": True,
        "criterion": "sparse_categorical_crossentropy",
        "optimizer": "adam",
    },
    "validation": {
        "batch_size": 128,
        "shuffle": True,
    },
    "test": {
        "shuffle": True,
    },
}

def main(a):
    # Generate GT data
    dataset, tokenizer = get_dataset()

    # Create model
    model = BidirectionalLSTM

    # Prepare trainer
    trainer = train_with_cross_validate(dataset, tokenizer, model, 5, CONFIG)
    trainer()

if __name__ == "__main__":
    app.run(main)
