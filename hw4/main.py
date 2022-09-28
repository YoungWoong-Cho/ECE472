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

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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
        
        if self.config['cuda']:
            image = image.to('cuda')
            label = label.to('cuda')

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

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class Trainer(object):
    """
    Trainer class for MNIST dataset classification
    """

    def __init__(self, config, dataloader: CIFARDataLoader, model):
        self.config = config
        self.dataloader = dataloader
        self.model = model
        if self.config['cuda']:
            self.model.to('cuda')
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
    "dataset_name": "CIFAR-100",  # CIFAR-10 or CIFAR-100
    "train_val_split": 0.8,
    "cuda": torch.cuda.is_available(),
    "train": {
        "batch_size": 128,
        "epoch": 400,
        "shuffle": True,
        "criterion": "CrossEntropyLoss",
        "optimizer": "Adam",
        "learning_rate": 0.0001,
        "l2_coeff": 0.001,
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
                num_classes = 10 if CONFIG['dataset_name'] == 'CIFAR-10' else 100,
                dim = 512,
                depth = 6,
                heads = 8,
                mlp_dim = 512,
                dropout = 0.1,
                emb_dropout = 0.1)

    # Prepare trainer
    trainer = Trainer(CONFIG, dataloader, model)
    trainer.train()


if __name__ == "__main__":
    app.run(main)
