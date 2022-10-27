"""configs.py - ViT model configurations, based on:
https://github.com/google-research/vision_transformer/blob/master/vit_jax/configs.py
"""
import os
import torch

def get_base_config():
    """Base ViT config ViT"""
    return dict(
      dim=768,
      ff_dim=3072,
      num_heads=12,
      num_layers=12,
      attention_dropout_rate=0.0,
      dropout_rate=0.1,
      representation_size=768,
      classifier='token'
    )

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = get_base_config()
    config.update(dict(patches=(16, 16)))
    return config

def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.update(dict(patches=(32, 32)))
    return config

def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = get_base_config()
    config.update(dict(
        patches=(16, 16),
        dim=1024,
        ff_dim=4096,
        num_heads=16,
        num_layers=24,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
        representation_size=1024
    ))
    return config

def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.update(dict(patches=(32, 32)))
    return config

def drop_head_variant(config):
    config.update(dict(representation_size=None))
    return config

CONFIG = {
    "data_root": "./dataset",
    "dataset_name": "CIFAR100",  # CIFAR10 or CIFAR100
    "train_val_split": 0.8,
    "cuda": torch.cuda.is_available(),
    "model": {
        "img_size": 112,
    },
    "train": {
        "batch_size": 512,
        "epoch": 20,
        "shuffle": True,
        "criterion": "CrossEntropyLoss",
        "optimizer": "SGD",
        "learning_rate": float(3e-2),
        "momentum": 0.9,
        "weight_decay": 0.0,
        "scheduler": "CosineAnnealingLR",
        "grad_clip": 1.0,
        "warm": 5,
        "log_iter": 100,
    },
    "validation": {
        "batch_size": 128,
        "shuffle": True,
    },
    "test": {
        "batch_size": 512,
        "shuffle": True,
    },
    "log_dir": os.path.join(os.path.dirname(os.path.realpath(__file__)), "log"),
    "save_dir": os.path.join(os.path.dirname(os.path.realpath(__file__)), "save"),
}

VIT_CONFIG = {
    'B16': {
        'patch_size': 16,
        'num_classes': int(CONFIG['dataset_name'][5:]),
        'dim': 768,
        'depth': 12,
        'heads': 12,
        'mlp_dim': 3072
    },
    'S16': {
        'patch_size': 16,
        'num_classes': int(CONFIG['dataset_name'][5:]),
        'dim': 384,
        'depth': 12,
        'heads': 6,
        'mlp_dim': 3072
    },
    'Ti16': {
        'patch_size': 16,
        'num_classes': int(CONFIG['dataset_name'][5:]),
        'dim': 192,
        'depth': 12,
        'heads': 3,
        'mlp_dim': 3072
    }
}

PIT_CONFIG = {
    'B16': {
        'patch_size': 14,
        'num_classes': int(CONFIG['dataset_name'][5:]),
        'stride': 7,
        'base_dims': [64, 64, 64],
        'depth': [3, 6, 4],
        'heads': [4, 8, 16],
        'mlp_ratio': 4
    },
    'S16': {
        'patch_size': 16,
        'num_classes': int(CONFIG['dataset_name'][5:]),
        'stride': 8,
        'base_dims': [48, 48, 48],
        'depth': [2, 6, 4],
        'heads': [3, 6, 12],
        'mlp_ratio': 4
    },
    'Ti16': {
        'patch_size': 16,
        'num_classes': int(CONFIG['dataset_name'][5:]),
        'stride': 8,
        'base_dims': [32, 32, 32],
        'depth': [2, 6, 4],
        'heads': [2, 4, 8],
        'mlp_ratio': 4
    }
}
