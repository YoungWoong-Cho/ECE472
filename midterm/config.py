import os
import torch

CONFIG = {
    "data_root": "./dataset",       # directory to download CIFAR dataset
    "dataset_name": "CIFAR100",     # CIFAR10 or CIFAR100
    "model": {
        "image_size": 112,
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
    "test": {
        "batch_size": 512,
        "shuffle": True,
    },
    "log_dir": os.path.join(os.path.dirname(os.path.realpath(__file__)), "log"),
    "save_dir": os.path.join(os.path.dirname(os.path.realpath(__file__)), "save"),
}


if torch.backends.mps.is_available():
    CONFIG['device'] = 'mps'
elif torch.cuda.is_available():
    CONFIG['device'] = 'cuda'
else:
    CONFIG['device'] = 'cpu'


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
        'stride': 7,
        'base_dims': [64, 64, 64],
        'depth': [3, 6, 4],
        'heads': [4, 8, 16],
        'mlp_ratio': 4,
        'num_classes': int(CONFIG['dataset_name'][5:]),
    },
    'S16': {
        'patch_size': 16,
        'stride': 8,
        'base_dims': [48, 48, 48],
        'depth': [2, 6, 4],
        'heads': [3, 6, 12],
        'mlp_ratio': 4,
        'num_classes': int(CONFIG['dataset_name'][5:]),
    },
    'Ti16': {
        'patch_size': 16,
        'stride': 8,
        'base_dims': [32, 32, 32],
        'depth': [2, 6, 4],
        'heads': [2, 4, 8],
        'mlp_ratio': 4,
        'num_classes': int(CONFIG['dataset_name'][5:]),
    }
}
