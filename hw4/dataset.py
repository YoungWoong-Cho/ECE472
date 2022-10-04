import pickle
import numpy as np
import torch

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


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

        if self.dataset_type == 'train':
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.50707516, 0.48654887, 0.44091784], [0.26733429, 0.25643846, 0.27615047]),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.RandomCrop(size=[32, 32], padding=4),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.50707516, 0.48654887, 0.44091784], [0.26733429, 0.25643846, 0.27615047]),
            ])

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
                image = np.transpose(image, (0, 2, 3, 1))
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
            images = np.transpose(images, (0, 2, 3, 1))
            labels = data[b"fine_labels"]
        else:
            raise Exception("dataset_name not understood.")

        return images, labels

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.transforms(self.image[idx])
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
        val_size = len(self.train_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset, [train_size, val_size]
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