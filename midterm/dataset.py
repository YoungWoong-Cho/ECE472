import torch
import torchvision
from torchvision import transforms


class CIFARDataLoader:
    def __init__(self, config):
        if config["dataset_name"] == "CIFAR-10":
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.49139968, 0.48215841, 0.44653091],
                        [0.24703223, 0.24348513, 0.26158784],
                    ),
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomCrop(size=[32, 32], padding=4),
                    transforms.Resize(112),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.49139968, 0.48215841, 0.44653091],
                        [0.24703223, 0.24348513, 0.26158784],
                    ),
                    transforms.Resize(112),
                ]
            )
            trainset = torchvision.datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transform_train
            )
            testset = torchvision.datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform_test
            )
        elif config["dataset_name"] == "CIFAR-100":
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                    ),
                    # transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.Resize(112),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                    ),
                    transforms.Resize(112),
                ]
            )
            trainset = torchvision.datasets.CIFAR100(
                root="./data", train=True, download=True, transform=transform_train
            )
            testset = torchvision.datasets.CIFAR100(
                root="./data", train=False, download=True, transform=transform_test
            )
        else:
            raise Exception("dataset_name not understood")

        self.train_dataloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            num_workers=4,
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            testset,
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            num_workers=4,
        )
