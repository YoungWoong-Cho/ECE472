import torch
import torchvision
from config import CONFIG
from torchvision import transforms

NORMALIZATION_FACTORS = {
    'CIFAR10': [(0.49139968, 0.48215841, 0.44653091),
                 (0.24703223, 0.24348513, 0.26158784)],
    'CIFAR100': [(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                  (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)]
}

class CIFARDataLoader:
    """
    Dataloader class for CIFAR dataset
    """
    def __init__(self):
        assert CONFIG["dataset_name"] in ['CIFAR10', 'CIFAR100'], \
            'dataset_name must be one of the followings: CIFAR10, CIFAR100'
        
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    *NORMALIZATION_FACTORS[CONFIG['dataset_name']]
                ),
                transforms.Resize(CONFIG['model']['image_size'])
            ]
        )

        trainset = getattr(torchvision.datasets, CONFIG['dataset_name'])(
            root=CONFIG['data_root'],
            train=True,
            download=True,
            transform=transform
        )
        testset = getattr(torchvision.datasets, CONFIG['dataset_name'])(
            root=CONFIG['data_root'],
            train=False,
            download=True,
            transform=transform
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=CONFIG["train"]["batch_size"],
            shuffle=CONFIG['train']['shuffle'],
            num_workers=4,
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            testset,
            batch_size=CONFIG["train"]["batch_size"],
            shuffle=CONFIG['test']['shuffle'],
            num_workers=4,
        )
