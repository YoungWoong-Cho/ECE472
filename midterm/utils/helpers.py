import math
import numpy as np
import torch

from config import CONFIG
from fvcore.nn import FlopCountAnalysis
from PIL import Image
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]


def compute_flops(model):
    input = torch.rand([1, 3, CONFIG['model']['image_size'], CONFIG['model']['image_size']])
    flops = FlopCountAnalysis(model, input)
    return flops.total()


def save_img(data, idx, fname='img'):
    images = data[b"data"]
    img = np.transpose(images.reshape(-1, 3, 32, 32), (0, 2, 3, 1))[idx]
    img = Image.fromarray(img)
    img.save(f"{fname}.png")


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def truncate_normal(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor
