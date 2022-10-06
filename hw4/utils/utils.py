import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import _LRScheduler

def save_img(data, idx):
    images = data[b"data"]
    img = np.transpose(images.reshape(-1, 3, 32, 32), (0, 2, 3, 1))[idx]
    img = Image.fromarray(img)
    img.save("img.png")

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
