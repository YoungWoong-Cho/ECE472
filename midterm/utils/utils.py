import math
import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import _LRScheduler

"""utils.py - Helper functions
"""

# import numpy as np
import torch
from torch.utils import model_zoo

from config import PRETRAINED_MODELS
from fvcore.nn import FlopCountAnalysis

def compute_flops(model, input):
    flops = FlopCountAnalysis(model, input)
    print(f'{flops.total() / 1e9} GFLOPS')


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
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]

def resize_positional_embedding_(posemb, posemb_new, has_class_token=True):
    #TODO: implement positional embedding for PiT
    """Rescale the grid of position embeddings in a sensible manner"""
    from scipy.ndimage import zoom

    # posemb.shape = (1, 197, 768)      note: sqrt(197-1) = 14 = patch_num
    # posemb_new.shape = (1, 5, 768)    note: sqrt(5-1) = 2 = patch_num_new

    assert len(posemb.shape) == len(posemb_new.shape), \
        'posemb and posemb_new shape is not compatible'
    
    if len(posemb.shape) == 3:
        # Deal with class token
        ntok_new = posemb_new.shape[1]
        if has_class_token:  # this means classifier == 'token'
            posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
            ntok_new -= 1
        else:
            posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

        # Get old and new grid sizes
        gs_old = int(np.sqrt(len(posemb_grid)))
        gs_new = int(np.sqrt(ntok_new))
        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1) # 196 x 768 -> 14 x 14 x 768

        # Rescale grid
        zoom_factor = (gs_new / gs_old, gs_new / gs_old, 1) # (2/14, 2/14, 1)
        posemb_grid = zoom(posemb_grid, zoom_factor, order=1)
        posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
        posemb_grid = torch.from_numpy(posemb_grid)

        # Deal with class token and return
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        return posemb
    
    elif len(posemb.shape) == 4:
        # Get old and new grid sizes
        gs_old = int(posemb.shape[2])
        gs_new = int(posemb_new.shape[2])

        # Rescale grid
        zoom_factor = (1, 1, gs_new / gs_old, gs_new / gs_old) # (2/14, 2/14, 1)
        posemb = zoom(posemb, zoom_factor, order=1)
        posemb = torch.from_numpy(posemb)
        return posemb


def load_pretrained_weights(
    model, 
    model_name=None, 
    weights_path=None, 
    load_first_conv=True, 
    load_fc=True, 
    load_repr_layer=False,
    resize_positional_embedding=False,
    verbose=True,
    strict=True,
):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): Full model (a nn.Module)
        model_name (str): Model name (e.g. B_16)
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_first_conv (bool): Whether to load patch embedding.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        resize_positional_embedding=False,
        verbose (bool): Whether to print on completion
    """
    assert bool(model_name) ^ bool(weights_path), 'Expected exactly one of model_name or weights_path'
    
    # Load or download weights
    if weights_path is None:
        url = PRETRAINED_MODELS[model_name]['url']
        if url:
            state_dict = model_zoo.load_url(url)
        else:
            raise ValueError(f'Pretrained model for {model_name} has not yet been released')
    else:
        state_dict = torch.load(weights_path)

    # Modifications to load partial state dict
    expected_missing_keys = []
    if not load_first_conv and 'patch_embedding.weight' in state_dict:
        expected_missing_keys += ['patch_embedding.weight', 'patch_embedding.bias']
    if not load_fc and 'fc.weight' in state_dict:
        expected_missing_keys += ['fc.weight', 'fc.bias']
    if not load_repr_layer and 'pre_logits.weight' in state_dict:
        expected_missing_keys += ['pre_logits.weight', 'pre_logits.bias']
    for key in expected_missing_keys:
        print(f'Missing keys popped: {key}')
        state_dict.pop(key)

    # Change size of positional embeddings
    if resize_positional_embedding: 
        posemb = state_dict['positional_embedding.pos_embedding']
        posemb_new = model.state_dict()['positional_embedding.pos_embedding']
        state_dict['positional_embedding.pos_embedding'] = \
            resize_positional_embedding_(posemb=posemb, posemb_new=posemb_new, 
                has_class_token=hasattr(model, 'class_token'))
        maybe_print('Resized positional embeddings from {} to {}'.format(
                    posemb.shape, posemb_new.shape), verbose)

    # Load state dict
    ret = model.load_state_dict(state_dict, strict=False)
    if strict:
        assert set(ret.missing_keys) == set(expected_missing_keys), \
            'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
        assert not ret.unexpected_keys, \
            'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)
        maybe_print('Loaded pretrained weights.', verbose)
    else:
        maybe_print('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys), verbose)
        maybe_print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys), verbose)
        return ret


def maybe_print(s: str, flag: bool):
    if flag:
        print(s)

def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
