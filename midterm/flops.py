from utils.helpers import compute_flops
from config import CONFIG

from model.ViT import get_ViT
from model.PiT import get_PiT

import torch

if __name__=='__main__':
    model = get_ViT('B16')
    model.eval()
    input = torch.rand([1, 3, CONFIG['model']['img_size'], CONFIG['model']['img_size']])
    
    compute_flops(model, input)
