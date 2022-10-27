from fvcore.nn import FlopCountAnalysis

import torch

from config import CONFIG
from model.ViT import ViT
from model.PiT import PiT

def compute_flops(model):
    input = torch.rand([1, CONFIG['train']['batch_size'], CONFIG['model']['img_size'], CONFIG['model']['img_size']])
    flops = FlopCountAnalysis(PiT_B16, input)
    flops.total()
    flops.by_operator()
    flops.by_module()
    flops.by_module_and_operator()