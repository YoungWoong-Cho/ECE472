from utils.utils import compute_flops
from config import CONFIG

from model.ViT import ViT
from model.PiT import PiT

import torch

if __name__=='__main__':
    # PiT_B16 = PiT(img_size=CONFIG['model']['img_size'], num_classes=100)
    # PiT_S16 = PiT(img_size=CONFIG['model']['img_size'],
    #               num_classes=100,
    #               patch_size=16,
    #               stride=8,
    #               base_dims=[48, 48, 48],
    #               depth=[2, 6, 4],
    #               heads=[3, 6, 12],
    #               mlp_ratio=4,)
    # PiT_Ti16 = PiT(img_size=CONFIG['model']['img_size'],
    #                 num_classes=100,
    #                 patch_size=16,
    #                 stride=8,
    #                 base_dims=[32, 32, 32],
    #                 depth=[2, 6, 4],
    #                 heads=[2, 4, 8],
    #                 mlp_ratio=4,)

    # ViT_B16 = ViT(image_size=CONFIG['model']['img_size'], patch_size=16, num_classes=100,
    #               dim=768, depth=12, heads=12, mlp_dim=3072)
    # ViT_S16 = ViT(image_size=CONFIG['model']['img_size'], patch_size=16, num_classes=100,
    #                 dim=384, depth=12, heads=6, mlp_dim=3072)
    ViT_Ti16 = ViT(image_size=CONFIG['model']['img_size'], patch_size=16, num_classes=100,
                    dim=192, depth=12, heads=3, mlp_dim=3072)

    model = ViT_Ti16
    model.eval()
    # torch.Size([512, 3, 112, 112])
    input = torch.rand([1, 3, CONFIG['model']['img_size'], CONFIG['model']['img_size']])
    
    compute_flops(model, input)
