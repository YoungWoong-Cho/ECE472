import torch
from torch import nn

from einops import repeat
from einops.layers.torch import Rearrange

from config import CONFIG, VIT_CONFIG
from model.transformer import FeedForward, Attention
from utils.helpers import as_tuple


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_heads = heads, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(
            self,
            image_size, 
            patch_size,
            num_classes,
            dim,
            depth,
            heads,
            mlp_dim,
            channels = 3,
            dropout = 0.,
            emb_dropout = 0.
        ):
        super().__init__()

        image_height, image_width = as_tuple(image_size)
        patch_height, patch_width = as_tuple(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x[:, 0]

        return self.mlp_head(x)


def get_ViT(name):
    assert name in VIT_CONFIG.keys(), \
        f'name should be one of the followings: {VIT_CONFIG.keys()}'
    
    model = ViT(
        image_size=CONFIG['model']['image_size'],
        patch_size=VIT_CONFIG[name]['patch_size'],
        num_classes=VIT_CONFIG[name]['num_classes'],
        dim=VIT_CONFIG[name]['dim'],
        depth=VIT_CONFIG[name]['depth'],
        heads=VIT_CONFIG[name]['heads'],
        mlp_dim=VIT_CONFIG[name]['mlp_dim']
    )

    return model
