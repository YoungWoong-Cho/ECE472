import math
import torch

from config import CONFIG, PIT_CONFIG
from functools import partial
from torch import nn

from model.common import Attention, FeedForward
from utils.helpers import as_tuple, truncated_normal


class SequentialTuple(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x):
        for module in self:
            x = module(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim=dim, hidden_dim=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(
            self, base_dim, depth, heads, mlp_ratio, pool=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for _ in range(depth)])
        self.pool = pool

    def forward(self, x):
        x, cls_tokens = x
        B, C, H, W = x.shape
        token_length = cls_tokens.shape[1]

        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.blocks(x)

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = x.transpose(1, 2).reshape(B, C, H, W)

        if self.pool is not None:
            x, cls_tokens = self.pool(x, cls_tokens)
        return x, cls_tokens


class ConvHeadPooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride):
        super().__init__()
        self.conv = nn.Conv2d(
            in_feature, out_feature,
            kernel_size=stride + 1,
            padding=stride // 2,
            stride=stride,
            groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):
        x = self.conv(x)
        cls_token = self.fc(cls_token)
        return x, cls_token


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, patch_size, stride)

    def forward(self, x):
        x = self.conv(x)
        return x


class PiT(nn.Module):
    """
    Based on "Rethinking Spatial Dimensions of Vision Transformers"
    https://arxiv.org/pdf/2103.16302.pdf
    """
    def __init__(
            self,
            name,
            image_size,
            patch_size,
            stride,
            base_dims,
            depth,
            heads,
            mlp_ratio,
            num_classes,
        ):
        super().__init__()

        self.name = name

        image_height, image_width = as_tuple(image_size)
        patch_height, patch_width = as_tuple(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        height = math.floor((image_height - patch_height) / stride + 1)
        width = math.floor((image_width - patch_width) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.pos_embed = nn.Parameter(torch.randn(1, base_dims[0] * heads[0], height, width))
        self.patch_embed = ConvEmbedding(3, base_dims[0] * heads[0], patch_size, stride)

        self.cls_token = nn.Parameter(torch.randn(1, 1, base_dims[0] * heads[0]))

        transformers = []
        for stage in range(len(depth)):
            pool = None
            if stage < len(heads) - 1:
                pool = ConvHeadPooling(
                    base_dims[stage] * heads[stage], base_dims[stage + 1] * heads[stage + 1], stride=2)
            transformers += [Transformer(
                base_dims[stage], depth[stage], heads[stage], mlp_ratio, pool=pool)
            ]
        self.transformers = SequentialTuple(*transformers)
        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        self.mlp_head = nn.Linear(self.embed_dim, num_classes)

        truncated_normal(self.pos_embed, std=.02)
        truncated_normal(self.cls_token, std=.02)

        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = self.patch_embed(x)        
        x = x + self.pos_embed
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        
        x, cls_tokens = self.transformers((x, cls_tokens))        
        cls_tokens = self.norm(cls_tokens)
        
        x_cls = self.mlp_head(cls_tokens[:, 0])
        return x_cls


def get_PiT(name):
    """
    Factory design for PiT models
    """
    assert name in PIT_CONFIG.keys(), \
        f'name should be one of the followings: {PIT_CONFIG.keys()}'
    
    model = PiT(
        name = name,
        image_size=CONFIG['model']['image_size'],
        patch_size=PIT_CONFIG[name]['patch_size'],
        num_classes=PIT_CONFIG[name]['num_classes'],
        stride=PIT_CONFIG[name]['stride'],
        base_dims=PIT_CONFIG[name]['base_dims'],
        depth=PIT_CONFIG[name]['depth'],
        heads=PIT_CONFIG[name]['heads'],
        mlp_ratio=PIT_CONFIG[name]['mlp_ratio']
    )

    return model
