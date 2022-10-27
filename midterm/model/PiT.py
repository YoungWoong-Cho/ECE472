import math
import torch

from config import CONFIG, PIT_CONFIG
from functools import partial
from typing import Tuple
from torch import nn

from utils.helpers import as_tuple, trunc_normal_
from .PiT_transformer import Block


class SequentialTuple(nn.Sequential):
    def __init__(self, *args):
        super(SequentialTuple, self).__init__(*args)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        for module in self:
            x = module(x)
        return x

class Transformer(nn.Module):
    def __init__(
            self, base_dim, depth, heads, mlp_ratio, pool=None, drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(depth)])

        self.pool = pool

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def __init__(self, in_feature, out_feature, stride, padding_mode='zeros'):
        super(ConvHeadPooling, self).__init__()

        self.conv = nn.Conv2d(
            in_feature, out_feature, kernel_size=stride + 1, padding=stride // 2, stride=stride,
            padding_mode=padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super(ConvEmbedding, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=patch_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class PiT(nn.Module):
    """ Pooling-based Vision Transformer

    A PyTorch implement of 'Rethinking Spatial Dimensions of Vision Transformers'
        - https://arxiv.org/abs/2103.16302
    """
    def __init__(
        self,
        pretrained_weight=None,
        img_size=224,
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        num_classes=1000,
        in_chans=3,
        distilled=False,
        attn_drop_rate=.0,
        drop_rate=.0,
        drop_path_rate=.0
        ):
        super().__init__()

        padding = 0
        img_size = as_tuple(img_size)
        patch_size = as_tuple(patch_size)
        height = math.floor((img_size[0] + 2 * padding - patch_size[0]) / stride + 1)
        width = math.floor((img_size[1] + 2 * padding - patch_size[1]) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes
        self.num_tokens = 2 if distilled else 1

        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, base_dims[0] * heads[0], height, width))
        self.patch_embed = ConvEmbedding(in_chans, base_dims[0] * heads[0], patch_size, stride, padding)

        self.cls_token = nn.Parameter(torch.randn(1, self.num_tokens, base_dims[0] * heads[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        transformers = []
        # stochastic depth decay rule
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depth)).split(depth)]
        for stage in range(len(depth)):
            pool = None
            if stage < len(heads) - 1:
                pool = ConvHeadPooling(
                    base_dims[stage] * heads[stage], base_dims[stage + 1] * heads[stage + 1], stride=2)
            transformers += [Transformer(
                base_dims[stage], depth[stage], heads[stage], mlp_ratio, pool=pool,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_prob=dpr[stage])
            ]
        self.transformers = SequentialTuple(*transformers)
        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        # Classifier head
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) \
            if num_classes > 0 and distilled else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        
        if pretrained_weight is not None:
            pretrained_num_channels = 3
            pretrained_num_classes = 1000
            pretrained_image_size = 224
            self.load_pretrained_weights(
                weights_path=pretrained_weight,
                load_first_conv=(in_chans == pretrained_num_channels),
                load_fc=(num_classes == pretrained_num_classes),
                load_repr_layer=False,
                resize_positional_embedding=(32 != pretrained_image_size),
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) \
            if num_classes > 0 and self.num_tokens == 2 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        # 224 x 224 -> 31 x 31
        # 32 x 32 -> 5 x 5
        
        x = self.pos_drop(x + self.pos_embed)
        # x.shape = (B, 256, 31, 31)
        # self.pos_embed.shape = (1, 256, 31, 31)
        
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1) # cls_tokens.shape = (B, 1, 256)
        
        x, cls_tokens = self.transformers((x, cls_tokens))
        # x.shape = (B, 1024, 8, 8)
        # cls_tokens.shape = (B, 1, 1024)
        
        cls_tokens = self.norm(cls_tokens)
        return cls_tokens
    
    def forward(self, x):
        x = self.forward_features(x)
        x_cls = self.head(x[:, 0])
        if self.num_tokens > 1:
            x_dist = self.head_dist(x[:, 1])
            if self.training and not torch.jit.is_scripting():
                return x_cls, x_dist
            else:
                return (x_cls + x_dist) / 2
        else:
            return x_cls

def get_PiT(name):
    assert name in PIT_CONFIG.keys(), \
        f'name should be one of the followings: {PIT_CONFIG.keys()}'
    
    model = PiT(
        img_size=CONFIG['model']['img_size'],
        patch_size=PIT_CONFIG[name]['patch_size'],
        num_classes=PIT_CONFIG[name]['num_classes'],
        stride=PIT_CONFIG[name]['stride'],
        base_dims=PIT_CONFIG[name]['base_dims'],
        depth=PIT_CONFIG[name]['depth'],
        heads=PIT_CONFIG[name]['heads'],
        mlp_ratio=PIT_CONFIG[name]['mlp_ratio']
    )

    return model
