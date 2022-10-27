# """model.py - Model and module class for ViT.
#    They are built to mirror those in the official Jax implementation.
# """

# from symbol import try_stmt
# from typing import Optional
# import torch
# from torch import nn
# from torch.nn import functional as F

# from .transformer import Transformer
# from utils.utils import load_pretrained_weights, as_tuple
# from config import PRETRAINED_MODELS
# from pdb import set_trace as bp

# class PositionalEmbedding1D(nn.Module):
#     """Adds (optionally learned) positional embeddings to the inputs."""

#     def __init__(self, seq_len, dim):
#         super().__init__()
#         self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
    
#     def forward(self, x):
#         """Input has shape `(batch_size, seq_len, emb_dim)`"""
#         return x + self.pos_embedding


# class ViT(nn.Module):
#     """
#     Args:
#         name (str): Model name, e.g. 'B_16'
#         pretrained (bool): Load pretrained weights
#         in_channels (int): Number of channels in input data
#         num_classes (int): Number of classes, default 1000

#     References:
#         [1] https://openreview.net/forum?id=YicbFdNTTy
#     """

#     def __init__(
#         self, 
#         name: Optional[str] = None, 
#         pretrained: bool = False, 
#         patches: int = 16,
#         dim: int = 768,
#         ff_dim: int = 3072,
#         num_heads: int = 12,
#         num_layers: int = 12,
#         attention_dropout_rate: float = 0.0,
#         dropout_rate: float = 0.1,
#         representation_size: Optional[int] = None,
#         load_repr_layer: bool = False,
#         classifier: str = 'token',
#         positional_embedding: str = '1d',
#         in_channels: int = 3, 
#         image_size: Optional[int] = None,
#         num_classes: Optional[int] = None,
#     ):
#         super().__init__()
        
#         assert name in PRETRAINED_MODELS.keys(), \
#             'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
#         config = PRETRAINED_MODELS[name]['config']
#         patches = config['patches']
#         dim = config['dim']
#         ff_dim = config['ff_dim']
#         num_heads = config['num_heads']
#         num_layers = config['num_layers']
#         attention_dropout_rate = config['attention_dropout_rate']
#         dropout_rate = config['dropout_rate']
#         representation_size = config['representation_size']
#         classifier = config['classifier']
#         if image_size is None:
#             image_size = PRETRAINED_MODELS[name]['image_size']
#         if num_classes is None:
#             num_classes = PRETRAINED_MODELS[name]['num_classes']

#         self.image_size = image_size                

#         # Image and patch sizes
#         h, w = as_tuple(image_size)  # image sizes
#         fh, fw = as_tuple(patches)  # patch sizes
#         gh, gw = h // fh, w // fw  # number of patches
#         seq_len = gh * gw

#         # Patch embedding
#         self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

#         # Class token
#         if classifier == 'token':
#             self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
#             seq_len += 1
        
#         # Positional embedding
#         if positional_embedding.lower() == '1d':
#             self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
#         else:
#             raise NotImplementedError()
        
#         # Transformer
#         self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads, 
#                                        ff_dim=ff_dim, dropout=dropout_rate)
        
#         # Representation layer
#         if representation_size and load_repr_layer:
#             self.pre_logits = nn.Linear(dim, representation_size)
#             pre_logits_size = representation_size
#         else:
#             pre_logits_size = dim

#         # Classifier head
#         self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
#         self.fc = nn.Linear(pre_logits_size, num_classes)

#         # Initialize weights
#         self.init_weights()
        
#         # Load pretrained model
#         if pretrained:
#             pretrained_num_channels = 3
#             pretrained_num_classes = PRETRAINED_MODELS[name]['num_classes']
#             pretrained_image_size = PRETRAINED_MODELS[name]['image_size']
#             load_pretrained_weights(
#                 self, name, 
#                 load_first_conv=(in_channels == pretrained_num_channels),
#                 load_fc=(num_classes == pretrained_num_classes),
#                 load_repr_layer=load_repr_layer,
#                 resize_positional_embedding=(image_size != pretrained_image_size),
#             )
        
#     @torch.no_grad()
#     def init_weights(self):
#         def _init(m):
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
#                 if hasattr(m, 'bias') and m.bias is not None:
#                     nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
#         self.apply(_init)
#         nn.init.constant_(self.fc.weight, 0)
#         nn.init.constant_(self.fc.bias, 0)
#         nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
#         nn.init.constant_(self.class_token, 0)

#     def forward(self, x):
#         """Breaks image into patches, applies transformer, applies MLP head.

#         Args:
#             x (tensor): `b,c,fh,fw`
#         """
#         b, c, fh, fw = x.shape
#         x = self.patch_embedding(x)  # b,d,gh,gw
#         x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
#         if hasattr(self, 'class_token'):
#             x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
#         if hasattr(self, 'positional_embedding'): 
#             x = self.positional_embedding(x)  # b,gh*gw+1,d 
#         x = self.transformer(x)  # b,gh*gw+1,d
#         if hasattr(self, 'pre_logits'):
#             x = self.pre_logits(x)
#             x = torch.tanh(x)
#         if hasattr(self, 'fc'):
#             x = self.norm(x)[:, 0]  # b,d
#             x = self.fc(x)  # b,num_classes
#         return x

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

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

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)