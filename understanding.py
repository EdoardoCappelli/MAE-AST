import torch
import torch.nn as nn
import numpy as np
from timm.layers.patch_embed import PatchEmbed
from types import SimpleNamespace   
import math
from positional_embedding import SinusoidalPositionalEncoding

import torch
import math


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


config = SimpleNamespace(
    kernel_size = (1,1),
    stride = (1,1),
    padding = 0,
    enc_embed_dim=4,           
    patch_size=(1, 1),
    img_size = (3,3),
    num_channels = 1,
    )

# Parametri comuni per il confronto


x_np = np.arange(0, 9).reshape(3, 3)
x_tensor = torch.tensor(x_np, dtype=torch.float32)

x_tensor_unsqueezed = x_tensor.unsqueeze(0).unsqueeze(0)
print(f"x_tensor_unsqueezed:\n{x_tensor_unsqueezed}")
print(f"x_tensor_unsqueezed:\n{x_tensor_unsqueezed.shape}")


patch_embedding = PatchEmbed(
    config.img_size,
    config.patch_size,
    config.num_channels,
    config.enc_embed_dim
) 

embedding = patch_embedding(x_tensor_unsqueezed)
print(f"patch_embedding:\n{embedding.shape}")
print(f"patch_embedding:\n{embedding}")



pos_embed_enc = SinusoidalPositionalEncoding(embed_dim=config.enc_embed_dim, height=config.img_size[0], width=config.img_size[1], cls_token=True)
print(f"pos_embed_enc:\n{pos_embed_enc.shape}")
print(f"pos_embed_enc:\n{pos_embed_enc}")
print(f"pos_embed_enc[:,1:,:]:\n{pos_embed_enc[:,1:,:]}")

# pos_embed = nn.Parameter(torch.zeros(1, patch_embedding.num_patches + 1, config.enc_embed_dim), requires_grad=False)  # fixed sin-cos embedding
# pos_embed = get_2d_sincos_pos_embed(pos_embed.shape[-1], int(patch_embedding.num_patches**.5), cls_token=True)
# print(f"pos_embed:\n{pos_embed.shape}")
# print(f"pos_embed:\n{pos_embed}")
