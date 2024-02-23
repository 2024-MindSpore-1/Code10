from functools import partial 
import mindspore.nn as nn
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def constrcut(self, x):
        return self.fn(self.norm(x)) + x


def mlp(dim, expansion_factor=4.0, dropout=0., dense=nn.Dense):
    inner_dim = int(dim * expansion_factor)
    return nn.SequentialCell([
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)]
    )


def mlp_mixer(*, image_size, channels, patch_size, dim, depth, num_classes,
              expansion_factor=4, expansion_factor_token=0.5, dropout=0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chn_first, chn_last = partial(nn.Conv1d, kernel_size=1), nn.Dense

    return nn.SequentialCell([
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.SequentialCell([
            PreNormResidual(dim, mlp(num_patches, expansion_factor, dropout, chn_first)),
            PreNormResidual(dim, mlp(dim, expansion_factor_token, dropout, chn_last))
        ]) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Dense(dim, num_classes)
        ])


def mlp_mixer_backbone(*, image_size, channels, patch_size, dim, depth,
              expansion_factor=4, expansion_factor_token=0.5, dropout=0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chn_first, chn_last = partial(nn.Conv1d, kernel_size=1), nn.Dense

    return nn.SequentialCell([
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Dense((patch_size ** 2) * channels, dim),
        *[nn.SequentialCell([
            PreNormResidual(dim, mlp(num_patches, expansion_factor, dropout, chn_first)),
            PreNormResidual(dim, mlp(dim, expansion_factor_token, dropout, chn_last))
        ]) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean')
    ])


def mlp_backbone(*, image_size, channels, patch_size=16, dim, depth, expansion_factor=0.5, dropout=0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    # num_patches = (image_h // patch_size) * (image_w // patch_size)
    chn_first, chn_last = partial(nn.Conv1d, kernel_size=1), nn.Dense

    return nn.SequentialCell([
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        nn.Dense((patch_size ** 2) * channels, dim),
        *[nn.SequentialCell([
            # PreNormResidual(dim, mlp(num_patches, expansion_factor, dropout, chn_first)),
            PreNormResidual(dim, mlp(dim, expansion_factor, dropout, chn_last))
        ]) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean')
    ])