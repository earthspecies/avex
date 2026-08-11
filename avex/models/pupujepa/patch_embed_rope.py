import torch.nn as nn
import collections.abc
from itertools import repeat
from typing import Optional, Callable, Union


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
to_1tuple = _ntuple(1)


class PatchEmbedRoPE(nn.Module):
    def __init__(
        self,
        img_size: Optional[Union[tuple, int]] = (200, 80),
        patch_size: Optional[Union[tuple, int]] = (4, 16),
        in_chans: Optional[int] = 1,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        frequency_first: bool = False,
    ) -> None:
        super().__init__()
        self.img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.num_patches = grid_size[0] * grid_size[1]
        self.flatten = flatten
        self.embed_dim = embed_dim
        self.frequency_first = frequency_first
        print(
            f"!!!!!!!!! ATTENTION: PatchEmbed is using frequency_first = {frequency_first} !!!!!!!!!"
        )

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        outputs = self.proj(x)

        # maintain ordering of dimensions as per original jax implementation
        if self.frequency_first:
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.permute(0, 2, 3, 1)

        if self.flatten:
            outputs = outputs.reshape(B, -1, self.embed_dim)

        outputs = self.norm_layer(outputs)
        return outputs
