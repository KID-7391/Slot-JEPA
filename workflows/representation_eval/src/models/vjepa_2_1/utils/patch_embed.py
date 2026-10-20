# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from einops import rearrange

import torch.nn as nn


class AudioPatchEmbed(nn.Module):
    """
    Audio to Patch Embedding
    """

    def __init__(self, freq_bands=128, tubelet_size=2, embed_dim=768):
        super().__init__()
        self.freq_bands = freq_bands
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(freq_bands, tubelet_size), stride=(freq_bands, tubelet_size))

    def forward(self, x):
        x = rearrange(x, "b t c f -> b c f t")
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, patch_size=16, patch_stride=None, patch_padding=0, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_size if patch_stride is None else patch_stride
        self.patch_padding = patch_padding
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=self.patch_stride,
            padding=self.patch_padding,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        patch_size=16,
        patch_stride=None,
        patch_padding=0,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_size if patch_stride is None else patch_stride
        self.patch_padding = patch_padding
        self.tubelet_size = tubelet_size

        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, self.patch_stride, self.patch_stride),
            padding=(0, self.patch_padding, self.patch_padding),
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
