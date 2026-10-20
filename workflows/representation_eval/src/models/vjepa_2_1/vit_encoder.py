"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
------------------------------------------------------------------------------

modelcustom API requirements:

API requirements for Encoder module:
    1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip (shape=[batch_size x num_channels x num_frames x height x width])
        :returns: (Tensor) Representations of video clip (shape=[batch_size x num_encoder_tokens x feature_dim])
    2) Needs to have a public attribute called 'embed_dim' (int) describing its
        output feature dimension.

API requirements for Predictor module:
    1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip tokens (shape=[batch_size x num_encoder_tokens x feature_dim])
        :param anticipation_time: (Tensor) Seconds into the future to predict for each sample in batch
            (shape=[batch_size])
        :returns: (Tensor) Representations of future frames (shape=[batch_size x num_output_tokens x feature_dim])
    2) Needs to have a public attribute called 'embed_dim' (int) describing its
        output feature dimension.
"""

import logging

import torch

from . import vision_transformer as vit

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_module(
    resolution: int,
    model_kwargs: dict,
    wrapper_kwargs: dict,
    **kwargs,
):

    img_as_video_nframes = wrapper_kwargs.get("img_as_video_nframes")
    # --
    enc_kwargs = model_kwargs["encoder"]
    enc_model_name = enc_kwargs.get("model_name")

    model = vit.__dict__[enc_model_name](
        img_size=resolution,
        num_frames=img_as_video_nframes,
        **enc_kwargs,
    )

    def forward_prehook(module, input):
        x = input[0]
        if x.ndim == 4:
            x = x.unsqueeze(2).repeat(1, 1, img_as_video_nframes, 1, 1)
            return (x,)
        if x.ndim == 5:
            return (x,)
        raise ValueError(f"Expected 4D or 5D input for VJEPA encoder, got shape={tuple(x.shape)}")

    model.register_forward_pre_hook(forward_prehook)

    return model
