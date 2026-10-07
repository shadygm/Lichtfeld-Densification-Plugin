# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


from typing import List, Tuple, Union

import torch
from ..device import device
import torch.nn as nn

from .blocks import _make_scratch, _make_fusion_block, custom_interpolate


class DPTHead(nn.Module):
    """
    DPT  Head for dense prediction tasks.

    This implementation follows the architecture described in "Vision Transformers for Dense Prediction"
    (https://arxiv.org/abs/2103.13413). The DPT head processes features from a vision transformer
    backbone and produces dense predictions by fusing multi-scale features.
    """

    def __init__(
        self,
        dim_in: int,
        out_dim: int,
        patch_size: int = 16,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
        align_corners: bool = True,
    ) -> None:
        super(DPTHead, self).__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.align_corners = align_corners
        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=oc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for oc in out_channels
            ]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False)

        # Attach additional modules to scratch.
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32

        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1, kernel_size=3, stride=1, padding=1
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1,
                head_features_1 // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            conv2_in_channels = head_features_1 // 2

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(
                    conv2_in_channels,
                    head_features_2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, out_dim, kernel_size=1, stride=1, padding=0),
            )

    def forward(
        self,
        tokens: torch.Tensor | List[torch.Tensor],
        *,
        img_A: torch.Tensor | None = None,
        img_B: torch.Tensor | None = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(tokens, torch.Tensor):
            B, H, W, D = tokens.shape
        else:
            B, H, W, D = tokens[0].shape
        return self._forward_impl(tokens, H, W)

    def _forward_impl(
        self,
        aggregated_tokens_list_or_tokens: List[torch.Tensor] | torch.Tensor,
        patch_h: int,
        patch_w: int,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            assert not isinstance(aggregated_tokens_list_or_tokens, torch.Tensor), (
                "aggregated_tokens_list_or_tokens should be a list of tensors"
            )
            aggregated_tokens_list = aggregated_tokens_list_or_tokens
            if len(aggregated_tokens_list) != len(self.out_channels):
                assert len(self.out_channels) % len(aggregated_tokens_list) == 0, (
                    "out_channels must be a multiple of intermediate_layer_idx"
                )
                factor = len(self.out_channels) // len(aggregated_tokens_list)
                aggregated_tokens_list = [
                    x
                    for xs in [
                        [aggregated_tokens_list[i]] * factor
                        for i in range(len(aggregated_tokens_list))
                    ]
                    for x in xs
                ]

            B = aggregated_tokens_list[0].shape[0]

            H, W = patch_h * self.patch_size, patch_w * self.patch_size

            out = []

            for dpt_idx in range(len(self.out_channels)):
                x = aggregated_tokens_list[dpt_idx]

                x = x.reshape(B, -1, x.shape[-1])

                x = self.norm(x)

                x = x.permute(0, 2, 1).reshape(
                    (x.shape[0], x.shape[-1], patch_h, patch_w)
                )

                x = self.projects[dpt_idx](x)
                if self.pos_embed:
                    x = self._apply_pos_embed(x, W, H)
                x = self.resize_layers[dpt_idx](x)

                out.append(x)

            # Fuse features from multiple layers.
            out = self.scratch_forward(out)
            # Interpolate fused output to match target image resolution.
            out = custom_interpolate(
                out,
                (
                    int(patch_h * self.patch_size / self.down_ratio),
                    int(patch_w * self.patch_size / self.down_ratio),
                ),
                mode="bilinear",
                align_corners=self.align_corners,
            )

            if self.pos_embed:
                out = self._apply_pos_embed(out, W, H)

            if self.feature_only:
                return out.view(B, *out.shape[1:])
        # float it for precision
        out = out.float()
        out = self.scratch.output_conv2(out)
        out = out.permute(0, 2, 3, 1)
        return out

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out


################################################################################
# Modules
################################################################################
