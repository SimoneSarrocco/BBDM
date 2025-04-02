# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence
import math
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.layers import Act
from monai.utils.misc import ensure_tuple_rep
import numpy as np
from model.VQGAN.vector_quantizer import EMAQuantizer, VectorQuantizer
# from model.VQGAN.quantize import VectorQuantizer2 as VectorQuantizer

__all__ = ["VQVAE"]


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


class VQVAEResidualUnit(nn.Module):
    """
    Implementation of the ResidualLayer used in the VQVAE network as originally used in Morphology-preserving
    Autoregressive 3D Generative Modelling of the Brain by Tudosiu et al. (https://arxiv.org/pdf/2209.03177.pdf) and
    the original implementation that can be found at
    https://github.com/AmigoLab/SynthAnatomy/blob/main/src/networks/vqvae/baseline.py#L150.

    Args:
        spatial_dims: number of spatial spatial_dims of the input data.
        num_channels: number of input channels.
        num_res_channels: number of channels in the residual layers.
        act: activation type and arguments. Defaults to RELU.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term. Defaults to True.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        num_res_channels: int,
        act: tuple | str | None = Act.RELU,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.spatial_dims = spatial_dims
        self.num_channels = num_channels
        self.num_res_channels = num_res_channels
        self.act = act
        self.dropout = dropout
        self.bias = bias

        self.conv1 = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=self.num_channels,
            out_channels=self.num_res_channels,
            adn_ordering="DA",
            act=self.act,
            dropout=self.dropout,
            bias=self.bias,
        )

        self.conv2 = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=self.num_res_channels,
            out_channels=self.num_channels,
            bias=self.bias,
            conv_only=True,
        )

    def forward(self, x):
        return torch.nn.functional.relu(x + self.conv2(self.conv1(x)), True)


class Encoder(nn.Module):
    """
    Encoder module for VQ-VAE.

    Args:
        spatial_dims: number of spatial spatial_dims.
        in_channels: number of input channels.
        out_channels: number of channels in the latent space (embedding_dim).
        num_channels: number of channels at each level.
        num_res_layers: number of sequential residual layers at each level.
        num_res_channels: number of channels in the residual layers at each level.
        downsample_parameters: A Tuple of Tuples for defining the downsampling convolutions. Each Tuple should hold the
            following information stride (int), kernel_size (int), dilation (int) and padding (int).
        dropout: dropout ratio.
        act: activation type and arguments.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_channels: Sequence[int],
        num_res_layers: int,
        num_res_channels: Sequence[int],
        downsample_parameters: Sequence[Sequence[int, int, int, int], ...],
        dropout: float,
        act: tuple | str | None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.num_res_layers = num_res_layers
        self.num_res_channels = num_res_channels
        self.downsample_parameters = downsample_parameters
        self.dropout = dropout
        self.act = act

        blocks = []

        for i in range(len(self.num_channels)):
            blocks.append(
                Convolution(
                    spatial_dims=self.spatial_dims,
                    in_channels=self.in_channels if i == 0 else self.num_channels[i - 1],
                    out_channels=self.num_channels[i],
                    strides=self.downsample_parameters[i][0],
                    kernel_size=self.downsample_parameters[i][1],
                    adn_ordering="DA",
                    act=self.act,
                    dropout=None if i == 0 else self.dropout,
                    dropout_dim=1,
                    dilation=self.downsample_parameters[i][2],
                    padding=self.downsample_parameters[i][3],
                )
            )

            for _ in range(self.num_res_layers):
                blocks.append(
                    VQVAEResidualUnit(
                        spatial_dims=self.spatial_dims,
                        num_channels=self.num_channels[i],
                        num_res_channels=self.num_res_channels[i],
                        act=self.act,
                        dropout=self.dropout,
                    )
                )

        blocks.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=self.num_channels[len(self.num_channels) - 1],
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x



class EncoderBBDM(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    """
    Decoder module for VQ-VAE.

    Args:
        spatial_dims: number of spatial spatial_dims.
        in_channels: number of channels in the latent space (embedding_dim).
        out_channels: number of output channels.
        num_channels: number of channels at each level.
        num_res_layers: number of sequential residual layers at each level.
        num_res_channels: number of channels in the residual layers at each level.
        upsample_parameters: A Tuple of Tuples for defining the upsampling convolutions. Each Tuple should hold the
            following information stride (int), kernel_size (int), dilation (int), padding (int), output_padding (int).
        dropout: dropout ratio.
        act: activation type and arguments.
        output_act: activation type and arguments for the output.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_channels: Sequence[int],
        num_res_layers: int,
        num_res_channels: Sequence[int],
        upsample_parameters: Sequence[Sequence[int, int, int, int], ...],
        dropout: float,
        act: tuple | str | None,
        output_act: tuple | str | None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.num_res_layers = num_res_layers
        self.num_res_channels = num_res_channels
        self.upsample_parameters = upsample_parameters
        self.dropout = dropout
        self.act = act
        self.output_act = output_act

        reversed_num_channels = list(reversed(self.num_channels))

        blocks = []
        blocks.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=self.in_channels,
                out_channels=reversed_num_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        reversed_num_res_channels = list(reversed(self.num_res_channels))
        for i in range(len(self.num_channels)):
            for _ in range(self.num_res_layers):
                blocks.append(
                    VQVAEResidualUnit(
                        spatial_dims=self.spatial_dims,
                        num_channels=reversed_num_channels[i],
                        num_res_channels=reversed_num_res_channels[i],
                        act=self.act,
                        dropout=self.dropout,
                    )
                )

            blocks.append(
                Convolution(
                    spatial_dims=self.spatial_dims,
                    in_channels=reversed_num_channels[i],
                    out_channels=self.out_channels if i == len(self.num_channels) - 1 else reversed_num_channels[i + 1],
                    strides=self.upsample_parameters[i][0],
                    kernel_size=self.upsample_parameters[i][1],
                    adn_ordering="DA",
                    act=self.act,
                    dropout=self.dropout if i != len(self.num_channels) - 1 else None,
                    norm=None,
                    dilation=self.upsample_parameters[i][2],
                    conv_only=i == len(self.num_channels) - 1,
                    is_transposed=True,
                    padding=self.upsample_parameters[i][3],
                    output_padding=self.upsample_parameters[i][4],
                )
            )

        if self.output_act:
            blocks.append(Act[self.output_act]())

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class DecoderBBDM(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VQVAE(nn.Module):
    """
    Vector-Quantised Variational Autoencoder (VQ-VAE) used in Morphology-preserving Autoregressive 3D Generative
    Modelling of the Brain by Tudosiu et al. (https://arxiv.org/pdf/2209.03177.pdf) and the original implementation
    that can be found at https://github.com/AmigoLab/SynthAnatomy/blob/main/src/networks/vqvae/baseline.py#L163/

    Args:
        spatial_dims: number of spatial spatial_dims.
        in_channels: number of input channels.
        out_channels: number of output channels.
        downsample_parameters: A Tuple of Tuples for defining the downsampling convolutions. Each Tuple should hold the
            following information stride (int), kernel_size (int), dilation (int) and padding (int).
        upsample_parameters: A Tuple of Tuples for defining the upsampling convolutions. Each Tuple should hold the
            following information stride (int), kernel_size (int), dilation (int), padding (int), output_padding (int).
        num_res_layers: number of sequential residual layers at each level.
        num_channels: number of channels at each level.
        num_res_channels: number of channels in the residual layers at each level.
        num_embeddings: VectorQuantization number of atomic elements in the codebook.
        embedding_dim: VectorQuantization number of channels of the input and atomic elements.
        commitment_cost: VectorQuantization commitment_cost.
        decay: VectorQuantization decay.
        epsilon: VectorQuantization epsilon.
        act: activation type and arguments.
        dropout: dropout ratio.
        output_act: activation type and arguments for the output.
        ddp_sync: whether to synchronize the codebook across processes.
        use_checkpointing if True, use activation checkpointing to save memory.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_channels: Sequence[int] | int = (96, 96, 192),
        num_res_layers: int = 3,
        num_res_channels: Sequence[int] | int = (96, 96, 192),
        downsample_parameters: Sequence[Sequence[int, int, int, int], ...]
        | Sequence[int, int, int, int] = ((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
        upsample_parameters: Sequence[Sequence[int, int, int, int, int], ...]
        | Sequence[int, int, int, int] = ((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
        num_embeddings: int = 32,
        embedding_dim: int = 64,
        embedding_init: str = "normal",
        commitment_cost: float = 0.25,
        decay: float = 0.5,
        epsilon: float = 1e-5,
        dropout: float = 0.0,
        act: tuple | str | None = Act.RELU,
        output_act: tuple | str | None = None,
        ddp_sync: bool = True,
        use_checkpointing: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.use_checkpointing = use_checkpointing

        if isinstance(num_res_channels, int):
            num_res_channels = ensure_tuple_rep(num_res_channels, len(num_channels))

        if len(num_res_channels) != len(num_channels):
            raise ValueError(
                "`num_res_channels` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )

        if not all(isinstance(values, (int, Sequence)) for values in downsample_parameters):
            raise ValueError("`downsample_parameters` should be a single tuple of integer or a tuple of tuples.")

        if not all(isinstance(values, (int, Sequence)) for values in upsample_parameters):
            raise ValueError("`upsample_parameters` should be a single tuple of integer or a tuple of tuples.")

        if all(isinstance(values, int) for values in upsample_parameters):
            upsample_parameters = (upsample_parameters,) * len(num_channels)

        if all(isinstance(values, int) for values in downsample_parameters):
            downsample_parameters = (downsample_parameters,) * len(num_channels)

        for parameter in downsample_parameters:
            if len(parameter) != 4:
                raise ValueError("`downsample_parameters` should be a tuple of tuples with 4 integers.")

        for parameter in upsample_parameters:
            if len(parameter) != 5:
                raise ValueError("`upsample_parameters` should be a tuple of tuples with 5 integers.")

        if len(downsample_parameters) != len(num_channels):
            raise ValueError(
                "`downsample_parameters` should be a tuple of tuples with the same length as `num_channels`."
            )

        if len(upsample_parameters) != len(num_channels):
            raise ValueError(
                "`upsample_parameters` should be a tuple of tuples with the same length as `num_channels`."
            )

        self.num_res_layers = num_res_layers
        self.num_res_channels = num_res_channels

        self.encoder = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=embedding_dim,
            num_channels=num_channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            downsample_parameters=downsample_parameters,
            dropout=dropout,
            act=act,
        )

        self.decoder = Decoder(
            spatial_dims=spatial_dims,
            in_channels=embedding_dim,
            out_channels=out_channels,
            num_channels=num_channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            upsample_parameters=upsample_parameters,
            dropout=dropout,
            act=act,
            output_act=output_act,
        )

        self.quantizer = VectorQuantizer(
            quantizer=EMAQuantizer(
                spatial_dims=spatial_dims,
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                decay=decay,
                epsilon=epsilon,
                embedding_init=embedding_init,
                ddp_sync=ddp_sync,
            )
        )

        """
        self.encoder = EncoderBBDM(
            ch=num_channels[0],
            out_ch=out_channels,
            ch_mult=num_channels,
            num_res_blocks=num_res_layers,
            attn_resolutions=[],
            in_channels=in_channels,
            resolution=512,
            z_channels=embedding_dim,
            double_z=False,
        )

        self.decoder = DecoderBBDM(
            ch=embedding_dim,
            out_ch=out_channels,
            ch_mult=num_channels,
            num_res_blocks=num_res_layers,
            attn_resolutions=[],
            in_channels=in_channels,
            resolution=512,
            z_channels=embedding_dim,
        )

        self.quantizer = VectorQuantizer(
            self.num_embeddings,
            self.embedding_dim,
            beta=0.25,
            remap=None,
            sane_index_shape=False,
        )

        self.quant_conv = torch.nn.Conv2d(embedding_dim, embedding_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embedding_dim, embedding_dim, 1)
        """

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.encoder, images, use_reentrant=False)
        else:
            return self.encoder(images)

    def quantize(self, encodings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_loss, x = self.quantizer(encodings)
        # quant, embed_loss, info = self.quantizer(encodings)
        return x, x_loss
        # return quant, embed_loss, info

    def decode(self, quantizations: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.decoder, quantizations, use_reentrant=False)
        else:
            # quantizations = self.post_quant_conv(quantizations)
            return self.decoder(quantizations)

    def index_quantize(self, images: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantize(self.encode(images=images))

    def decode_samples(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.decode(self.quantizer.embed(embedding_indices))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quantizations, quantization_losses = self.quantize(self.encode(images))
        reconstruction = self.decode(quantizations)

        return reconstruction, quantization_losses

    def encode_stage_2_inputs(self, x: torch.Tensor, quantized: bool = True) -> torch.Tensor:
        z = self.encode(x)
        e, _ = self.quantize(z)
        if quantized:
            return e
        return z

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        e, _ = self.quantize(z)
        image = self.decode(e)
        return image
