from abc import abstractmethod
import math

import torch
import torch as th
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction

from inpainting.training.modules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    normalization,
    timestep_embedding,
)
from inpainting.training.modules.mambair_arch import VSSModule


# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def exists(val):
    return val is not None

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            # elif isinstance(layer, SpatialTransformer):
            #     x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        if self.use_conv:
            x = self.conv(x)
        return x

# class TransposedUpsample(nn.Module):
#     'Learned 2x upsampling without padding'
#     def __init__(self, channels, out_channels=None, ks=5):
#         super().__init__()
#         self.channels = channels
#         self.out_channels = out_channels or channels
#
#         self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)
#
#     def forward(self,x):
#         return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class GetMambaLayer(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1):
        super().__init__()
        mamba_layers = []
        for _ in range(depth):
            mamba_layers.append(VSSModule(hidden_dim=in_channels))

        self.mamba_layer = nn.Sequential(*mamba_layers)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous() # [B,H,W,C]
        x = self.mamba_layer(x)
        x = x.permute(0, 3, 1, 2).contiguous() #[B,C,H,W]
        return x

class OSSModule(nn.Module):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
    ):
        super().__init__()

        self.mamba_module = GetMambaLayer(out_ch, out_ch)

    def forward(self, x):
        h = self.mamba_module(x)
        return h

class MFFM(nn.Module):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.in_ch//2, self.out_ch//2, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(self.in_ch//2, self.out_ch//2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.out_ch//2, self.out_ch//2, kernel_size=3, padding=1),
        )
        self.out_conv = nn.Conv2d(out_ch, out_ch, kernel_size=1)

        if in_ch == out_ch:
            self.skip_connect = nn.Identity()
        else:
            self.skip_connect = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        h = x

        conv_h_1, conv_h_2 = torch.chunk(h, 2, dim=1)

        conv_1 = self.conv_1(conv_h_1)
        conv_2 = self.conv_2(conv_h_2)

        connect_x = torch.cat([conv_1, conv_2], dim=1)
        connect_x = self.out_conv(connect_x)
        return connect_x + self.skip_connect(h)

class MFFB(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels * 2)
        )

        self.dropout = nn.Dropout(self.dropout)
        self.effn = MFFM(channels, out_channels)
        self.effn_2 = MFFM(out_channels, out_channels)

        if channels == out_channels:
            self.skip_connect = nn.Identity()
        else:
            self.skip_connect = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x, emb):
        emb_out = self.emb_layers(emb).type(x.dtype)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]

        h = self.effn(x)

        scale, shift = emb_out.chunk(2, dim=1)
        h = h * (scale + 1) + shift

        h = self.dropout(h)
        h = self.effn_2(h)
        h = h + self.skip_connect(x)
        return h

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
            nn.SiLU(),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, True, dims)
            self.x_upd = Upsample(channels, True, dims)
        elif down:
            self.h_upd = Downsample(channels, True, dims)
            self.x_upd = Downsample(channels, True, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.Dropout(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1),
            nn.SiLU(),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            h = x
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = self.in_layers(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            # h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,

        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32

        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4

        sinu_pos_emb = SinusoidalPosEmb(model_channels)
        self.time_embed = nn.Sequential(
            sinu_pos_emb,
            linear(model_channels, time_embed_dim),
            nn.GELU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(in_channels, model_channels, 3, padding=1),
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            ch = model_channels * mult
            for _ in range(num_res_blocks):
                layers = [

                    MFFB(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(OSSModule(ch, ch))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = model_channels * channel_mult[level + 1]
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            MFFB(
                ch,
                time_embed_dim,
                dropout,
                out_channels=ch,
            ),


            OSSModule(ch, ch),


            MFFB(
                ch,
                time_embed_dim,
                dropout,
                out_channels=ch,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    MFFB(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(OSSModule(ch, ch))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            conv_nd(dims, model_channels, out_channels, 3, padding=1)
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        if isinstance(timesteps, int) or isinstance(timesteps, float):
            timesteps = torch.tensor([timesteps]).to(x.device)
        hs = []
        emb = self.time_embed(timesteps)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        h = self.out(h)
        return h

