from typing import Any, Optional, Union, List

import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import numpy as np
import os, sys
from torch import pi, sin, exp, tan, sqrt, log
import math
from numpy import arctan
import pytorch_lightning as pl

import torch.nn.functional as F

from utils import instantiate_from_config
from utils import count_params
from torch.optim.lr_scheduler import LambdaLR

from modules.ema import LitEma
from utils.img_utils import calculate_ssim, calculate_psnr, calculate_p_loss

__conditioning_keys__ = {'concat': 'c_concat',
                         'cross_attn': 'c_cross_attn',
                         'adm': 'y'}

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def sigmoid(x):
    return 1 / (1 + exp(-x))

def interpolation_reduction(img, polarization=False):

    img = torch.nn.functional.interpolate(img,
                                          scale_factor=0.5,
                                          mode='bilinear'
                                          )
    img = torch.nn.functional.interpolate(img,
                                          scale_factor=0.5,
                                          mode='bilinear'
                                          )
    if polarization:
        img[img > -1] = 1

    return img

def get_input(batch, k):
    x = batch[k]
    if len(x.shape) == 3:
        x = x[..., None]
    # x = rearrange(x, 'b h w c -> b c h w')
    x = x.to(memory_format=torch.contiguous_format).float()
    return x

def disabled_train(self, mode=True):
    """Overwrite model.
    train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class DiffusionWrapper(nn.Module):
    def __init__(self,
                 diffusion_config,
                 conditioning_key
                 ):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diffusion_config)
        self.conditioning_key = conditioning_key

    def forward(self, x, t, c_concat: list = None):
        xc = torch.cat([x] + c_concat, dim=1)
        out = self.diffusion_model(xc, t)
        return out


class BlurDiffusion(pl.LightningModule):
    """main class"""
    def __init__(self,
                 unet_config,
                 sigma_blur_max=4.0,
                 img_size=256,
                 use_masks=True,
                 cond_stage_key="image",
                 concat_mode=True,
                 conditioning_key=None,
                 time_steps=1000,
                 loss_type="l1",
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 channels=3,
                 log_every_t=100,
                 parameterization="eps",
                 scheduler_config=None,
                 ):

        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'cross_attn'
        # if cond_stage_config == '__is_unconditional__':
        #     conditioning_key = None

        super().__init__()

        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.channels = channels
        self.sigma_blur_max = sigma_blur_max
        self.img_size = img_size
        self.time_steps = time_steps

        self.concat_mode = concat_mode
        self.cond_stage_key = cond_stage_key
        self.use_masks = use_masks

        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        if monitor is not None:
            self.monitor = monitor

        self.loss_type = loss_type

        # self.instantiate_first_stage(first_stage_config)
        # self.instantiate_cond_stage(cond_stage_config)

        self.val_nums = 0
        self.val_ssim = 0
        self.val_p_loss = 0
        self.val_psnr = 0
        self.val_fid = 0
        # self.instantiate_blur_schedule(sigma_blur_min, sigma_blur_max)

    def DCT(self, x, norm='ortho'):
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N).contiguous()

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        # Vc = torch.fft.rfft(v, 1)
        Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

        k = - torch.arange(N, dtype=x.dtype,
                           device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        if norm == 'ortho':
            V[:, 0] /= math.sqrt(N) * 2
            V[:, 1:] /= math.sqrt(N / 2) * 2

        V = 2 * V.contiguous().view(*x_shape).contiguous()

        return V

    def IDCT(self, X, norm='ortho'):
        """
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
        Our definition of idct is that idct(dct(x)) == x
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the inverse DCT-II of the signal over the last dimension
        """

        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]).contiguous() / 2

        if norm == 'ortho':
            X_v[:, 0] *= math.sqrt(N) * 2
            X_v[:, 1:] *= math.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype,
                         device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        # v = torch.fft.irfft(V, 1)
        v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.contiguous().view(*x_shape).contiguous()

    def get_frequency_scaling(self, t, min_scale=0.001):
        # compute dissipation time
        sigma_t = self.sigma_blur_max * sin(t * pi / 2) ** 2
        dissipation_time = (sigma_t ** 2 / 2)[:, None, None, None]

        # compute frequencies
        freqs = pi * torch.linspace(0, self.img_size - 1, self.img_size) / self.img_size

        labda = freqs[None, None, :, None] ** 2 + freqs[None, :, None, None] ** 2
        # compute scaling for frequencies
        scaling = exp(- labda * dissipation_time) * (1 - min_scale)
        scaling = scaling + min_scale
        return scaling

    def get_noise_schaling_cosine(self, t, logsnr_min=-10, logsnr_max=10):
        limit_max = arctan(np.exp(-0.5 * logsnr_max))
        limit_min = arctan(np.exp(-0.5 * logsnr_min)) - limit_max
        logsnr = -2 * log(tan(limit_min * t + limit_max))
        # Transform logsnr to a, sigma .
        return sqrt(sigmoid(logsnr))[:, None, None, None], sqrt(sigmoid(- logsnr))[:, None, None, None]

    def get_alpha_sigma(self, t):
        freq_scaling = self.get_frequency_scaling(t)
        a, sigma = self.get_noise_schaling_cosine(t)
        alpha = a * freq_scaling  # Combine dissipation and scaling .

        return alpha, sigma

    def diffuse(self, x, t):
        x_freq = self.DCT(x)
        alpha, sigma = self.get_alpha_sigma(t)
        alpha = alpha.to(x_freq.device).to(torch.float32)
        sigma = sigma.to(x_freq.device).to(torch.float32)
        eps = torch.randn_like(x)
        # Since we chose sigma to be a scalar , eps does not need to be
        # passed through a DCT/ IDCT in this case .
        z_t = self.IDCT(alpha * x_freq) + sigma * eps
        return z_t, eps

    def denoise(self, z_t, t, s, c, delta=1e-8):

        alpha_s, sigma_s = self.get_alpha_sigma(s / self.time_steps)
        alpha_t, sigma_t = self.get_alpha_sigma(t / self.time_steps)

        alpha_s = alpha_s.to(self.device).to(torch.float32)
        sigma_s = sigma_s.to(self.device).to(torch.float32)
        alpha_t = alpha_t.to(self.device).to(torch.float32)
        sigma_t = sigma_t.to(self.device).to(torch.float32)

        # Compute helpful coefficients .
        alpha_ts = alpha_t / alpha_s
        alpha_st = 1 / alpha_ts
        sigma2_ts = (sigma_t ** 2 - alpha_ts ** 2 * sigma_s ** 2)
        # Denoising variance .
        sigma2_denoise = 1 / torch.clip(
            1 / torch.clip(sigma_s ** 2, min=delta) +
            1 / torch.clip(sigma_t ** 2 / alpha_ts ** 2 - sigma_s ** 2, min=delta),
            min=delta)
        # The coefficients for u_t and u_eps .
        coeff_term1 = alpha_ts * sigma2_denoise / (sigma2_ts + delta)
        coeff_term2 = alpha_st * sigma2_denoise / torch.clip(sigma_s ** 2, min=delta)
        # Get neural net prediction .
        t = t.to(self.device)
        hat_eps = self.apply_model(z_t, t, c)
        z_t = z_t.permute(0, 2, 3, 1).contiguous()
        hat_eps = hat_eps.permute(0, 2, 3, 1).contiguous()
        # Compute terms .
        u_t = self.DCT(z_t)
        term1 = self.IDCT(coeff_term1 * u_t)
        term2 = self.IDCT(coeff_term2 * (u_t - sigma_t * self.DCT(hat_eps)))
        mu_denoise = term1 + term2
        # Sample from the denoising distribution .
        eps = torch.randn_like(mu_denoise)
        result = mu_denoise + self.IDCT(torch.sqrt(sigma2_denoise) * eps)
        return result.permute(0, 3, 1, 2).contiguous()


    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=False,
                      logger=True, on_step=True, on_epoch=True, sync_dist=True)

        self.log("step", float(self.global_step + 1),
                 prog_bar=True, logger=False, on_step=True, on_epoch=False, sync_dist=True)

        self.log('total_loss', loss_dict['train/loss'],
                 prog_bar=True, logger=False, on_step=True, on_epoch=False, sync_dist=True)

        self.log('loss_m', loss_dict['train/loss_m'],
                 prog_bar=True, logger=False, on_step=True, on_epoch=False, sync_dist=True)

        self.log('loss_u', loss_dict['train/loss_u'],
                 prog_bar=True, logger=False, on_step=True, on_epoch=False, sync_dist=True)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=False, on_step=True, on_epoch=False, sync_dist=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, c, mask = self.get_input(batch, self.first_stage_key)
        ssim, lpips, psnr = self.val_shared_step(x, c, mask)

        self.val_ssim += ssim
        self.val_p_loss += lpips
        self.val_psnr += psnr

    @torch.no_grad()
    def val_shared_step(self, gt, cond, mask):
        ssim = 0.0
        lpips = 0.0
        psnr = 0.0
        cond_list = torch.chunk(cond, cond.shape[0], dim=0)
        mask_list = torch.chunk(mask, mask.shape[0], dim=0)
        gt_list = torch.chunk(gt, gt.shape[0], dim=0)
        for gt, cond, mask in zip(gt_list, cond_list, mask_list):

            mask_reduce_t = torch.clamp((mask + 1.0) / 2.0,
                                        min=0.0, max=1.0)
            reverse_cond = torch.concat([cond, mask], dim=1)


            b = cond.shape[0]
            shape = cond.shape[1:]

            predict = self.p_sample_loop(step=20,
                                         cond=reverse_cond,
                                         batch_size=b,
                                         shape=shape,
                                         mask=mask_reduce_t,
                                         x0=cond,
                                         )


            masked_image = torch.clamp((cond + 1.0) / 2.0,
                                       min=0.0, max=1.0)
            mask = torch.clamp((mask + 1.0) / 2.0,
                               min=0.0, max=1.0)
            gt = torch.clamp((gt + 1.0) / 2.0,
                               min=0.0, max=1.0) * 255.0

            predict = torch.clamp((predict+1.0)/2.0, min=0.0, max=1.0)

            inpainted = (1-mask)*masked_image+mask*predict
            output = (inpainted[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            gt = gt[0].cpu().numpy().transpose(1, 2, 0)
            ssim += calculate_ssim(output, gt)
            psnr += calculate_psnr(output, gt)
            lpips += calculate_p_loss(output, gt)

        nums = len(gt_list)

        self.val_nums += nums

        return ssim, lpips, psnr

    @torch.no_grad()
    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:

        if self.val_nums == 0:
            raise ValueError("nums is 0")
        else:
            self.val_ssim /= self.val_nums
            self.val_p_loss /= self.val_nums
            self.val_psnr /= self.val_nums

        self.log("val_ssim", self.val_ssim,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_lpips", self.val_p_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_psnr", self.val_psnr,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        self.val_ssim = self.val_p_loss = self.val_psnr = self.val_nums = 0

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @torch.no_grad()
    def get_input(self, batch, k):
        x = get_input(batch, k)
        x = x.to(self.device)
        z = x.detach()
        xc = get_input(batch, self.cond_stage_key).to(self.device)
        c = xc.to(self.device)

        out = [z, c]
        if self.use_masks:
            mask = batch["mask"]
            out = [z, c, mask]

        return out

    def shared_step(self, batch, **kwargs):
        x, c, mask = self.get_input(batch, self.first_stage_key)
        loss = self(x, c, mask)
        return loss

    def forward(self, x, c, mask, *args, **kwargs):
        t = torch.randint(1, self.time_steps+1, (x.shape[0],)).long()
        return self.p_losses(x, c, t, mask, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_cross_attn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)

        return x_recon

    def p_losses(self, x_start, cond, t, mask, noise=None):

        mask_cond = torch.clamp((mask + 1.0) / 2.0,
                                    min=0.0, max=1.0)

        concat_cond = torch.cat([cond, mask], dim=1)
        scale_t = t

        x_res = x_start.permute(0, 2, 3, 1).contiguous()

        x_noisy, eps = self.diffuse(x_res, scale_t / self.time_steps)
        x_noisy = x_noisy.permute(0, 3, 1, 2).contiguous()
        eps = eps.permute(0, 3, 1, 2).contiguous()
        scale_t = scale_t.to(self.device)


        model_output = self.apply_model(x_noisy, scale_t, concat_cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = eps
        else:
            raise NotImplementedError()


        loss_m, loss_u = self.get_loss(model_output, target, mask_cond, mean=False)
        loss_m = loss_m.mean()
        loss_u = loss_u.mean() * 10
        loss = loss_m + loss_u

        loss_dict.update({f'{prefix}/loss_m': loss_m})
        loss_dict.update({f'{prefix}/loss_u': loss_u})
        loss_dict.update({f'{prefix}/loss': loss})
        return loss, loss_dict

    def get_loss(self, pred, target, mask=None, mean=True):
        if self.loss_type == 'l1':
            loss_fn = F.l1_loss
        elif self.loss_type == 'l2':
            loss_fn = F.mse_loss
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        if mask is not None:
            loss_m = loss_fn(pred * (1-mask), target * (1-mask))
            loss_u = loss_fn(pred * mask, target * mask)
        else:
            loss_m = loss_fn(pred, target)
            loss_u = 0
        return loss_m, loss_u

        return loss

    @torch.no_grad()
    def p_sample_loop(self, step, cond, batch_size, shape,
                      x_T=None, verbose=True, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, start_T=None, reverse_noise=None):
        c = self.time_steps // step
        sample_timesteps = np.asarray(list(range(1, self.time_steps+1, c)))
        # sample_timesteps = sample_timesteps + 1
        prev_sample_timesteps = np.insert(sample_timesteps, 0, 0)
        prev_sample_timesteps = np.delete(prev_sample_timesteps, -1)

        device = cond.device
        b = batch_size
        shape = (b,) + shape
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        if timesteps is None:
            timesteps = self.time_steps

        if start_T is not None:
            timesteps = min(timesteps, start_T)

        iterator = np.flip(sample_timesteps)
        prev_iterator = np.flip(prev_sample_timesteps)


        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        tran_x0 = x0.permute(0, 2, 3, 1).contiguous()

        for t, s in zip(iterator, prev_iterator):
            time = torch.full((b,), t, dtype=torch.long)
            prev_s = torch.full((b,), s, dtype=torch.long)

            # using unmasked regions to guide reverse process
            if mask is not None:
                img_orig, _ = self.diffuse(tran_x0, time/self.time_steps)
                img_orig = img_orig.permute(0, 3, 1, 2)
                img = img_orig * (1-mask) + mask * img

            img = self.denoise(img, time, prev_s, cond)

        return img

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())

        opt = torch.optim.Adam(params=params, lr=lr, betas=(0.9, 0.99))
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule, last_epoch=-1),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

