import itertools
import pdb
import random
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
# from model.VQGAN.vqgan import VQModel
from model.VQGAN.vqvae import VQVAE


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        # self.vqgan = VQModel(**vars(model_config.VQGAN.params)).eval()
        self.vqgan = VQVAE(spatial_dims=2,
                           in_channels=1,
                           out_channels=1,
                           num_channels=(256, 512),
                           num_res_channels=512,
                           num_res_layers=2,
                           downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
                           upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
                           num_embeddings=model_config.VQGAN.params.n_embed,
                           embedding_dim=model_config.VQGAN.params.embed_dim,
                           )
        # Define the path to your checkpoint
        checkpoint_path = model_config.VQGAN.params.ckpt_path
        checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        self.vqgan.load_state_dict(checkpoint["state_dict"])
        self.vqgan.eval()
        # self.quant_conv = torch.nn.Conv2d(32, 32, 1)
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"load vqgan from {model_config.VQGAN.params.ckpt_path}")

        # Condition Stage Model
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
        else:
            raise NotImplementedError

    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.condition_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    def forward(self, x, x_cond, context=None):
        with torch.no_grad():
            x_latent = self.encode(x, cond=False)
            x_cond_latent = self.encode(x_cond, cond=True)
        # context = self.get_cond_stage_context(x_cond)
        context = x_cond_latent
        # print(f'x_latent range of pixels: {torch.min(x_latent)},{torch.max(x_latent)}')
        # print(f'x_cond_latent range of pixels: {torch.min(x_cond_latent)},{torch.max(x_cond_latent)}')
        return super().forward(x_latent.detach(), x_cond_latent.detach(), context)

    # def get_cond_stage_context(self, x_cond):
    #    if self.cond_stage_model is not None:
    #        context, _ = self.cond_stage_model(x_cond)
    #        if self.condition_key == 'first_stage':
    #            context = context.detach()
    #    else:
    #        context = None
    #    return context

    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            if self.condition_key == "first_stage":
                context = self.cond_stage_model.encode(x_cond)
                context = context.detach()
            elif self.condition_key == "SpatialRescaler":
                context = self.cond_stage_model(x_cond)
        else:
            context = None
        return context

    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        model = self.vqgan
        x_latent = model.encode(x)  # the output here is a tensor with z_channels channels, so the latent space has dimension (h, w, z_channels), where h and w depends on how many downsampling blocks we choose
        # if not self.model_config.latent_before_quant_conv:
        #    x_latent = model.quant_conv(x_latent)  # here we apply a convolutional layer with input channels z_channels and output channels embed_dim
        if normalize:
            if cond:
                x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
            else:
                x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
        return x_latent  # this is the latent representation of our original image, so we haven't already applied the vector quantization
        # in other words, we apply the BBDM on the latent representation and not on the vector quantization

    @torch.no_grad()
    def decode(self, x_latent, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        if normalize:
            if cond:
                x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
            else:
                x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
        model = self.vqgan
        if self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        # x_latent_quant, _, _ = model.quantize(x_latent)  # after the BBDM process, we quantize the latent representation and then decode it back to the pixel space through the decoder of the VQGAN
        x_latent_quant, _ = model.quantize(x_latent)  # after the BBDM process, we quantize the latent representation and then decode it back to the pixel space through the decoder of the VQGAN
        # print(f'x_latent_quant shape after quantization: {x_latent_quant.shape}')
        # print(f'x_latent_quant pixel range after quantization: {torch.min(x_latent_quant), torch.max(x_latent_quant)}')
        out = model.decode(x_latent_quant)
        return out

    @torch.no_grad()
    def sample(self, batch, clip_denoised=False, sample_mid_step=False, device=None):
        # print(f'Sampling: x_cond shape before encoding: {x_cond.shape}')
        # print(f'Sampling: x_cond pixel range before encoding: {torch.min(x_cond)}, {torch.max(x_cond)}')
        x, x_cond = batch
        batch_size = x.shape[0] if x.shape[0] < 4 else 4

        x = x[0:batch_size].to(device)
        x_cond = x_cond[0:batch_size].to(device)

        x_cond_latent = self.encode(x_cond, cond=True)  # x_cond: ART10 --> x_cond_latent: latent encoding of ART10, which is the starting point of the diffusion process/brownian bridge
        # print(f'Sampling: x_cond_latent shape after encoding: {x_cond_latent.shape}')
        # print(f'Sampling: x_cond_latent pixel range after encoding: {torch.min(x_cond_latent), torch.max(x_cond_latent)}')
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=x_cond_latent,  # the latent encoding of ART10 is our "y"
                                                     context=self.get_cond_stage_context(x_cond),
                                                     clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step)
            # p_sample_loop performs the entire brownian-bridge process, returning the latent reconstruction of x, so we go from the latent representation of x_cond (ART10) to the latent representation of x (pseudoART100)
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(temp[i].detach(), cond=False)
                out_samples.append(out.to('cpu'))

            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps",
                          dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(out.to('cpu'))
            sample_vqgan_art10 = self.sample_vqgan(x_cond)  # this is just the reconstruction of the input ART10 (x_cond) given by the VQGAN model without any diffusion process
            sample_vqgan_pseudoart100 = self.sample_vqgan(x)  # this is just the reconstruction of the target pseudoART100 (x) given by the VQGAN model without any diffusion process
            return out_samples, one_step_samples, sample_vqgan_art10, sample_vqgan_pseudoart100
        else:
            # p_sample_loop performs the entire brownian-bridge process, returning the latent reconstruction of x, so we go from the latent representation of x_cond (ART10) to the latent representation of x (pseudoART100)
            temp = self.p_sample_loop(y=x_cond_latent,
                                      context=self.get_cond_stage_context(x_cond),
                                      clip_denoised=clip_denoised,
                                      sample_mid_step=sample_mid_step)
            x_latent = temp
            # print(f'Sampling: x_latent shape after p_sample_loop (before decoding): {x_latent.shape}')
            # print(f'Sampling: x_latent pixel range after p_sample_loop (before decoding): {torch.min(x_latent)}, {torch.max(x_latent)}')
            out = self.decode(x_latent, cond=False)
            # print(f'Sampling: out shape after decoding (our final output): {out.shape}')
            # print(f'Sampling: out pixel range after decoding (our final output): {torch.min(out), torch.max(out)}')
            sample_vqgan_art10 = self.sample_vqgan(x_cond)  # this is just the reconstruction of the input ART10 (x_cond) given by the VQGAN model without any diffusion process
            sample_vqgan_pseudoart100 = self.sample_vqgan(x)  # this is just the reconstruction of the target pseudoART100 (x) given by the VQGAN model without any diffusion process
            return out, sample_vqgan_art10, sample_vqgan_pseudoart100

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqgan(x)
        return x_rec

    # @torch.no_grad()
    # def reverse_sample(self, x, skip=False):
    #     x_ori_latent = self.vqgan.encoder(x)
    #     temp, _ = self.brownianbridge.reverse_p_sample_loop(x_ori_latent, x, skip=skip, clip_denoised=False)
    #     x_latent = temp[-1]
    #     x_latent = self.vqgan.quant_conv(x_latent)
    #     x_latent_quant, _, _ = self.vqgan.quantize(x_latent)
    #     out = self.vqgan.decode(x_latent_quant)
    #     return out
