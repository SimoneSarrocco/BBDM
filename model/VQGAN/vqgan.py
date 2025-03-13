import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pdb
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from model.VQGAN.model import Encoder, Decoder
from model.VQGAN.quantize import VectorQuantizer2 as VectorQuantizer
from model.VQGAN.quantize import GumbelQuantize
import importlib
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import numpy as np
from dataset import OCTDataset
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import loggers as pl_loggers


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    # pdb.set_trace()
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    if config.__contains__('params'):
        # return get_obj_from_str(config["target"])(**vars(config['params']))
        return get_obj_from_str(config["target"])(**config['params'])
    else:
        return get_obj_from_str(config["target"])()


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 lr_ae,
                 lr_disc,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**vars(ddconfig))
        self.decoder = Decoder(**vars(ddconfig))
        # self.loss = instantiate_from_config(vars(lossconfig))
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig.z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig.z_channels, 1)
        self.learning_rate_autoencoder = lr_ae
        self.learning_rate_discriminator = lr_disc
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.psnr_metric = PeakSignalNoiseRatio().to(self.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure().to(self.device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(self.device)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        # sd = torch.load(path, map_location="cpu")
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch):
        x = batch
        if len(x.shape) == 3:
            x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        x = x.to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx):
        self.automatic_optimization = False  # Enable manual optimization
        opt_ae, opt_disc = self.configure_optimizers()
        x = self.get_input(batch)
        xrec, qloss = self(x)

        # self.toggle_optimizer(opt_ae)
        opt_ae.zero_grad()
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")

        # self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        aeloss.backward()
        opt_ae.step()
        # self.untoggle_optimizer(opt_ae)
        # return aeloss

        # self.toggle_optimizer(opt_disc)
        opt_disc.zero_grad()
        # discriminator
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        # self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        # self.manual_backward(discloss)
        discloss.backward()
        opt_disc.step()
        # self.untoggle_optimizer(opt_disc)
        # return discloss

        # ------------------
        # Autoencoder Step
        # ------------------
        # self.toggle_optimizer(opt_ae)

        """
        aeloss, log_dict_ae = self.loss(
            qloss, x, xrec, 0, self.global_step,
            last_layer=self.get_last_layer(), split="train"
        )
        """
        # self.log_dict(log_dict_ae, prog_bar=False, on_step=True, on_epoch=True)

        # opt_ae.zero_grad()
        # self.manual_backward(aeloss)
        # opt_ae.step()

        # self.untoggle_optimizer(opt_ae)

        # if self.trainer is not None and hasattr(self.trainer, 'training') and self.trainer.training:
        #    self.log("train/aeloss", aeloss, prog_bar=True, on_step=True, on_epoch=True)

        # ------------------
        # Discriminator Step
        # ------------------
        # self.toggle_optimizer(opt_disc)

        """
        discloss, log_dict_disc = self.loss(
            qloss, x, xrec, 1, self.global_step,
            last_layer=self.get_last_layer(), split="train"
        )
        """
        # self.log_dict(log_dict_disc, prog_bar=False, on_step=True, on_epoch=True)

        # opt_disc.zero_grad()
        # self.manual_backward(discloss)
        # opt_disc.step()

        # self.untoggle_optimizer(opt_disc)

        # if self.trainer is not None and hasattr(self.trainer, 'training') and self.trainer.training:
        #    self.log("train/discloss", discloss, prog_bar=True, on_step=True, on_epoch=True)

        # return {"train/aeloss": aeloss, "train/discloss": discloss}
        # Return results instead of logging them
        return {
            "x": x,
            "xrec": xrec,
            "qloss": qloss,
            "aeloss": aeloss,
            "discloss": discloss,
            "log_dict_ae": log_dict_ae,
            "log_dict_disc": log_dict_disc
        }

    def validation_step(self, batch, batch_idx):
        # print(f"Validation Step {batch_idx} Started")  # Debugging print
        # self.trainer.testing = True
        # self.trainer.validating = True
        # if not self.trainer or not self.trainer.training:
        #    return

        x = self.get_input(batch)
        xrec, qloss = self(x)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        # print(f"Validation Step {batch_idx} Completed Forward Pass")  # Debugging print

        # Debugging: Check if loss dictionaries are empty
        # print(f"log_dict_ae: {log_dict_ae}")
        # print(f"log_dict_disc: {log_dict_disc}")
        # rec_loss = log_dict_ae.get("val/rec_loss", torch.tensor(0.0, device=self.device))

        # --- Fix: Ensure Trainer is running before calling `self.log()` ---
        # if self.trainer is None or self.trainer.training is False:
        #     print("⚠️ WARNING: Trainer is not initialized properly or not in training mode!")
        #    return rec_loss  # Skip logging and return loss safely

        # if self.trainer is not None:
            # Log only if `Trainer` is running validation
        #    self.log("val/rec_loss", rec_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        #    self.log("val/aeloss", aeloss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        #    self.log("val/discloss", discloss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # Ensure dictionaries exist before logging
        # if log_dict_ae:
        #    self.log_dict(log_dict_ae, prog_bar=False, on_step=False, on_epoch=True)
        # if log_dict_disc:
        #    self.log_dict(log_dict_disc, prog_bar=False, on_step=False, on_epoch=True)

        # --- Fix: Ensure the metrics are moved to the correct device ---
        # x, xrec = x.to(self.device), xrec.to(self.device)

        # Compute evaluation metrics
        psnr = self.psnr_metric(xrec, x)
        ssim = self.ssim_metric(xrec, x)
        mse = mean_flat((xrec - x) ** 2).mean()

        # Convert grayscale to 3 channels for LPIPS
        # xrec_3_channels = xrec.repeat(1, 3, 1, 1)  # (Batch, Channels, Height, Width)
        # x_3_channels = x.repeat(1, 3, 1, 1)  # (Batch, Channels, Height, Width)

        # Ensure LPIPS is correctly computed
        # lpips_val = self.lpips_metric(xrec_3_channels, x_3_channels)

        # if self.trainer is not None:
            # Log computed metrics
        #    self.log("val/psnr", psnr, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        #    self.log("val/ssim", ssim, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        #    self.log("val/mse", mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        #    self.log("val/lpips", lpips_val, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # return self.log_dict
        # return {"val/rec_loss", rec_loss}
        return {
            "x": x,
            "xrec": xrec,
            "val/rec_loss": log_dict_ae.get("val/rec_loss", torch.tensor(0.0, device=self.device)),
            "val/aeloss": aeloss,
            "val/discloss": discloss,
            "val/psnr": psnr,
            "val/ssim": ssim,
            "val/mse": mse,
            # "val/lpips": lpips_val,
            "log_dict_ae": log_dict_ae,
            "log_dict_disc": log_dict_disc
        }

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate_autoencoder, betas=(0.9, 0.999))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=self.learning_rate_discriminator, betas=(0.9, 0.999))
        return [opt_ae, opt_disc]

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


class GumbelVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)   # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.temperature_scheduling()
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


# --- Lightning Module for Training ---
# We wrap the VQModel in a LightningModule for training convenience.
class LitVQGAN(pl.LightningModule):
    def __init__(self, ddconfig, lossconfig, n_embed, embed_dim, lr_ae=1e-4, lr_disc=5e-4, ckpt_path=None, ignore_keys=[]):
        super().__init__()
        # self.learning_rate = lr
        self.model = VQModel(ddconfig=ddconfig,
                             lossconfig=lossconfig,
                             n_embed=n_embed,
                             embed_dim=embed_dim,
                             lr_ae=lr_ae,
                             lr_disc=lr_disc,
                             ckpt_path=ckpt_path,
                             ignore_keys=ignore_keys,
                             )
        self.automatic_optimization = False  # Enable manual optimization

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Get optimizers
        # opt_ae, opt_disc = self.optimizers()
        tensorboard = self.logger.experiment
        # Get model outputs without optimization
        outputs = self.model.training_step(batch, batch_idx)
        # self.model.training_step(batch, batch_idx, optimizer_idx)

        # Extract values
        aeloss = outputs["aeloss"]
        discloss = outputs["discloss"]
        log_dict_ae = outputs["log_dict_ae"]
        log_dict_disc = outputs["log_dict_disc"]
        input_image = outputs["x"]
        reconstruction = outputs["xrec"]
        print(f'Input shape: {input_image.shape}')
        print(f'Reconstruction shape: {reconstruction.shape}')
        print(f'Input range: {torch.min(input_image)},{torch.max(input_image)}')
        print(f'Reconstruction range: {torch.min(reconstruction)},{torch.max(reconstruction)}')
        reconstruction = (reconstruction - torch.min(reconstruction)) / (
                    torch.max(reconstruction) - torch.min(reconstruction))
        if batch_idx % 10 == 0:
            # self.writer.add_image(f'Training/Input_{batch_idx}', x.squeeze(0), self.global_step)
            tensorboard.add_image(f'Training/Input_{batch_idx}', input_image.squeeze(0), self.global_step)
            tensorboard.add_image(f'Training/Reconstruction_{batch_idx}', reconstruction.squeeze(0).clamp(0., 1.), self.global_step)
            # self.writer.add_image(f'Training/Reconstruction_{batch_idx}', xrec.squeeze(0), self.global_step)
        # ------------------
        # Autoencoder Step
        # ------------------
        """
        self.toggle_optimizer(opt_ae)

        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        self.untoggle_optimizer(opt_ae)

        # ------------------
        # Discriminator Step
        # ------------------
        self.toggle_optimizer(opt_disc)

        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

        self.untoggle_optimizer(opt_disc)
        """
        # Logging (now handled by LitVQGAN)
        self.log("train/aeloss", aeloss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/discloss", discloss, prog_bar=True, on_step=True, on_epoch=True)

        # Log dictionaries if needed
        self.log_dict(log_dict_ae, prog_bar=False, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, on_step=True, on_epoch=True)

        """
        # Get input and run forward pass
        x = self.model.get_input(batch)
        xrec, qloss = self.model(x)

        if optimizer_idx == 0:
            # Autoencoder step
            aeloss, log_dict_ae = self.model.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                  last_layer=self.model.get_last_layer(), split="train")
            self.log("train/aeloss", aeloss, prog_bar=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, on_step=True, on_epoch=True)
            return aeloss

        elif optimizer_idx == 1:
            # Discriminator step
            discloss, log_dict_disc = self.model.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                      last_layer=self.model.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, on_step=True, on_epoch=True)
            return discloss
        """
        # return self.model.training_step(batch, batch_idx)
        return {"train/aeloss": aeloss, "train/discloss": discloss}

    def validation_step(self, batch, batch_idx):
        # Get model outputs
        outputs = self.model.validation_step(batch, batch_idx)

        # Log metrics
        # self.log("val/rec_loss", outputs["val/rec_loss"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", outputs["val/aeloss"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/discloss", outputs["val/discloss"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/psnr", outputs["val/psnr"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/ssim", outputs["val/ssim"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mse", outputs["val/mse"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("val/lpips", outputs["val/lpips"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # Create modified dictionaries to avoid duplicate logging
        log_dict_ae = outputs["log_dict_ae"].copy() if outputs["log_dict_ae"] else {}
        log_dict_disc = outputs["log_dict_disc"].copy() if outputs["log_dict_disc"] else {}

        # Remove keys that we've already logged directly
        for key in ["val/rec_loss"]:
            if key in log_dict_ae:
                del log_dict_ae[key]

        # Log the cleaned dictionaries if they still have entries
        if log_dict_ae:
            self.log_dict(log_dict_ae, prog_bar=False, on_step=False, on_epoch=True)
        if log_dict_disc:
            self.log_dict(log_dict_disc, prog_bar=False, on_step=False, on_epoch=True)

        # Log rec_loss separately to avoid conflicts
        self.log("val/rec_loss", outputs["val/rec_loss"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        tensorboard = self.logger.experiment
        input_image = outputs["x"]
        reconstruction = outputs["xrec"]
        print(f'Input val shape: {input_image.shape}')
        print(f'Reconstruction val shape: {reconstruction.shape}')
        print(f'Input val range: {torch.min(input_image)},{torch.max(input_image)}')
        print(f'Reconstruction val range: {torch.min(reconstruction)},{torch.max(reconstruction)}')
        # reconstruction = (reconstruction - torch.min(reconstruction)) / (torch.max(reconstruction) - torch.min(reconstruction))
        if batch_idx % 10 == 0:
            # self.writer.add_image(f'Training/Input_{batch_idx}', x.squeeze(0), self.global_step)
            tensorboard.add_image(f'Validation/Input_{batch_idx}', input_image.squeeze(0).clamp(0., 1.), self.global_step)
            tensorboard.add_image(f'Validation/Reconstruction_{batch_idx}', reconstruction.squeeze(0).clamp(0., 1.), self.global_step)

        return outputs["val/rec_loss"]  # Return the primary validation metric
        # return self.model.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return self.model.configure_optimizers()


# --- Dummy Configuration Classes ---
# These dummy classes mimic the YAML configuration used in BBDM.
class DummyDDConfig:
    def __init__(self):
        self.double_z = False
        self.z_channels = 8      # Make sure this matches your encoder output
        self.resolution = 512
        self.in_channels = 1
        self.out_ch = 1
        self.ch = 128
        self.ch_mult = (1, 2, 4, 8)
        self.num_res_blocks = 2
        self.attn_resolutions = [16]
        self.dropout = 0.0


if __name__ == "__main__":
    ddconfig = DummyDDConfig()

    train = np.load('/home/simone.sarrocco/thesis/project/data/train_set_patient_split.npz')['images']
    val = np.load('/home/simone.sarrocco/thesis/project/data/val_set_patient_split.npz')['images']
    test = np.load('/home/simone.sarrocco/thesis/project/data/test_set_patient_split.npz')['images']

    """
    train_pseudoart100_images = []
    val_pseudoart100_images = []
    test_pseudoart100_images = []

    for i in range(len(train)):
        image = torch.tensor(train[i, -1:, ...])
        train_pseudoart100_images.append(image)
    train_pseudoart100_images = torch.stack(train_pseudoart100_images, 0)

    for i in range(len(val)):
        image = torch.tensor(val[i, -1:, ...])
        val_pseudoart100_images.append(image)
    val_pseudoart100_images = torch.stack(val_pseudoart100_images, 0)

    for i in range(len(test)):
        image = torch.tensor(test[i, -1:, ...])
        test_pseudoart100_images.append(image)
    test_pseudoart100_images = torch.stack(test_pseudoart100_images, 0)

    train_data = OCTDataset(train_pseudoart100_images, transform=True)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)
    print(f'Shape of training set: {train_pseudoart100_images.shape}')

    val_data = OCTDataset(val_pseudoart100_images)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
    print(f'Shape of validation set: {val_pseudoart100_images.shape}')
    """
    train_data_split = torch.tensor(train).view((-1, 1, 496, 768))  # Pass shape as a tuple
    val_data_split = torch.tensor(val).view((-1, 1, 496, 768))  # Pass shape as a tuple
    test_data_split = torch.tensor(test).view((-1, 1, 496, 768))  # Pass shape as a tuple

    train_data = OCTDataset(train_data_split, transform=True)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)
    print(f'Shape of training set: {train_data_split.shape}')

    val_data = OCTDataset(val_data_split)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
    print(f'Shape of validation set: {val_data_split.shape}')

    lossconfig = {
        "target": "vqperceptual.VQLPIPSWithDiscriminator",  # Adjust module path as needed.
        "params": {
            "disc_start": 0,  # Step at which discriminator loss begins to be applied.
            "codebook_weight": 1.0,
            "pixelloss_weight": 1.0,
            "disc_num_layers": 3,
            "disc_in_channels": 1,  # For grayscale OCT images; if RGB, use 3.
            "disc_factor": 1.0,
            "disc_weight": 1.0,
            "perceptual_weight": 0.001,
            "use_actnorm": False,
            "disc_conditional": False,
            "disc_ndf": 64,
            "disc_loss": "vanilla"
        }
    }

    # writer = SummaryWriter(log_dir='/home/simone.sarrocco/thesis/project/models/diffusion_model/BBDM/lightning_logs')
    # lossconfig_ns = SimpleNamespace(**lossconfig)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs/')

    # --- Instantiate the Lightning Module ---
    lit_model = LitVQGAN(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        n_embed=16384,
        embed_dim=8,
        lr_ae=1e-4,
        lr_disc=5e-4,
        ckpt_path=None,
    )

    # --- Set Up Checkpointing ---
    checkpoint_callback = ModelCheckpoint(
        monitor="val/rec_loss",
        dirpath="checkpoints",
        filename="vqgan-{epoch:02d}-{val_rec_loss:.2f}",
        save_top_k=1,
        mode="min",
        every_n_epochs=1  # Save every 10 epochs
    )

    # --- Initialize Trainer and Start Training ---
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=100,
        # callbacks=[TQDMProgressBar(), checkpoint_callback],
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        # progress_bar_refresh_rate=20,
        # strategy='ddp_find_unused_parameters_true',  # multi-optimizer support
    )
    print("🔥 Trainer is starting training!")
    trainer.fit(lit_model, train_loader, val_loader)
