# conda activate opensora

import os
import sys
import json
import time

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
import torch.distributed as dist

from clip import ClipEncoder

from diffusers.models import AutoencoderKL
from einops import rearrange

import torchvision.transforms as transforms

# sys.path.append(".")  # noqa
sys.path.append("./magicdrive/dataset/")  # noqa
from fpv_nuscenes_dataset import DatasetFromCSV

from video_transforms import ToTensorVideo


class Config:

    def __init__(self) -> None:
        self.debug = False

        self.override_preprocessed = False
        self.preprocess_batch_size = 1
        self.preprocessed_dir = "encoder_out"
        self.text_key = "text"  # "text", "short_text", "category"

        self.use_preprocessed_data = True

        self.num_frames = 16
        self.frame_interval = 2
        self.image_size = (224, 400)

        self.patch_size = (1, 2, 2)

        # for classifier-free guidance
        self.token_drop_prob = 0.1

        self.use_ema = True

        # pretrained vae
        # self.vae_pretrained = "stabilityai/sd-vae-ft-ema"
        # self.subfolder = ""
        # self.vae_pretrained = "madebyollin/sdxl-vae-fp16-fix"
        # self.subfolder = ""
        self.vae_pretrained = "runwayml/stable-diffusion-v1-5"
        self.subfolder = "vae"

        # text encoder
        # self.textenc_pretrained = "runwayml/stable-diffusion-v1-5"
        self.textenc_pretrained = "/home/ubuntu/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9"
        self.model_max_length = 77
        self.text_encoder_output_dim = 768

        # Define dataset
        self.root = "/home/ubuntu/Documents/ml_data/nuscenes_data"
        self.data_path = "data.csv"

        self.use_image_transform = False
        self.num_workers = 4

        # Define acceleration
        self.dtype = "fp32"
        self.grad_checkpoint = True
        self.plugin = "zero2"
        self.sp_size = 1

        # Others
        self.seed = 123
        self.outputs = "outputs"
        self.wandb = False


class VideoAutoEncoderKL(nn.Module):

    def __init__(self,
                 pretrained_model,
                 subfolder="",
                 dtype=torch.float16,
                 micro_batch_size=None,
                 patch_size=(1, 8, 8),
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.image_vae = AutoencoderKL.from_pretrained(pretrained_model,
                                                       subfolder=subfolder,
                                                       torch_dtype=dtype)
        self.dtype = dtype
        self.scaling_factor = self.image_vae.config.scaling_factor
        self.out_channels = self.image_vae.config.latent_channels
        self.micro_batch_size = micro_batch_size
        self.patch_size = patch_size  # down factor f = 2^3 = 8

    def encode(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        b = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")

        if self.micro_batch_size is None:
            x = self.image_vae.encode(x).latent_dist.sample().mul_(
                self.scaling_factor)
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_mb = x[i:i + bs]
                x_mb = self.image_vae.encode(x_mb).latent_dist.sample().mul_(
                    self.scaling_factor)
                x_out.append(x_mb)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=b)

        return x

    def decode(self, x):
        b = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        if self.micro_batch_size is None:
            x = self.image_vae.decode(x / self.scaling_factor).sample
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_mb = x[i:i + bs]
                x_mb = self.image_vae.decode(x_mb / self.scaling_factor).sample
                x_out.append(x_mb)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=b)

        return x

    def get_latent_size(self, input_size):
        for i in range(3):
            assert input_size[i] % self.patch_size[
                i] == 0, "input size must be divisible by patch size"
        latent_size = [input_size[i] // self.patch_size[i] for i in range(3)]
        return latent_size


def get_transforms_video():
    video_trans = transforms.Compose([
        ToTensorVideo(),
        # RandomHorizontalFlipVideo(),
        # UCFCenterCropVideo(resolution),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5],
                             inplace=True),
    ])

    return video_trans


def to_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        dtype_mapping = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype not in dtype_mapping:
            raise ValueError
        dtype = dtype_mapping[dtype]
        return dtype
    else:
        raise ValueError


def main():
    # create configs
    cfg = Config()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.manual_seed(cfg.seed)
    torch.cuda.set_device(device)
    dtype = to_torch_dtype(cfg.dtype)
    torch.set_default_dtype(dtype)

    # Setup an experiment folder:
    save_dir = os.path.join(cfg.root, cfg.preprocessed_dir)
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        # write config to json
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            f.write(json.dumps(cfg.__dict__, indent=2, sort_keys=False))

    # prepare dataset
    dataset = DatasetFromCSV(
        cfg.data_path,
        num_frames=cfg.num_frames,
        frame_interval=cfg.frame_interval,
        transform=get_transforms_video(),
        img_size=cfg.image_size,
        root=cfg.root,
    )

    dataloader = DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        sampler=SequentialSampler(dataset),
        batch_size=cfg.preprocess_batch_size,
        drop_last=False,
    )

    print(f"Dataset contains {len(dataset):,} videos ({cfg.data_path})")
    total_batch_size = cfg.preprocess_batch_size * dist.get_world_size(
    ) // cfg.sp_size
    print(f"Total batch size: {total_batch_size}")

    # video VAE
    vae = VideoAutoEncoderKL(cfg.vae_pretrained, cfg.subfolder, dtype=dtype)

    # text encoder
    text_encoder = ClipEncoder(from_pretrained=cfg.textenc_pretrained,
                               model_max_length=cfg.model_max_length,
                               dtype=dtype)

    # 4.3. move to device
    vae = vae.to(device)
    vae.eval()

    num_steps_per_epoch = len(dataloader)

    # =======================================================
    # 6. encoder loop
    # =======================================================
    start_epoch = start_step = log_step = sampler_start_idx = 0

    dataloader_iter = iter(dataloader)
    epoch = 0
    with tqdm(
            range(start_step, num_steps_per_epoch),
            desc=f"Epoch {epoch}",
            # disable=not coordinator.is_master(),
            total=num_steps_per_epoch,
            initial=start_step,
    ) as pbar:

        for step in pbar:
            global_step = epoch * num_steps_per_epoch + step

            # step
            batch = next(dataloader_iter)

            x = batch["rgb"].to(device, dtype)  # [B, C, T, N, H, W]
            depth = batch["depth"].to(dtype)
            semantic_map = batch["semantic_map"].to(dtype)
            y = batch[cfg.text_key]
            video_ids = batch["video_id"]

            # video and text encoding
            with torch.no_grad():
                # here we only encode rgb data with VAE
                x = rearrange(x, "B C T N H W -> B C (T N) H W")
                x = vae.encode(x)
                x = rearrange(x,
                              "B C (T N) H W -> B C T N H W",
                              N=dataset.n_cams)
                model_args = text_encoder.encode(y)

                # if encode only, we save results to file
                for idx in range(len(video_ids)):
                    vid = video_ids[idx]
                    save_fpath = os.path.join(save_dir, vid + ".pt")
                    if not os.path.exists(
                            save_fpath) or cfg.override_preprocessed:
                        saved_data = {
                            "x": x[idx].cpu(),
                            "depth": depth[idx].cpu(),
                            "semantic_map": semantic_map[idx].cpu(),
                            "y": model_args["y"][idx].cpu(),
                            "mask": model_args["mask"][idx].cpu(),
                            "video_id": vid,
                        }
                        torch.save(saved_data, save_fpath)

    print("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
