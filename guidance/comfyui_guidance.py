import os
import random
import sys
from typing import Sequence, Mapping, Any, Union

from dataclasses import dataclass, field
from typing import List

import numpy as np
import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = "custom/threestudio-comfyui/ComfyUI"
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")
    else:
        raise RuntimeError("Could not find the ComfyUI directory.")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from ..ComfyUI.main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
# add_extra_model_paths()

from ..ComfyUI.nodes import (
    EmptyLatentImage,
    VAEDecode,
    VAEEncode,
    SaveImage,
    CheckpointLoaderSimple,
    CLIPTextEncode,
    KSampler,
    NODE_CLASS_MAPPINGS,
)

@threestudio.register("comfyui-guidance")
class ComfyUIGUidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = ""

    cfg: Config

    def configure(self) -> None:

        self.checkpointloadersimple = CheckpointLoaderSimple()
        self.checkpointloadersimple_4 = self.checkpointloadersimple.load_checkpoint(
            ckpt_name="v2-1_768-ema-pruned.safetensors"
        )
    
    def densify(self, factor=2):
        pass

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        )
        img_input = rgb_BCHW_512.permute(0, 2, 3, 1)

        with torch.no_grad():
            # emptylatentimage = EmptyLatentImage()
            # emptylatentimage_5 = emptylatentimage.generate(
            #     width=512, height=512, batch_size=1
            # )
            
            vaeencode = VAEEncode()
            emptylatentimage_5 = vaeencode.encode(
                pixels=img_input,
                vae=get_value_at_index(self.checkpointloadersimple_4, 2),
            )
            
            cliptextencode = CLIPTextEncode()
            cliptextencode_6 = cliptextencode.encode(
                text="a delicious hamburger",
                clip=get_value_at_index(self.checkpointloadersimple_4, 1),
            )

            cliptextencode_7 = cliptextencode.encode(
                text="text, watermark", clip=get_value_at_index(self.checkpointloadersimple_4, 1)
            )

            ksampler = KSampler()

            ksampler_3 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=1,
                cfg=100,
                sampler_name="euler",
                scheduler="normal",
                denoise=random.randint(20, 980) / 1000,
                model=get_value_at_index(self.checkpointloadersimple_4, 0),
                positive=get_value_at_index(cliptextencode_6, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                latent_image=get_value_at_index(emptylatentimage_5, 0),
            )
            # ksampler_3 = ksampler.sample(
            #     seed=random.randint(1, 2**64),
            #     steps=20,
            #     cfg=8,
            #     sampler_name="euler",
            #     scheduler="normal",
            #     denoise=1.0,
            #     model=get_value_at_index(self.checkpointloadersimple_4, 0),
            #     positive=get_value_at_index(cliptextencode_6, 0),
            #     negative=get_value_at_index(cliptextencode_7, 0),
            #     latent_image=get_value_at_index(emptylatentimage_5, 0),
            # )

            pred_latents = get_value_at_index(ksampler_3, 0)['samples']

            vaedecode = VAEDecode()
            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(self.checkpointloadersimple_4, 2),
            )
            # saveimage = SaveImage()

            # saveimage_9 = saveimage.save_images(
            #     filename_prefix="threestudio_test", images=get_value_at_index(vaedecode_8, 0)
            # )
        pred_image = get_value_at_index(vaedecode_8, 0).detach().to(img_input.device)
        loss_sds = 0.5 * F.mse_loss(img_input, pred_image, reduction="sum")

        return {
            "loss_sds": loss_sds,
        }