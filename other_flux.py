import logging
import time
from enum import Enum
from typing import Literal

import torch
from diffusers import AutoencoderTiny, FluxTransformer2DModel
from PIL import Image
from torchao.quantization import float8_dynamic_activation_float8_weight, quantize_
from torchao.quantization.quant_api import PerRow
from transformers import T5EncoderModel
import diffusers
import transformers


# First, disable scaled dot product attention to use regular attention instead
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

pipe = diffusers.FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    vae=None,
    transformer=None,
    text_encoder_2=None,
    torch_dtype=torch.bfloat16,
).to("cuda")
pipe.scheduler.max_shift = 1.2

text_encoder_2 = transformers.T5EncoderModel.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", subfolder="text_encoder_2", torch_dtype=torch.bfloat16
).to("cuda")
vae = diffusers.AutoencoderTiny.from_pretrained(
    "madebyollin/taef1", torch_dtype=torch.bfloat16
)

print("hihi")
transformer = diffusers.FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

transformer.fuse_qkv_projections()
torch.cuda.empty_cache()

quantize_(
    transformer,
    float8_dynamic_activation_float8_weight(granularity=PerRow()),
    ignore_modules=["self_attn"],
    device="cuda",
)
transformer.to(memory_format=torch.channels_last)
transformer = torch.compile(
    transformer
)
torch.cuda.empty_cache()

quantize_(
    vae,
    float8_dynamic_activation_float8_weight(granularity=PerRow()),
    device="cuda",
)
vae.to(memory_format=torch.channels_last)
vae.decoder = torch.compile(
    vae.decoder
)

pipe.vae = vae
pipe.transformer = transformer
pipe.text_encoder_2 = text_encoder_2

pipe.set_progress_bar_config(disable=True)

# breakpoint()

with torch.inference_mode():
    image=pipe("a bird",
                generator=None,
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=256, 
                height=1024,
                width=1024, output_type="pil").images[0]
    image.save("hihi.jpg")