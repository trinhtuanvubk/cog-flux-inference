import os
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from typing import List, Tuple

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from flux.sampling import denoise, get_noise, get_schedule, prepare
from flux.util import load_ae, load_clip, load_flow_model, load_t5
from fp8.flux_pipeline import FluxPipeline
from fp8.util import LoadedModels

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "3:2": (1216, 832),
    "4:5": (896, 1088),
}

class SchnellPredictor:
    def __init__(self):
        self.device = "cuda"
        self.num_steps = 4
        self.shift = False
        print("Initializing Schnell model...")
        self.setup()

    def setup(self) -> None:
        # Load base models
        self.t5 = load_t5(self.device, max_length=256)
        self.clip = load_clip(self.device)
        self.flux = load_flow_model("flux-schnell", device=self.device)
        self.flux = self.flux.eval()
        self.ae = load_ae("flux-schnell", device=self.device)

        # Initialize FP8 pipeline
        shared_models = LoadedModels(
            flow=None, 
            ae=self.ae, 
            clip=self.clip, 
            t5=self.t5, 
            config=None
        )

        self.fp8_pipe = FluxPipeline.load_pipeline_from_config_path(
            "fp8/configs/config-1-flux-schnell-h100.json",
            shared_models=shared_models,
            compile_whole_model=True,
            compile_extras=True,
            compile_blocks=True
        )
        
        # Warm up the model
        print("Warming up model...")
        self.fp8_pipe.generate(
            prompt="warmup",
            width=1024,
            height=1024,
            num_steps=4,
            guidance=3,
            seed=123
        )

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        num_outputs: int = 1,
        seed: int = None,
        go_fast: bool = True
    ) -> List[Image.Image]:
        width, height = ASPECT_RATIOS.get(aspect_ratio, (1024, 1024))
        
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if go_fast:
            # Use FP8 pipeline for faster inference
            images = self.fp8_pipe.generate(
                prompt=prompt,
                width=width,
                height=height,
                num_steps=self.num_steps,
                guidance=3.0,
                seed=seed,
                num_images=num_outputs
            )
        else:
            # Use base BF16 pipeline
            images = self._base_generate(
                prompt=prompt,
                num_outputs=num_outputs,
                seed=seed,
                width=width,
                height=height
            )
        
        return images

    def _base_generate(
        self,
        prompt: str,
        num_outputs: int,
        seed: int,
        width: int = 1024,
        height: int = 1024,
    ) -> List[Image.Image]:
        # Get initial noise
        x = get_noise(
            num_outputs,
            height,
            width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=seed
        )
        
        # Get timesteps schedule
        timesteps = get_schedule(
            self.num_steps, 
            (x.shape[-1] * x.shape[-2]) // 4,
            shift=self.shift
        )

        # Prepare inputs
        inp = prepare(
            t5=self.t5,
            clip=self.clip,
            img=x,
            prompt=[prompt] * num_outputs
        )

        # Run denoising
        x, _ = denoise(
            self.flux,
            **inp,
            timesteps=timesteps,
            guidance=3.0
        )

        # Decode latents
        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        # Convert to PIL images
        images = []
        for i in range(num_outputs):
            img_array = (127.5 * (rearrange(x[i], "c h w -> h w c").clamp(-1, 1) + 1.0))\
                .cpu()\
                .byte()\
                .numpy()
            images.append(Image.fromarray(img_array))

        return images

def unpack(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return rearrange(
        x, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=height//8, w=width//8, ph=2, pw=2
    )

# Example usage
if __name__ == "__main__":
    model = SchnellPredictor()
    
    # Generate images
    images = model.generate(
        prompt="a beautiful mountain landscape at sunset",
        aspect_ratio="16:9",
        num_outputs=1,
        seed=42,
        go_fast=True
    )[0]
    
    # Save images
    # for i, img in enumerate(images):
    img.save(f"output_0.png")