import torch
import diffusers
import transformers
from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, int8_weight_only
from torchao.quantization.quant_api import PerRow
import gc

FLUX_CHECKPOINT = "black-forest-labs/FLUX.1-schnell"
DTYPE = torch.bfloat16
NUM_STEPS = 4

def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

def get_transformer(quantize: bool = True):
    model = diffusers.FluxTransformer2DModel.from_pretrained(
        FLUX_CHECKPOINT, 
        subfolder="transformer", 
        torch_dtype=DTYPE
    )
    if quantize:
        try:
            quantize_(
                model, 
                float8_dynamic_activation_float8_weight(granularity=PerRow()),
                device="cuda"
            )
            model = model.to(memory_format=torch.channels_last)
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except RuntimeError as e:
            print(f"Float8 quantization failed, falling back to int8: {e}")
            quantize_(model, int8_weight_only())
    return model

def get_text_encoder_2(quantize: bool = True):
    model = transformers.T5EncoderModel.from_pretrained(
        FLUX_CHECKPOINT, 
        subfolder="text_encoder_2", 
        torch_dtype=DTYPE
    )
    if quantize:
        try:
            quantize_(model, int8_weight_only())
            model = model.to(memory_format=torch.channels_last)
        except RuntimeError as e:
            print(f"Quantization failed for text encoder: {e}")
    return model

def get_vae(quantize: bool = True):
    model = diffusers.AutoencoderTiny.from_pretrained(
        "madebyollin/taef1", 
        torch_dtype=DTYPE
    )
    if quantize:
        try:
            quantize_(
                model,
                float8_dynamic_activation_float8_weight(),
                device="cuda"
            )
            model = model.to(memory_format=torch.channels_last)
            model.decoder = torch.compile(
                model.decoder, 
                mode="reduce-overhead",
                fullgraph=False
            )
        except RuntimeError as e:
            print(f"Float8 quantization failed for VAE, falling back to int8: {e}")
            quantize_(model, int8_weight_only())
    return model

def load_pipeline():
    # Disable flash attention if causing issues
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    
    empty_cache()
    
    # Load base pipeline
    pipe = diffusers.FluxPipeline.from_pretrained(
        FLUX_CHECKPOINT,
        torch_dtype=DTYPE,
        vae=None,
        transformer=None,
        text_encoder_2=None,
    )
    pipe.scheduler.max_shift = 1.2
    
    # Load and quantize components
    pipe.transformer = get_transformer(quantize=True)
    pipe.text_encoder_2 = get_text_encoder_2(quantize=True)
    pipe.vae = get_vae(quantize=True)
    
    # Enable CPU offload for memory efficiency
    pipe._exclude_from_cpu_offload = ["vae"]
    pipe.enable_sequential_cpu_offload()
    
    # Warmup run
    empty_cache()
    pipe(
        "warmup",
        guidance_scale=0.0,
        max_sequence_length=256,
        num_inference_steps=NUM_STEPS
    )
    
    return pipe

@torch.inference_mode()
def infer(prompt: str, width: int = 1024, height: int = 1024, seed: int = None, pipeline=None):
    if pipeline is None:
        pipeline = load_pipeline()
        
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = None

    empty_cache()
    
    image = pipeline(
        prompt=prompt,
        width=width,
        height=height,
        guidance_scale=0.0,
        generator=generator,
        output_type="pil",
        max_sequence_length=256,
        num_inference_steps=NUM_STEPS
    ).images[0]
    
    return image

# Example usage
if __name__ == "__main__":
    pipe = load_pipeline()
    image = infer(
        prompt="a beautiful mountain landscape",
        width=1024,
        height=1024,
        seed=42,
        pipeline=pipe
    )
    image.save("output.jpg")