import os
import subprocess

# Define model URLs and paths
MODEL_PATHS = {
    "t5": {
        "url": "https://weights.replicate.delivery/default/official-models/flux/t5/t5-v1_1-xxl.tar",
        "cache": "./model-cache/t5"
    },
    "clip": {
        "url": "https://weights.replicate.delivery/default/official-models/flux/clip/clip-vit-large-patch14.tar",
        "cache": "./model-cache/clip"
    },
    "schnell": {
        "url": "https://weights.replicate.delivery/default/official-models/flux/schnell/schnell.sft",
        "cache": "./model-cache/schnell/schnell.sft"
    },
    "dev": {
        "url": "https://weights.replicate.delivery/default/official-models/flux/dev/dev.sft",
        "cache": "./model-cache/dev/dev.sft"
    },
    "ae": {
        "url": "https://weights.replicate.delivery/default/official-models/flux/ae/ae.sft",
        "cache": "./model-cache/ae/ae.sft"
    }
}

def download_models():
    # Create cache directories
    for model_info in MODEL_PATHS.values():
        os.makedirs(os.path.dirname(model_info["cache"]), exist_ok=True)

    # Download each model if it doesn't exist
    for model_name, model_info in MODEL_PATHS.items():
        if not os.path.exists(model_info["cache"]):
            print(f"Downloading {model_name} model...")
            try:
                subprocess.run([
                    "wget", 
                    "-O", 
                    model_info["cache"],
                    model_info["url"]
                ], check=True)
                print(f"Successfully downloaded {model_name}")
            except subprocess.CalledProcessError as e:
                print(f"Error downloading {model_name}: {e}")
        else:
            print(f"{model_name} model already exists at {model_info['cache']}")

if __name__ == "__main__":
    download_models()