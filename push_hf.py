from huggingface_hub import HfApi, create_repo, upload_file
import os

def push_to_hf(
    repo_name: str,
    token: str,
    model_dir: str = "./model-cache",
    organization: str = None  # Optional: specify if pushing to an organization
):
    # Initialize Hugging Face API
    api = HfApi()
    
    # Create the full repository name
    if organization:
        full_repo_name = f"{organization}/{repo_name}"
    else:
        full_repo_name = repo_name

    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=full_repo_name,
            token=token,
            repo_type="model",
            private=True  # Set to False if you want a public repository
        )
    except Exception as e:
        print(f"Repository might already exist or there was an error: {e}")

    # Walk through the model directory and upload all files
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            local_path = os.path.join(root, file)
            # Create relative path for HF
            relative_path = os.path.relpath(local_path, model_dir)
            
            print(f"Uploading {relative_path}...")
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=relative_path,
                    repo_id=full_repo_name,
                    token=token
                )
                print(f"Successfully uploaded {relative_path}")
            except Exception as e:
                print(f"Error uploading {relative_path}: {e}")

# Usage example
if __name__ == "__main__":
    # Get your HF token from: https://huggingface.co/settings/tokens
    HF_TOKEN = "your_huggingface_token"  
    REPO_NAME = "flux-schnell"  # Your desired repository name
    
    push_to_hf(
        repo_name=REPO_NAME,
        token=HF_TOKEN,
        model_dir="./model-cache",
        organization=None  # Set this if pushing to an organization
    )