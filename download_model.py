import torch
import os
from diffusers import FluxPipeline

def download_model():
    model_name = os.environ.get("MODEL_NAME", "black-forest-labs/FLUX.2-flex-fp8")
    print(f"Downloading {model_name} model...")
    # Esto descargará el modelo y lo guardará en caché en la imagen de Docker
    FluxPipeline.from_pretrained(
        model_name, 
        torch_dtype=torch.float8_e4m3fn
    )
    print("Download complete!")

if __name__ == "__main__":
    download_model()
