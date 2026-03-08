import os
import runpod
import torch
import base64
from io import BytesIO

# Usamos DiffusionPipeline para que autodetecte la arquitectura nueva (Flux2KleinPipeline)
from diffusers import DiffusionPipeline

hf_token = os.environ.get("HF_TOKEN")

# Cargamos el modelo en memoria al iniciar el contenedor (Warm-up)
# Usamos Flux 2 en versión FP8 para que sea rápido y quepa en GPUs de 24GB
model_name = os.environ.get("MODEL_NAME", "black-forest-labs/FLUX.2-klein-9B")

print(f"Cargando {model_name}...")

pipe = DiffusionPipeline.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,
    token=hf_token
).to("cuda")

def handler(job):
    job_input = job.get("input", {})
    prompt = job_input.get("prompt")
    if not prompt:
        prompt = "a futuristic city"
    
    # Para la serie 2 de Flux, 20 pasos es un buen punto de partida
    image = pipe(
        prompt=prompt, 
        num_inference_steps=20, 
        guidance_scale=3.5
    ).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return {"image_base64": image_base64}

runpod.serverless.start({"handler": handler})
