import os
import runpod
import torch
import base64
from io import BytesIO

# Esta es la línea clave que faltaba para que Python entienda qué es FluxPipeline
from diffusers import FluxPipeline

# Cargamos el token de Hugging Face configurado en RunPod
hf_token = os.environ.get("HF_TOKEN")

# Inicializamos el modelo oficial (versión rápida y sin filtros)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B", 
    torch_dtype=torch.bfloat16,
    token=hf_token
).to("cuda")

def handler(job):
    job_input = job["input"]
    prompt = job_input.get("prompt", "a futuristic city")
    
    # Generación de la imagen (4 pasos es el estándar para schnell)
    image = pipe(
        prompt, 
        num_inference_steps=4, 
        guidance_scale=0.0
    ).images[0]

    # Conversión a base64 para que la API lo pueda devolver
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return {"image_base64": image_base64}

# Arranque del worker de RunPod
runpod.serverless.start({"handler": handler})
