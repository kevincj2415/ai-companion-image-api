import runpod
import torch
import os
from diffusers import FluxPipeline # O el pipeline de Seedream 4.5
import base64
from io import BytesIO

hf_token = os.environ.get("HF_TOKEN")
# Cargamos el modelo en memoria al iniciar el contenedor (Warm-up)
# Usamos Flux 2 en versión FP8 para que sea rápido y quepa en GPUs de 24GB
model_name = os.environ.get("MODEL_NAME", "black-forest-labs/FLUX.2-klein-9B")
pipe = FluxPipeline.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,
    token=hf_token
).to("cuda")

def handler(job):
    job_input = job.get("input", {})
    prompt = job_input.get("prompt")
    
    if not prompt:
        return {"error": "El campo 'prompt' es requerido en el input."}
    
    # Generación
    image = pipe(
        prompt, 
        num_inference_steps=20, 
        guidance_scale=3.5
    ).images[0]

    # Convertir a Base64 para devolverlo por la API
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return {"image": base64.b64encode(buffer.getvalue()).decode("utf-8")}

runpod.serverless.start({"handler": handler})
