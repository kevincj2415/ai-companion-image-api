import os
import runpod
import torch
from diffusers import DiffusionPipeline
import base64
from io import BytesIO

# 1. Leemos el token por si el modelo lo requiere
hf_token = os.environ.get("HF_TOKEN")

# 2. Usamos EXACTAMENTE el modelo y la clase que encontraste en la documentación
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B", 
    torch_dtype=torch.bfloat16,
    token=hf_token
).to("cuda")

def handler(job):
    job_input = job["input"]
    prompt = job_input.get("prompt")
    
    # 3. Generamos la imagen basada solo en tu texto (Text-to-Image)
    image = pipe(
        prompt=prompt,
        num_inference_steps=25, 
        guidance_scale=3.5
    ).images[0]

    # 4. Empaquetamos en Base64 para enviarlo a tu app
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return {"image_base64": base64.b64encode(buffer.getvalue()).decode("utf-8")}

runpod.serverless.start({"handler": handler})
