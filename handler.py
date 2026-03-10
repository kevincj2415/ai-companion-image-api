import os
import runpod
import torch
import base64
from io import BytesIO

# Usamos Flux2KleinPipeline tal y como indica la documentación de HuggingFace
from diffusers import Flux2KleinPipeline

# Ya no necesitamos HF_TOKEN en tiempo de ejecución porque
# el Dockerfile descargó los pesos localmente en la caché de la imagen.
model_name = os.environ.get("MODEL_NAME", "black-forest-labs/FLUX.2-klein-9B")
print(f"Cargando {model_name}...")

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained(
    model_name, 
    torch_dtype=dtype,
    local_files_only=False # Cambiamos a false para prevenir errores si falta un archivo menor, pero usará caché primordialmente
)
# Ahorrar VRAM descargando partes del modelo a CPU según la documentación
pipe.enable_model_cpu_offload()

def handler(job):
    job_input = job.get("input", {})
    prompt = job_input.get("prompt")
    if not prompt:
        prompt = "a futuristic city"
    
    # Parámetros recomendados por la documentación de BFL (guidance_scale=1.0, 4 pasos)
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=1.0, 
        num_inference_steps=4,
        generator=torch.Generator(device=device).manual_seed(0)
    ).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return {"image_base64": image_base64}

runpod.serverless.start({"handler": handler})
