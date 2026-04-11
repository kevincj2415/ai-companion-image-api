import os
import runpod
import torch
import base64
from io import BytesIO
from PIL import Image

# Usamos AutoPipelineForImage2Image para compatibilidad con Image-to-Image y usamos el pipeline original para TXT2IMG
from diffusers import Flux2KleinPipeline, AutoPipelineForImage2Image

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

# Configurar pipeline de Image-to-Image. Con from_pipe, reutilizamos los componentes en memoria
# para no duplicar el consumo de VRAM ni de RAM.
img2img_pipe = AutoPipelineForImage2Image.from_pipe(pipe)
img2img_pipe.enable_model_cpu_offload()

def handler(job):
    job_input = job.get("input", {})
    prompt = job_input.get("prompt")
    if not prompt:
        prompt = "a futuristic city"
        
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    num_inference_steps = job_input.get("num_inference_steps", 4)
    guidance_scale = job_input.get("guidance_scale", 1.0)
    seed = job_input.get("seed", 0)
    
    # Extraer variables para image-to-image
    input_image_b64 = job_input.get("image")
    strength = job_input.get("strength", 0.5)

    generator = torch.Generator(device=device).manual_seed(seed)
    
    if input_image_b64:
        try:
            # Detectar cabeceras en base64 en caso de venir de web client, o asumir puro base64
            if "," in input_image_b64:
                input_image_b64 = input_image_b64.split(",")[1]
            
            image_bytes = base64.b64decode(input_image_b64)
            init_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            
            # Prevenir imágenes gigantescas o forzar dimensiones (Opcional, de momento generamos normal)
            init_image = init_image.resize((width, height))
            
            # Ejecutar inferencia Image-to-Image
            image = img2img_pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images[0]
            
        except Exception as e:
            return {"error": f"Hubo un error procesando la imagen de entrada: {str(e)}"}
    else:
        # Parámetros recomendados por la documentación de BFL (guidance_scale=1.0, 4 pasos)
        # Ejecutar inferencia Text-to-Image
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale, 
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return {"image_base64": image_base64}

runpod.serverless.start({"handler": handler})
