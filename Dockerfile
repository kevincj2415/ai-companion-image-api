FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel

WORKDIR /app

# Instalar git primero ya que la imagen base no lo contiene
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Instalar diffusers desde git para soporte de Flux2KleinPipeline y huggingface-hub para bajar el repo
RUN pip install git+https://github.com/huggingface/diffusers.git transformers accelerate sentencepiece protobuf runpod huggingface_hub --upgrade --break-system-packages

# Argumento de build para el Token de Hugging Face
ARG HF_TOKEN
# Descargamos el modelo durante el build para hornearlo en la imagen
# Descargamos el modelo durante el build asegurándonos de usar python directamente
RUN python -c "\
from huggingface_hub import snapshot_download; \
import os; \
snapshot_download(repo_id='black-forest-labs/FLUX.2-klein-9B', token=os.environ.get('HF_TOKEN'))"

COPY handler.py .

CMD [ "python", "-u", "handler.py" ]
