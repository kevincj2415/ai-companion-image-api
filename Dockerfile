FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel

WORKDIR /app

# Instalar git primero ya que la imagen base no lo contiene
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Instalar diffusers desde git para soporte de Flux2KleinPipeline
RUN pip install git+https://github.com/huggingface/diffusers.git transformers accelerate sentencepiece protobuf runpod --upgrade --break-system-packages

COPY handler.py .

CMD [ "python", "-u", "handler.py" ]
