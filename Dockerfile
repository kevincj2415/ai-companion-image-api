FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel

WORKDIR /app

# Instalar diffusers desde git para soporte de Flux2KleinPipeline
RUN pip install git+https://github.com/huggingface/diffusers.git transformers accelerate sentencepiece protobuf runpod --upgrade --break-system-packages

COPY handler.py .

CMD [ "python", "-u", "handler.py" ]
