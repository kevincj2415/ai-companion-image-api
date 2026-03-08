FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel

WORKDIR /app

# Agregamos --break-system-packages para saltar el bloqueo de Linux
RUN pip install runpod diffusers transformers accelerate sentencepiece protobuf --upgrade --break-system-packages

COPY handler.py .

CMD [ "python", "-u", "handler.py" ]
