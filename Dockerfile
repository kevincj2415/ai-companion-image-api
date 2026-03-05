FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

WORKDIR /app

# Agregamos --break-system-packages para saltar el bloqueo de Linux
RUN pip install runpod diffusers transformers accelerate sentencepiece protobuf --upgrade --break-system-packages

COPY handler.py .

CMD [ "python", "-u", "handler.py" ]
