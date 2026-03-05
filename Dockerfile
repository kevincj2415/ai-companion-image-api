FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel

WORKDIR /app
# El --upgrade asegura que bajemos la versión de diffusers que soporta Flux 2
RUN pip install runpod diffusers transformers accelerate sentencepiece protobuf --upgrade

COPY handler.py .

CMD [ "python", "-u", "handler.py" ]
