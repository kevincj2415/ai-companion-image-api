FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

WORKDIR /app
RUN pip install runpod diffusers==0.30.2 transformers==4.44.2 accelerate sentencepiece protobuf

COPY handler.py .

CMD [ "python", "-u", "handler.py" ]
