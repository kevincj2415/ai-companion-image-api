FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descargar el modelo dentro de la imagen para evitar cold starts largos
COPY download_model.py .
RUN python download_model.py

COPY handler.py .

# Comando de inicio para RunPod
CMD [ "python", "-u", "handler.py" ]