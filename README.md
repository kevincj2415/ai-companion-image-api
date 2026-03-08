# AI Companion Image API

Este proyecto proporciona una API serverless diseñada para ser ejecutada en **RunPod** que permite la generación de imágenes a partir de texto (Text-to-Image). Utiliza el modelo de vanguardia **FLUX.2-klein-9B** de Black Forest Labs mediante la librería `diffusers` de Hugging Face.

## Características

*   **Generación Text-to-Image**: Convierte descripciones de texto en imágenes detalladas.
*   **Modelo FLUX.2**: Emplea `Flux2KleinPipeline` para obtener resultados de alta calidad.
*   **RunPod Serverless**: Estructurado específicamente para funcionar como un endpoint sin servidor en RunPod (`runpod.serverless`).
*   **Optimización de VRAM**: Implementa la descarga fluida a la CPU (`enable_model_cpu_offload()`) para evitar problemas de falta de memoria (OOM) durante la inferencia.
*   **Salida en Base64**: Retorna la imagen generada directamente codificada en Base64 para facilitar su consumo desde clientes o interfaces.

## Requisitos Previos

Para ejecutar y probar este entorno necesitarás:
*   Docker y Docker Compose (para construcción local).
*   Una cuenta de [RunPod](https://www.runpod.io/).
*   Un **Token de Hugging Face** (`HF_TOKEN`) con permisos de acceso al repositorio del modelo `black-forest-labs/FLUX.2-klein-9B` (si el modelo requiere aceptación de términos).

## Estructura de Archivos

*   `handler.py`: El script principal de la aplicación. Configura la tubería de Hugging Face y define el comportamiento para cada petición (job) gestionado por RunPod.
*   `Dockerfile`: Las instrucciones de construcción del contenedor, configurando dependencias de SO, clonando versiones recientes de librerías base (como diffusers directo de git para soporte de última hora) y ajustando el entorno.
*   `requirements.txt`: Dependencias de Python necesarias.

## Configuración y Despliegue

### Variables de Entorno

El contenedor o endpoint de RunPod debe tener configuradas las siguientes variables de entorno:

| Variable | Descripción | Valor por defecto |
| :--- | :--- | :--- |
| `HF_TOKEN` | Tu Token de acceso de Hugging Face (Requerido). | `None` |
| `MODEL_NAME` | El string identificador de Hugging Face para el modelo. | `black-forest-labs/FLUX.2-klein-9B` |

### Construcción de la Imagen Docker

Construye la imagen Docker localmente (asegúrate de etiquetarla para tu registro de contenedores, por ejemplo, Docker Hub):

```bash
docker build -t tu_usuario/ai-companion-image-api:latest .
```

Sube la imagen a tu repositorio:

```bash
docker push tu_usuario/ai-companion-image-api:latest
```

### Creación del Endpoint en RunPod

1. Ve a tu consola de **RunPod** -> **Serverless** -> **Templates**.
2. Crea un nuevo Template utilizando la imagen de tu contenedor (`tu_usuario/ai-companion-image-api:latest`).
3. Agrega las variables de entorno `HF_TOKEN` (y `MODEL_NAME` si deseas anularlo).
4. Guarda el Template.
5. Ve a **Endpoints** y despliega un nuevo punto de conexión asociándolo al Template que acabas de crear. Se recomienda usar GPUs con suficiente VRAM (ej. 24GB+ dependiendo del modelo).

## Uso de la API

Una vez desplegado el Endpoint, puedes enviarle peticiones POST o utilizar el SDK de RunPod. El esquema esperado para el cuerpo de la solicitud JSON es:

### Payload de Entrada

```json
{
  "input": {
    "prompt": "una ciudad futurista cyberpunk bajo la lluvia, iluminación de neón"
  }
}
```

*Nota: Si se omite el `prompt`, se usará `"a futuristic city"` por defecto.*

### Respuesta (Ejemplo)

El endpoint devolverá un objeto JSON donde el resultado de la inferencia se encuentra codificado en `image_base64`.

```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

Puedes decodificar esta cadena Base64 desde tu cliente / frontend para mostrar la imagen resultante.

## Notas Técnicas

*   Las inferencias se están realizando a `1024x1024` de resolución.
*   El modelo `FLUX.2` utiliza valores de `guidance_scale=1.0` y `num_inference_steps=4` según las recomendaciones predeterminadas en `handler.py` de BFL, agilizando enormemente los tiempos de generación sin afectar drásticamente el resultado.
