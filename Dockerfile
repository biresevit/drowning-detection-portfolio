FROM nvidia/cuda:13.0.1-cudnn8-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV FORCE_CUDA="1"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    libopenblas-dev \
    ffmpeg \
    docker.io \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy wheel files and install PyTorch, torchvision, onnxruntime (x86_64)
COPY install/*.whl /tmp/wheels/
RUN pip3 install /tmp/wheels/torch-2.7.0-cp310-cp310-linux_x86_64.whl
RUN pip3 install /tmp/wheels/torchvision-0.22.0-cp310-cp310-linux_x86_64.whl
RUN pip3 install /tmp/wheels/onnxruntime_gpu-1.22.0-cp310-cp310-linux_x86_64.whl

# Install additional dependencies
RUN pip3 install git+https://github.com/facebookresearch/pytorchvideo

# Install requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Create models folder
RUN mkdir -p /models

# Copy environment file (cloud.env or no-cloud.env)
ARG ENV_FILE=cloud.env
COPY ${ENV_FILE} /app/.env

# Download models if no-cloud
RUN --mount=type=secret,id=azure_cred \
    python3 - <<'PYCODE'
import os
if os.environ.get("ENV_FILE") == "no-cloud.env":
    from azure.ai.ml import MLClient
    from azure.identity import ClientSecretCredential

    cred_file = '/run/secrets/azure_cred'
    with open(cred_file) as f:
        lines = f.read().splitlines()
    env = dict(line.split('=', 1) for line in lines)

    credential = ClientSecretCredential(
        tenant_id=env['AZURE_TENANT_ID'],
        client_id=env['AZURE_CLIENT_ID'],
        client_secret=env['AZURE_CLIENT_SECRET']
    )

    DOWNLOAD_PATH = "/models"
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    SUBSCRIPTION_ID = os.environ["AZURE_SUBSCRIPTION_ID"]
    RESOURCE_GROUP = os.environ["AZURE_RESOURCE_GROUP"]
    WORKSPACE = os.environ["AZURE_ML_WORKSPACE"]

    ml_client = MLClient(credential,
                         subscription_id=SUBSCRIPTION_ID,
                         resource_group_name=RESOURCE_GROUP,
                         workspace_name=WORKSPACE)

    required_models = ["humans", "water_640", "drowning_x3d_s"]

    for model_name in required_models:
        model = ml_client.models.get(name=model_name, label="latest")
        ml_client.models.download(name=model_name,
                                  version=model.version,
                                  download_path=DOWNLOAD_PATH)
PYCODE

# Flatten model folders
RUN [ "$ENV_FILE" = "no-cloud.env" ] && \
    for d in /models/*; do \
        if [ -d "$d" ]; then \
            f=$(ls -A $d | head -n 1); \
            if [ -n "$f" ]; then \
                mv "$d/$f" /models/; \
            fi; \
            rmdir "$d" 2>/dev/null || true; \
        fi; \
    done || true

RUN rm -rf /tmp/wheels/

# Install Python docker package
RUN pip3 install docker

# Set working directory
WORKDIR /app
ENV PYTHONPATH=/app/src:/usr/local/lib/python3.10/dist-packages

# Copy project source
COPY src/ /app/src/

# Remove unneeded scripts based on env
RUN [ "$ENV_FILE" = "no-cloud.env" ] && rm -f /app/src/action_recognition/check_model_registry.py || true
RUN [ "$ENV_FILE" = "cloud.env" ] && rm -f /app/src/rest_api_processing.py || true

# Install PyInstaller
RUN pip3 install pyinstaller

# Create runtime hook
RUN echo 'import sys, os\n'\
'import os\n'\
'python_path = os.environ.get("PYTHONPATH")\n'\
'if python_path:\n'\
'    for p in python_path.split(os.pathsep):\n'\
'        if p not in sys.path:\n'\
'            sys.path.append(p)\n' > /app/runtime-hook.py

# Build exe only for no-cloud mode
RUN [ "$ENV_FILE" = "no-cloud.env" ] && \
    pyinstaller --onedir /app/src/rest_api_processing.py \
                --distpath /app/bin \
                --name rest_api_processing \
                --runtime-hook=/app/runtime-hook.py \
                --exclude-module torch \
                --exclude-module torchvision \
                --exclude-module onnxruntime \
                --exclude-module numpy \
                --exclude-module mlflow \
                --exclude-module scikit_learn \
                --exclude-module python_dotenv \
                --exclude-module pycuda \
                --exclude-module ultralytics \
                --exclude-module ffmpeg_python \
                --exclude-module tqdm \
                --exclude-module schedule \
                --exclude-module azure_identity \
                --exclude-module azure_ai_ml \
                --exclude-module opentelemetry_api \
                --exclude-module opentelemetry_sdk \
                --exclude-module opentelemetry_exporter_prometheus \
                --exclude-module opencv_contrib_python \
                --exclude-module pytorchvideo \
                --hidden-import=unittest.mock \
                --hidden-import=modulefinder \
                --hidden-import=html.parser \
                --hidden-import=cProfile \
                --collect-all PIL || true

RUN [ "$ENV_FILE" = "no-cloud.env" ] && rm -rf /app/src || true

ARG CACHEBUST=1
