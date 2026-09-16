# Keep the CUDA/PyTorch versions used by the worker while avoiding the much
# larger RunPod development image. A smaller runtime image reduces Serverless
# image-pull time and therefore cold-start latency.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/runpod-volume/huggingface-cache

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --break-system-packages --no-deps \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torchvision==0.23.0" \
    && python -m pip install --no-cache-dir --break-system-packages \
    --ignore-installed cryptography "runpod~=1.7.6" \
    && python -m pip install --no-cache-dir --break-system-packages \
    -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

COPY . /app/

CMD ["python", "-u", "/app/handler.py"]
