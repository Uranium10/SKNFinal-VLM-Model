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

# Pin the small quotation LoRA inside the worker image. Runtime cold starts
# must not depend on Hugging Face availability or rate limits.
ARG ADAPTER_MODEL_ID=lyc9872/qwen_3.5_9b_peft
ARG ADAPTER_MODEL_REVISION=96ebf2f31aef9ad9c0b66f597e5eca4eccbc5885
ENV ADAPTER_LOCAL_PATH=/opt/models/quotation-lora
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${ADAPTER_MODEL_ID}', revision='${ADAPTER_MODEL_REVISION}', local_dir='${ADAPTER_LOCAL_PATH}', allow_patterns=['adapter_config.json', 'adapter_model.safetensors'])" \
    && test -s "${ADAPTER_LOCAL_PATH}/adapter_config.json" \
    && test -s "${ADAPTER_LOCAL_PATH}/adapter_model.safetensors" \
    && rm -rf /root/.cache/huggingface

COPY . /app/

CMD ["python", "-u", "/app/handler.py"]
