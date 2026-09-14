FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

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
