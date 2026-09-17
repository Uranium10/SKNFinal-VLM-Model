"""Qwen3.5 multimodal runtime with the quotation LoRA adapter."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_ADAPTER_MODEL = "lyc9872/qwen_3.5_9b_peft"
DEFAULT_ADAPTER_REVISION = "96ebf2f31aef9ad9c0b66f597e5eca4eccbc5885"
HF_CACHE_ROOT = Path(os.getenv("HF_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub"))


def _cached_snapshot(model_id: str, revision: str | None = None) -> str | None:
    """Resolve a RunPod cached-model snapshot when one is mounted."""

    if "/" not in model_id:
        return None
    owner, name = model_id.split("/", 1)
    model_root = HF_CACHE_ROOT / f"models--{owner}--{name}"
    snapshots = model_root / "snapshots"
    if revision and (snapshots / revision).is_dir():
        return str(snapshots / revision)

    ref = model_root / "refs" / "main"
    if ref.is_file():
        candidate = snapshots / ref.read_text(encoding="utf-8").strip()
        if candidate.is_dir():
            return str(candidate)

    if snapshots.is_dir():
        candidates = sorted(
            (path for path in snapshots.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])
    return None


def _local_adapter_source(
    model_id: str,
    revision: str | None,
    local_path: str | None,
) -> str:
    """Resolve an adapter without ever falling back to the Hugging Face API."""

    if local_path:
        candidate = Path(local_path)
        if candidate.is_dir():
            return str(candidate)
        raise RuntimeError(f"quotation LoRA directory does not exist: {candidate}")
    cached = _cached_snapshot(model_id, revision)
    if cached:
        return cached
    raise RuntimeError(
        "quotation LoRA is not available locally; bake it into the worker "
        "image or mount it in the Hugging Face cache"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object while tolerating an accidental Markdown fence."""

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model output did not contain a JSON object")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model output JSON must be an object")
    return value


class QuotationModelRuntime:
    """Load the model once per worker and serialize GPU generation calls."""

    def __init__(self) -> None:
        self.base_model_id = os.getenv("BASE_MODEL_ID", DEFAULT_BASE_MODEL).strip()
        self.base_model_revision = os.getenv("BASE_MODEL_REVISION", "").strip() or None
        self.adapter_model_id = os.getenv("ADAPTER_MODEL_ID", DEFAULT_ADAPTER_MODEL).strip()
        self.adapter_local_path = os.getenv("ADAPTER_LOCAL_PATH", "").strip()
        self.adapter_revision = (
            os.getenv("ADAPTER_MODEL_REVISION", DEFAULT_ADAPTER_REVISION).strip() or None
        )
        self.min_pixels = int(os.getenv("VISION_MIN_PIXELS", str(256 * 28 * 28)))
        self.max_pixels = int(os.getenv("VISION_MAX_PIXELS", "802816"))
        self.attention = os.getenv("ATTN_IMPLEMENTATION", "sdpa").strip()
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device = "cuda:0"
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self.load_seconds: float | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            started = time.perf_counter()
            import torch
            from peft import PeftModel
            from transformers import AutoProcessor, BitsAndBytesConfig

            try:
                from transformers import AutoModelForMultimodalLM as AutoVisionModel
            except ImportError:  # pragma: no cover - compatibility fallback
                from transformers import AutoModelForImageTextToText as AutoVisionModel

            if not torch.cuda.is_available():
                raise RuntimeError("quotation vision inference requires a CUDA GPU")

            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            base_source = _cached_snapshot(self.base_model_id, self.base_model_revision)
            base_source = base_source or self.base_model_id
            common: dict[str, Any] = {
                "trust_remote_code": False,
            }
            if self.base_model_revision and base_source == self.base_model_id:
                common["revision"] = self.base_model_revision

            self._processor = AutoProcessor.from_pretrained(
                base_source,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                **common,
            )
            base_model = AutoVisionModel.from_pretrained(
                base_source,
                device_map="auto",
                dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation=self.attention,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                ),
                **common,
            )
            adapter_source = _local_adapter_source(
                self.adapter_model_id,
                self.adapter_revision,
                self.adapter_local_path,
            )
            adapter_kwargs: dict[str, Any] = {
                "is_trainable": False,
                "local_files_only": True,
            }
            self._model = PeftModel.from_pretrained(
                base_model,
                adapter_source,
                **adapter_kwargs,
            ).eval()
            self._torch = torch
            self._device = str(next(self._model.parameters()).device)
            torch.cuda.synchronize()
            self.load_seconds = time.perf_counter() - started

    def extract(
        self,
        images: list[Image.Image],
        document_text: str,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> tuple[dict[str, Any], str, float]:
        self.load()
        user_text = user_prompt.replace("<image>\n", "", 1)
        if document_text:
            user_text += (
                "\n\n[Python 라이브러리로 추출한 견적 원문 및 표]\n"
                + document_text
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": image} for image in images),
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        with self._generation_lock, self._torch.inference_mode():
            prompt = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            processor_kwargs: dict[str, Any] = {
                "text": [prompt],
                "return_tensors": "pt",
            }
            if images:
                processor_kwargs["images"] = images
            inputs = self._processor(**processor_kwargs)
            inputs = {
                key: value.to(self._device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            input_length = inputs["input_ids"].shape[-1]
            self._torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            self._torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - started
            generated = outputs[:, input_length:]
            raw_text = self._processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        return extract_json_object(raw_text), raw_text, generation_seconds


    def model_revision(self) -> str | None:
        config = getattr(self._model, "config", None)
        return getattr(config, "_commit_hash", None) if config is not None else None

