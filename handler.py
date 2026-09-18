"""RunPod queue worker entry point for quotation extraction."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import runpod

from document_input import (
    documents_to_images,
    read_document_text,
    read_documents,
)
from model_runtime import QuotationModelRuntime
from prompt_contract import prompt_values
from quotation_pipeline import run_quotation_pipeline


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("biddingflow.quotation_worker")
RUNTIME = QuotationModelRuntime()
WORKER_VERSION = "document-text-v1"


def _max_new_tokens(payload: dict[str, Any]) -> int:
    default = int(os.getenv("MAX_NEW_TOKENS", "512"))
    cap = int(os.getenv("MAX_NEW_TOKENS_CAP", "1024"))
    requested = int(payload.get("max_new_tokens") or default)
    if requested <= 0 or requested > cap:
        raise ValueError(f"max_new_tokens must be between 1 and {cap}")
    return requested


def _ocr_max_new_tokens() -> int:
    value = int(os.getenv("OCR_MAX_NEW_TOKENS", "2048"))
    cap = int(os.getenv("OCR_MAX_NEW_TOKENS_CAP", "4096"))
    if value <= 0 or value > cap:
        raise ValueError(f"OCR_MAX_NEW_TOKENS must be between 1 and {cap}")
    return value


def handler(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input")
    if not isinstance(payload, dict):
        raise ValueError("RunPod event must contain an input object")

    if payload.get("healthcheck") is True:
        return {
            "status": "ok",
            "worker": "biddingflow-quotation-extractor",
            "worker_version": WORKER_VERSION,
            "model_loaded": RUNTIME.loaded,
            "prompt_source": "request",
        }

    # Explicit paid preparation requested by the administrator's timed lease.
    # Do not alter the cheap build healthcheck or require a real quotation.
    if payload.get("warmup") is True:
        RUNTIME.load()
        return {
            "status": "ok", "worker": "biddingflow-quotation-extractor",
            "worker_version": WORKER_VERSION,
            "model_loaded": RUNTIME.loaded,
            "model_load_seconds": RUNTIME.load_seconds,
        }

    request_id = str(payload.get("request_id") or event.get("id") or "").strip()
    if not request_id:
        raise ValueError("request_id is required")
    required_pipeline = str(payload.get("pipeline_version") or "").strip()
    if required_pipeline and required_pipeline != WORKER_VERSION:
        raise ValueError("unsupported pipeline_version")
    # "extraction" (default) preserves the existing ERP-mapping contract.
    # Other values (e.g. "spec_eval") skip the strict extraction-schema
    # validation so the same base model can be reused for other structured
    # JSON tasks driven purely by system_prompt/user_prompt.
    task = str(payload.get("task") or "extraction").strip().lower()

    system_prompt, user_prompt, prompt_version, prompt_hash = prompt_values(payload)
    document_text = read_document_text(payload)
    documents = read_documents(payload)
    if not documents and not document_text:
        raise ValueError("at least one document or document_text is required")
    images = documents_to_images(documents)
    started = time.perf_counter()
    try:
        result = run_quotation_pipeline(
            RUNTIME,
            images,
            document_text,
            system_prompt,
            user_prompt,
            _max_new_tokens(payload),
            _ocr_max_new_tokens(),
            task=task,
        )
    finally:
        for image in images:
            image.close()

    elapsed_seconds = time.perf_counter() - started
    LOGGER.info(
        "quotation extracted request_id=%s image_views=%d ocr=%s elapsed=%.3fs",
        request_id,
        len(images),
        bool(images),
        elapsed_seconds,
    )
    response: dict[str, Any] = {
        "status": "success",
        "request_id": request_id,
        "worker_version": WORKER_VERSION,
        "task": task,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_hash,
        "model": {
            "base": RUNTIME.base_model_id,
            "base_revision": RUNTIME.model_revision(),
            "adapter": RUNTIME.adapter_model_id,
            "adapter_revision": RUNTIME.adapter_revision,
        },
        "extraction": result.extraction,
        "document_text": result.document_text,
        "metrics": {
            # Kept for compatibility with existing job-result consumers.
            "pages": len(images),
            "image_views": len(images),
            "input_mode": (
                "ocr_text" if images and document_text
                else "ocr" if images
                else "text"
            ),
            "model_load_seconds": RUNTIME.load_seconds,
            "ocr_attempted": bool(images),
            "ocr_generation_seconds": result.ocr_generation_seconds,
            "structure_generation_seconds": result.structure_generation_seconds,
            "generation_seconds": (
                result.ocr_generation_seconds + result.structure_generation_seconds
            ),
            "elapsed_seconds": elapsed_seconds,
        },
    }
    if os.getenv("INCLUDE_RAW_MODEL_OUTPUT", "false").lower() == "true":
        response["raw_model_output"] = result.raw_model_output
    return response


# Keep the registration at module scope. RunPod's GitHub importer performs a
# lightweight repository scan and may not recognize registrations hidden
# behind a ``__main__`` guard, even though the Docker entry point would execute
# that guard correctly.
runpod.serverless.start({"handler": handler})
