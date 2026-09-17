"""RunPod queue worker entry point for quotation extraction."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import runpod

from document_input import documents_to_images, read_document_text, read_documents
from model_runtime import QuotationModelRuntime
from prompt_contract import prompt_values
from schemas import validate_extraction


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("biddingflow.quotation_worker")
RUNTIME = QuotationModelRuntime()


def _max_new_tokens(payload: dict[str, Any]) -> int:
    default = int(os.getenv("MAX_NEW_TOKENS", "512"))
    cap = int(os.getenv("MAX_NEW_TOKENS_CAP", "1024"))
    requested = int(payload.get("max_new_tokens") or default)
    if requested <= 0 or requested > cap:
        raise ValueError(f"max_new_tokens must be between 1 and {cap}")
    return requested


def handler(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input")
    if not isinstance(payload, dict):
        raise ValueError("RunPod event must contain an input object")

    if payload.get("healthcheck") is True:
        return {
            "status": "ok",
            "worker": "biddingflow-quotation-extractor",
            "model_loaded": RUNTIME.loaded,
            "prompt_source": "request",
        }

    # Explicit paid preparation requested by the administrator's timed lease.
    # Do not alter the cheap build healthcheck or require a real quotation.
    if payload.get("warmup") is True:
        RUNTIME.load()
        return {
            "status": "ok", "worker": "biddingflow-quotation-extractor",
            "model_loaded": RUNTIME.loaded,
            "model_load_seconds": RUNTIME.load_seconds,
        }

    request_id = str(payload.get("request_id") or event.get("id") or "").strip()
    if not request_id:
        raise ValueError("request_id is required")

    system_prompt, user_prompt, prompt_version, prompt_hash = prompt_values(payload)
    document_text = read_document_text(payload)
    documents = read_documents(payload)
    if not documents and not document_text:
        raise ValueError("at least one document or document_text is required")
    images = documents_to_images(documents)
    started = time.perf_counter()
    try:
        extracted, raw_text, generation_seconds = RUNTIME.extract(
            images,
            document_text,
            system_prompt,
            user_prompt,
            _max_new_tokens(payload),
        )
        validated = validate_extraction(extracted)
    finally:
        for image in images:
            image.close()

    elapsed_seconds = time.perf_counter() - started
    LOGGER.info(
        "quotation extracted request_id=%s image_views=%d elapsed=%.3fs",
        request_id,
        len(images),
        elapsed_seconds,
    )
    response: dict[str, Any] = {
        "status": "success",
        "request_id": request_id,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_hash,
        "model": {
            "base": RUNTIME.base_model_id,
            "base_revision": RUNTIME.model_revision(),
            "adapter": RUNTIME.adapter_model_id,
            "adapter_revision": RUNTIME.adapter_revision,
        },
        "extraction": validated,
        "metrics": {
            # Kept for compatibility with existing job-result consumers.
            "pages": len(images),
            "image_views": len(images),
            "input_mode": (
                "hybrid" if images and document_text
                else "vision" if images
                else "text"
            ),
            "model_load_seconds": RUNTIME.load_seconds,
            "generation_seconds": generation_seconds,
            "elapsed_seconds": elapsed_seconds,
        },
    }
    if os.getenv("INCLUDE_RAW_MODEL_OUTPUT", "false").lower() == "true":
        response["raw_model_output"] = raw_text
    return response


# Keep the registration at module scope. RunPod's GitHub importer performs a
# lightweight repository scan and may not recognize registrations hidden
# behind a ``__main__`` guard, even though the Docker entry point would execute
# that guard correctly.
runpod.serverless.start({"handler": handler})
