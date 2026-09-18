"""Two-stage quotation extraction: LoRA OCR, then base-model structuring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from document_input import merge_document_text
from schemas import validate_extraction


@dataclass(frozen=True)
class PipelineResult:
    extraction: dict[str, Any]
    document_text: str
    raw_model_output: str
    ocr_generation_seconds: float
    structure_generation_seconds: float


def run_quotation_pipeline(
    runtime: Any,
    images: list[Image.Image],
    library_text: str,
    system_prompt: str,
    user_prompt: str,
    structure_max_new_tokens: int,
    ocr_max_new_tokens: int,
    task: str = "extraction",
) -> PipelineResult:
    """Run OCR only for visual inputs and always structure normalized text.

    ``task`` controls how the structuring stage's JSON is post-processed:
    - "extraction" (default, unchanged behavior): validated against the
      strict QuotationExtraction schema, since downstream ERP mapping
      depends on that stable contract.
    - anything else (e.g. "spec_eval"): returned as-is, with no schema
      imposed here. This lets the same worker/base-model be reused for
      other structured-JSON tasks (via system_prompt/user_prompt) without
      forcing their output into the extraction shape. The caller is
      responsible for validating the shape it asked for.
    """

    ocr_text = ""
    ocr_seconds = 0.0
    if images:
        ocr_text, ocr_seconds = runtime.transcribe_document(
            images,
            ocr_max_new_tokens,
        )

    document_text = merge_document_text(library_text, ocr_text)
    if not document_text:
        raise ValueError("document transcription produced no text")

    extracted, raw_text, structure_seconds = runtime.structure_text(
        document_text,
        system_prompt,
        user_prompt,
        structure_max_new_tokens,
    )
    result_payload = (
        validate_extraction(extracted) if task == "extraction" else extracted
    )
    return PipelineResult(
        extraction=result_payload,
        document_text=document_text,
        raw_model_output=raw_text,
        ocr_generation_seconds=ocr_seconds,
        structure_generation_seconds=structure_seconds,
    )
