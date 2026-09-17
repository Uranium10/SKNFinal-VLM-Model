from __future__ import annotations

import base64
import hashlib
import io
import sys
from pathlib import Path

from PIL import Image
import pytest


WORKER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKER_ROOT))

from document_input import (  # noqa: E402
    _detail_views,
    documents_to_images,
    merge_document_text,
    read_document_text,
    read_documents,
)
from model_runtime import (  # noqa: E402
    OCR_SYSTEM_PROMPT,
    OCR_USER_PROMPT,
    _local_adapter_source,
    extract_json_object,
)
from prompt_contract import prompt_values  # noqa: E402
from quotation_pipeline import run_quotation_pipeline  # noqa: E402
from schemas import validate_extraction  # noqa: E402


def _valid_payload() -> dict:
    return {
        "quotation_id": None,
        "supplier_name": "테스트 공급사",
        "business_registration_no": None,
        "quotation_date": "2026-09-14",
        "valid_until": None,
        "currency": "KRW",
        "subtotal": 1000,
        "tax_amount": 100,
        "total_amount": 1100,
        "items": [{
            "item_code": "ITEM-001",
            "item_name": "안전모",
            "description": None,
            "quantity": 1,
            "unit": "EA",
            "unit_price": 1000,
            "amount": 1000,
            "expected_delivery_date": None,
            "lead_time_days": None,
            "specifications": {},
            "raw_description": "안전모 1 EA",
        }],
        "notes": None,
    }


def test_schema_accepts_expected_contract() -> None:
    assert validate_extraction(_valid_payload())["total_amount"] == 1100


def test_schema_rejects_unexpected_fields() -> None:
    payload = _valid_payload()
    payload["unexpected"] = True
    with pytest.raises(Exception):
        validate_extraction(payload)


def test_schema_rejects_missing_items() -> None:
    payload = _valid_payload()
    del payload["items"]
    with pytest.raises(Exception):
        validate_extraction(payload)


def test_ocr_prompt_preserves_handwriting_tables_and_conflicts() -> None:
    assert "손글씨" in OCR_SYSTEM_PROMPT
    assert "Markdown 표" in OCR_USER_PROMPT
    assert "둘 다 보존" in OCR_USER_PROMPT


def test_library_and_ocr_text_are_kept_as_separate_evidence() -> None:
    merged = merge_document_text("이메일 본문", "유효기간: 2026-09-30")

    assert "[Python 라이브러리 추출 원문]\n이메일 본문" in merged
    assert "[LoRA OCR 원문]\n유효기간: 2026-09-30" in merged


def test_merge_document_text_does_not_fuzzy_deduplicate_conflicts() -> None:
    merged = merge_document_text(
        "유효기간: 2026-09-30",
        "유효기간: 2026-09-20",
    )

    assert "2026-09-30" in merged
    assert "2026-09-20" in merged


def test_visual_pipeline_runs_lora_ocr_before_base_structuring() -> None:
    calls = []

    class Runtime:
        def transcribe_document(self, images, max_new_tokens):
            calls.append(("ocr", len(images), max_new_tokens))
            return "유효기간: 2026-09-30", 1.25

        def structure_text(self, text, system_prompt, user_prompt, max_new_tokens):
            calls.append(("structure", text, max_new_tokens))
            return _valid_payload(), '{"quotation_id": null}', 0.75

    result = run_quotation_pipeline(
        Runtime(),
        [Image.new("RGB", (10, 20), "white")],
        "",
        "system",
        "user",
        512,
        2048,
    )

    assert calls[0] == ("ocr", 1, 2048)
    assert calls[1][0] == "structure"
    assert "[LoRA OCR 원문]" in calls[1][1]
    assert result.ocr_generation_seconds == 1.25
    assert result.structure_generation_seconds == 0.75


def test_library_text_pipeline_skips_visual_ocr() -> None:
    class Runtime:
        def transcribe_document(self, *_args):
            raise AssertionError("text documents must not use visual OCR")

        def structure_text(self, text, *_args):
            assert "[Python 라이브러리 추출 원문]" in text
            return _valid_payload(), "{}", 0.5

    result = run_quotation_pipeline(
        Runtime(), [], "품목명: 안전모", "system", "user", 512, 2048,
    )

    assert result.ocr_generation_seconds == 0.0


def test_json_parser_tolerates_markdown_fence() -> None:
    assert extract_json_object('```json\n{"total": 10}\n```') == {"total": 10}


def test_adapter_uses_explicit_local_directory(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    assert _local_adapter_source("owner/adapter", "revision", str(adapter)) == str(adapter)


def test_missing_explicit_adapter_never_falls_back_to_hub(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        _local_adapter_source(
            "owner/adapter",
            "revision",
            str(tmp_path / "missing-adapter"),
        )


def test_prompt_contract_requires_matching_hash() -> None:
    system_prompt = "system"
    user_prompt = "user"
    digest = hashlib.sha256(f"{system_prompt}\n{user_prompt}".encode()).hexdigest()
    values = prompt_values({
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt_version": "test-v1",
        "prompt_sha256": digest,
    })
    assert values[2:] == ("test-v1", digest)

    with pytest.raises(ValueError, match="prompt_sha256"):
        prompt_values({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "prompt_version": "test-v1",
            "prompt_sha256": "0" * 64,
        })


def test_base64_image_input_round_trip() -> None:
    image = Image.new("RGB", (2, 2), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    documents = read_documents({"image_base64": encoded, "filename": "test.png"})
    images = documents_to_images(documents)
    try:
        assert len(images) == 1
        assert images[0].mode == "RGB"
    finally:
        for decoded in images:
            decoded.close()


def test_text_only_input_does_not_require_documents() -> None:
    payload = {"document_text": "품목명 | 수량\n테스트 품목 | 1", "documents": []}

    assert read_documents(payload) == []
    assert documents_to_images([]) == []
    assert read_document_text(payload).startswith("품목명")


def test_tall_quotation_image_gets_top_and_bottom_detail_views() -> None:
    image = Image.new("RGB", (1000, 1600), "white")

    views = _detail_views(image)

    assert len(views) == 2
    assert views[0].size == (1000, 928)
    assert views[1].size == (1000, 800)


def test_landscape_image_does_not_create_redundant_detail_views() -> None:
    image = Image.new("RGB", (1600, 1000), "white")

    assert _detail_views(image) == []


def test_document_text_limit(monkeypatch) -> None:
    monkeypatch.setenv("MAX_DOCUMENT_TEXT_CHARS", "5")

    with pytest.raises(ValueError, match="MAX_DOCUMENT_TEXT_CHARS"):
        read_document_text({"document_text": "123456"})
