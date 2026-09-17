"""Download and decode image/PDF inputs without exposing local file access."""

from __future__ import annotations

import base64
import binascii
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image


@dataclass(frozen=True)
class DocumentBytes:
    data: bytes
    filename: str


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _allowed_hosts() -> set[str]:
    return {
        value.strip().lower()
        for value in os.getenv("ALLOWED_DOCUMENT_HOSTS", "").split(",")
        if value.strip()
    }


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    allow_http = os.getenv("ALLOW_HTTP_DOCUMENT_URLS", "false").lower() == "true"
    allowed_schemes = {"https", "http"} if allow_http else {"https"}
    if parsed.scheme.lower() not in allowed_schemes or not parsed.hostname:
        raise ValueError("document URL must use an allowed HTTP scheme and host")

    allowed_hosts = _allowed_hosts()
    if allowed_hosts and parsed.hostname.lower() not in allowed_hosts:
        raise ValueError("document URL host is not allowlisted")


def _download(url: str, filename: str | None) -> DocumentBytes:
    _validate_url(url)
    max_bytes = _positive_int_env("MAX_DOCUMENT_BYTES", 6 * 1024 * 1024)
    with requests.get(
        url,
        stream=True,
        timeout=(5, 60),
        allow_redirects=True,
        headers={"User-Agent": "BiddingFlow-RunPod-Quotation/1.0"},
    ) as response:
        response.raise_for_status()
        # Redirects are convenient for signed object-storage URLs, but the final
        # host must satisfy the same allowlist as the original URL.
        _validate_url(response.url)
        content_length = int(response.headers.get("content-length") or 0)
        if content_length > max_bytes:
            raise ValueError("document exceeds MAX_DOCUMENT_BYTES")

        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            size += len(chunk)
            if size > max_bytes:
                raise ValueError("document exceeds MAX_DOCUMENT_BYTES")
            chunks.append(chunk)

        inferred_name = Path(urlparse(response.url).path).name or "quotation.bin"
        return DocumentBytes(b"".join(chunks), filename or inferred_name)


def _decode_base64(value: str, filename: str | None) -> DocumentBytes:
    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("document_base64 is not valid base64") from exc

    max_bytes = _positive_int_env("MAX_DOCUMENT_BYTES", 6 * 1024 * 1024)
    if len(data) > max_bytes:
        raise ValueError("document exceeds MAX_DOCUMENT_BYTES")
    return DocumentBytes(data, filename or "quotation.png")


def _one_document(spec: dict[str, Any]) -> DocumentBytes:
    url = str(spec.get("url") or spec.get("document_url") or "").strip()
    encoded = str(spec.get("base64") or spec.get("document_base64") or "").strip()
    filename = str(spec.get("filename") or "").strip() or None
    if bool(url) == bool(encoded):
        raise ValueError("each document requires exactly one of url or base64")
    return _download(url, filename) if url else _decode_base64(encoded, filename)


def read_document_text(payload: dict[str, Any]) -> str:
    """Read normalized text produced by the backend's safe document parsers."""

    value = str(payload.get("document_text") or "").strip()
    max_chars = _positive_int_env("MAX_DOCUMENT_TEXT_CHARS", 60_000)
    if len(value) > max_chars:
        raise ValueError("document_text exceeds MAX_DOCUMENT_TEXT_CHARS")
    return value


def read_documents(payload: dict[str, Any]) -> list[DocumentBytes]:
    """Accept optional documents and a small set of legacy single-document keys."""

    specs = payload.get("documents")
    if specs is None:
        legacy_url = payload.get("document_url") or payload.get("image_url")
        legacy_base64 = payload.get("document_base64") or payload.get("image_base64")
        specs = ([{
            "url": legacy_url,
            "base64": legacy_base64,
            "filename": payload.get("filename"),
        }] if legacy_url or legacy_base64 else [])
    if not isinstance(specs, list):
        raise ValueError("documents must be a list")
    if len(specs) > _positive_int_env("MAX_DOCUMENTS", 8):
        raise ValueError("too many documents")
    if not all(isinstance(spec, dict) for spec in specs):
        raise ValueError("each documents entry must be an object")
    return [_one_document(spec) for spec in specs]


def _is_pdf(document: DocumentBytes) -> bool:
    return document.data.startswith(b"%PDF-") or Path(document.filename).suffix.lower() == ".pdf"


def _detail_views(image: Image.Image) -> list[Image.Image]:
    """Return top/bottom detail views for tall quotation pages.

    A full A4 page is resized to the processor's per-image pixel budget. Small
    validity and terms text is therefore easy for a VLM to miss even though
    item rows remain readable. Supplying two overlapping crops preserves the
    full-page context while giving those regions enough effective resolution.
    """

    width, height = image.size
    if height < int(width * 1.15) or min(width, height) < 600:
        return []

    overlap = 0.08
    split = 0.58
    top_end = max(1, min(height, round(height * split)))
    bottom_start = max(0, min(height - 1, round(height * (split - overlap))))
    return [
        image.crop((0, 0, width, top_end)),
        image.crop((0, bottom_start, width, height)),
    ]


def _append_page_with_details(
    images: list[Image.Image],
    page_image: Image.Image,
    *,
    include_details: bool = True,
) -> None:
    """Append a detached overview followed by optional detached detail crops."""

    overview = page_image.convert("RGB").copy()
    images.append(overview)
    if include_details:
        images.extend(_detail_views(overview))


def documents_to_images(documents: list[DocumentBytes]) -> list[Image.Image]:
    """Convert supported documents into detached RGB PIL images."""

    images: list[Image.Image] = []
    max_pages = _positive_int_env("MAX_PDF_PAGES", 8)
    detail_pages_remaining = _positive_int_env("MAX_DETAIL_CROP_PAGES", 2)
    for document in documents:
        if _is_pdf(document):
            try:
                import fitz
            except ImportError as exc:  # pragma: no cover - container dependency
                raise RuntimeError("PyMuPDF is required for PDF input") from exc

            pdf = fitz.open(stream=document.data, filetype="pdf")
            try:
                if len(pdf) > max_pages:
                    raise ValueError("PDF exceeds MAX_PDF_PAGES")
                for page in pdf:
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    with Image.open(io.BytesIO(pixmap.tobytes("png"))) as page_image:
                        _append_page_with_details(
                            images,
                            page_image,
                            include_details=detail_pages_remaining > 0,
                        )
                        detail_pages_remaining -= 1
            finally:
                pdf.close()
            continue

        try:
            with Image.open(io.BytesIO(document.data)) as source_image:
                _append_page_with_details(
                    images,
                    source_image,
                    include_details=detail_pages_remaining > 0,
                )
                detail_pages_remaining -= 1
        except Exception as exc:
            raise ValueError(f"unsupported image document: {document.filename}") from exc

    return images
