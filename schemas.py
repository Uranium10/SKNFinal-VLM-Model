"""Strict response contract shared by the worker and its tests.

The model is allowed to say that a value is unknown by returning ``null``.
It is not allowed to invent new keys because downstream ERP mapping relies on
this stable contract.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QuotationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_code: str | None
    item_name: str | None
    description: str | None
    quantity: float | int | None
    unit: str | None
    unit_price: float | int | None
    amount: float | int | None
    expected_delivery_date: str | None
    lead_time_days: int | None
    specifications: dict[str, Any]
    raw_description: str | None


class QuotationExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quotation_id: str | None
    supplier_name: str | None
    business_registration_no: str | None
    quotation_date: str | None
    valid_until: str | None
    currency: str | None
    subtotal: float | int | None
    tax_amount: float | int | None
    total_amount: float | int | None
    items: list[QuotationItem]
    notes: str | None


def validate_extraction(value: dict[str, Any]) -> dict[str, Any]:
    """Validate and convert a model response into JSON-safe primitives."""

    return QuotationExtraction.model_validate(value).model_dump(mode="json")
