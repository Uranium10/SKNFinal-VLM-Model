"""Validate prompts supplied by the BiddingFlow backend."""

from __future__ import annotations

import hashlib
import os
from typing import Any


def prompt_values(payload: dict[str, Any]) -> tuple[str, str, str, str]:
    system_prompt = str(payload.get("system_prompt") or "").strip()
    user_prompt = str(payload.get("user_prompt") or "").strip()
    prompt_version = str(payload.get("prompt_version") or "").strip()
    supplied_hash = str(payload.get("prompt_sha256") or "").strip().lower()
    if not system_prompt or not user_prompt:
        raise ValueError("system_prompt and user_prompt are required")
    if not prompt_version or not supplied_hash:
        raise ValueError("prompt_version and prompt_sha256 are required")

    max_prompt_chars = int(os.getenv("MAX_PROMPT_CHARS", "30000"))
    if len(system_prompt) + len(user_prompt) > max_prompt_chars:
        raise ValueError("prompt exceeds MAX_PROMPT_CHARS")

    calculated_hash = hashlib.sha256(
        (system_prompt + "\n" + user_prompt).encode("utf-8")
    ).hexdigest()
    if supplied_hash != calculated_hash:
        raise ValueError("prompt_sha256 does not match the supplied prompts")
    return system_prompt, user_prompt, prompt_version, calculated_hash
