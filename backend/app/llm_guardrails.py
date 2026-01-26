from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


ALLOWED_ROLES = {"M.A.R.Y.A.M", "A.D.A.M", "R.A.I.S.Y.A"}
BLOCK_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b\d{16}\b"),
    re.compile(r"\b\d{10}\b"),
]


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str
    sanitized: str


def validate_response(role: str, content: str) -> GuardrailResult:
    if role not in ALLOWED_ROLES:
        return GuardrailResult(False, "invalid_role", "blocked by guardrails")

    for pattern in BLOCK_PATTERNS:
        if pattern.search(content):
            return GuardrailResult(False, "pii_detected", "blocked by guardrails")

    if role == "A.D.A.M" and ("because" in content.lower() or "since" in content.lower()):
        return GuardrailResult(False, "adam_no_explanations", "blocked by guardrails")

    if role == "M.A.R.Y.A.M" and "recommend" in content.lower():
        return GuardrailResult(False, "maryam_no_recommendations", "blocked by guardrails")

    return GuardrailResult(True, "ok", content)


def stub_response(role: str, prompt: str) -> GuardrailResult:
    responses = {
        "M.A.R.Y.A.M": "Analysis complete. Confidence: 0.62. No recommendations issued.",
        "A.D.A.M": "HOLD",
        "R.A.I.S.Y.A": "Operator, the system is monitoring the area and logging observations.",
    }
    content = responses.get(role, "blocked by guardrails")
    return validate_response(role, content)
