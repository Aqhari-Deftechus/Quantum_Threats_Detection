from __future__ import annotations

from app.llm_guardrails import validate_response


def test_allows_valid_response() -> None:
    result = validate_response("R.A.I.S.Y.A", "Operator, status is green.")
    assert result.allowed


def test_blocks_pii() -> None:
    result = validate_response("R.A.I.S.Y.A", "User SSN 123-45-6789 detected.")
    assert not result.allowed
    assert result.reason == "pii_detected"


def test_adam_no_explanations() -> None:
    result = validate_response("A.D.A.M", "HOLD because policy.")
    assert not result.allowed


def test_maryam_no_recommendations() -> None:
    result = validate_response("M.A.R.Y.A.M", "Recommend evacuation.")
    assert not result.allowed
