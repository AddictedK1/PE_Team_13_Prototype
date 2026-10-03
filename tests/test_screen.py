"""Tests for screen() execution, error handling, and retry mechanics."""

import pytest
from app.llm_client import LLMAPIError, MockLLMClient
from app.screen import screen


def test_screen_valid_output_shortlist():
    valid_json = '{"decision": "shortlist", "score": 88, "reason": "Demonstrates strong Python & data structures."}'
    mock_client = MockLLMClient(responses=[valid_json])

    res = screen(
        resume="Python engineer with 3 years backend experience.",
        prompt_version="v1",
        client=mock_client,
    )

    assert res.valid is True
    assert res.decision == "shortlist"
    assert res.score == 88
    assert res.retried is False
    assert mock_client.call_count == 1


def test_screen_valid_output_reject():
    valid_json = '{"decision": "reject", "score": 35, "reason": "No programming languages or relevant experience."}'
    mock_client = MockLLMClient(responses=[valid_json])

    res = screen(
        resume="Accountant with experience in auditing spreadsheets.",
        prompt_version="v2",
        client=mock_client,
    )

    assert res.valid is True
    assert res.decision == "reject"
    assert res.score == 35
    assert res.retried is False


def test_screen_retry_on_invalid_json_then_success():
    invalid_raw = "I think this candidate is great, 85 out of 100!"
    valid_raw = '{"decision": "shortlist", "score": 85, "reason": "Strong coding fundamentals."}'
    mock_client = MockLLMClient(responses=[invalid_raw, valid_raw])

    res = screen(
        resume="Java developer resume.",
        prompt_version="v1",
        client=mock_client,
        max_retries=1,
    )

    assert res.valid is True
    assert res.decision == "shortlist"
    assert res.score == 85
    assert res.retried is True
    assert mock_client.call_count == 2
    # Ensure retry prompt included correction instruction
    assert "CRITICAL CORRECTION REQUIRED" in mock_client.recorded_prompts[1]


def test_screen_retry_on_score_out_of_range_then_success():
    invalid_score = '{"decision": "shortlist", "score": 150, "reason": "Off the charts."}'
    valid_score = '{"decision": "shortlist", "score": 95, "reason": "Normalized score."}'
    mock_client = MockLLMClient(responses=[invalid_score, valid_score])

    res = screen(
        resume="SDE resume.",
        prompt_version="v2",
        client=mock_client,
        max_retries=1,
    )

    assert res.valid is True
    assert res.score == 95
    assert res.retried is True
    assert mock_client.call_count == 2


def test_screen_retry_failure_returns_invalid_object():
    invalid_1 = "Not json"
    invalid_2 = "Still not json"
    mock_client = MockLLMClient(responses=[invalid_1, invalid_2])

    res = screen(
        resume="Candidate resume.",
        prompt_version="v1",
        client=mock_client,
        max_retries=1,
    )

    # CRITICAL: System failure MUST NOT become decision='reject'
    assert res.valid is False
    assert res.decision is None
    assert res.score is None
    assert "LLM returned invalid output after retry" in res.reason
    assert res.retried is True
    assert res.error == "validation_failure"


def test_screen_api_error_handling():
    error_client = MockLLMClient(raise_error=LLMAPIError("API rate limit exceeded", status_code=429))

    res = screen(
        resume="Candidate resume.",
        prompt_version="v1",
        client=error_client,
    )

    assert res.valid is False
    assert res.decision is None
    assert res.score is None
    assert "rate limit exceeded" in res.reason
    assert res.error == "api_error"


def test_screen_v1_and_v2_prompt_passed():
    mock_client_v1 = MockLLMClient()
    screen("Resume 1", prompt_version="v1", client=mock_client_v1)
    assert "JOB DESCRIPTION:" in mock_client_v1.recorded_prompts[0]

    mock_client_v2 = MockLLMClient()
    screen("Resume 2", prompt_version="v2", client=mock_client_v2)
    assert "ATTRIBUTE BLINDNESS" in mock_client_v2.recorded_prompts[0]
