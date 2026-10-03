"""Tests for schema validation of LLM candidate screening outputs."""

import pytest
from pydantic import ValidationError
from app.schema import (
    CandidateEvaluation,
    ScreeningResult,
    extract_json_from_text,
    parse_and_validate_llm_output,
)


def test_valid_shortlist_json():
    raw = '{"decision": "shortlist", "score": 85, "reason": "Strong coding skills and relevant experience."}'
    evaluation = parse_and_validate_llm_output(raw)
    assert evaluation.decision == "shortlist"
    assert evaluation.score == 85
    assert "Strong coding" in evaluation.reason


def test_valid_reject_json():
    raw = '{"decision": "reject", "score": 42, "reason": "Missing required software engineering fundamentals."}'
    evaluation = parse_and_validate_llm_output(raw)
    assert evaluation.decision == "reject"
    assert evaluation.score == 42
    assert "Missing" in evaluation.reason


def test_markdown_code_fence_json():
    raw = """Here is the screening result:
```json
{
    "decision": "shortlist",
    "score": 90,
    "reason": "Outstanding open-source contributions."
}
```
Hope this helps!"""
    evaluation = parse_and_validate_llm_output(raw)
    assert evaluation.decision == "shortlist"
    assert evaluation.score == 90


def test_invalid_json_syntax():
    raw = "Not a json string at all."
    with pytest.raises(ValueError, match="Could not extract a valid JSON object"):
        parse_and_validate_llm_output(raw)


def test_missing_score():
    raw = '{"decision": "shortlist", "reason": "Great candidate."}'
    with pytest.raises(ValidationError):
        parse_and_validate_llm_output(raw)


def test_score_above_100():
    raw = '{"decision": "shortlist", "score": 105, "reason": "Over-qualified candidate."}'
    with pytest.raises(ValidationError):
        parse_and_validate_llm_output(raw)


def test_score_below_zero():
    raw = '{"decision": "reject", "score": -5, "reason": "Bad performance."}'
    with pytest.raises(ValidationError):
        parse_and_validate_llm_output(raw)


def test_invalid_decision():
    raw = '{"decision": "hire", "score": 90, "reason": "Great fit."}'
    with pytest.raises(ValidationError):
        parse_and_validate_llm_output(raw)


def test_empty_reason():
    raw = '{"decision": "shortlist", "score": 80, "reason": "   "}'
    with pytest.raises(ValidationError):
        parse_and_validate_llm_output(raw)


def test_screening_result_genuine_reject_vs_failure():
    # Genuine rejection
    reject_res = ScreeningResult(
        valid=True,
        decision="reject",
        score=35,
        reason="Does not meet minimum requirements.",
        prompt_version="v1",
    )
    eval_dict = reject_res.to_evaluation_dict(pair_id=1, candidate_id="1A")
    assert eval_dict["valid"] is True
    assert eval_dict["decision"] == "reject"
    assert eval_dict["score"] == 35

    # System failure
    fail_res = ScreeningResult(
        valid=False,
        decision=None,
        score=None,
        reason="LLM returned invalid output after retry.",
        error="validation_failure",
        prompt_version="v1",
    )
    eval_fail = fail_res.to_evaluation_dict(pair_id=1, candidate_id="1A")
    assert eval_fail["valid"] is False
    assert eval_fail["decision"] is None
    assert eval_fail["score"] is None
    assert "invalid output" in eval_fail["reason"]
