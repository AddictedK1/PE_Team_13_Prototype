"""Comprehensive screening validation tests explicitly covering all 13 required acceptance criteria:

1. Valid LLM JSON
2. Invalid JSON
3. Missing score
4. Score > 100
5. Invalid decision
6. Empty reason
7. Retry after invalid output
8. Retry failure
9. Empty resume
10. Off-topic input
11. Valid resume
12. v1 prompt selection
13. v2 prompt selection
"""

import pytest
from pydantic import ValidationError
from app.guardrails import validate_screening_guardrails
from app.llm_client import MockLLMClient
from app.prompts import get_prompt
from app.schema import parse_and_validate_llm_output
from app.screen import screen


# 1. Valid LLM JSON
def test_1_valid_llm_json():
    raw = '{"decision": "shortlist", "score": 85, "reason": "Candidate has solid systems experience."}'
    evaluation = parse_and_validate_llm_output(raw)
    assert evaluation.decision == "shortlist"
    assert evaluation.score == 85
    assert evaluation.reason == "Candidate has solid systems experience."


# 2. Invalid JSON
def test_2_invalid_json():
    raw = "Definitely shortlist this candidate, score 90."
    with pytest.raises(ValueError, match="Could not extract a valid JSON object"):
        parse_and_validate_llm_output(raw)


# 3. Missing score
def test_3_missing_score():
    raw = '{"decision": "shortlist", "reason": "Good qualifications."}'
    with pytest.raises(ValidationError):
        parse_and_validate_llm_output(raw)


# 4. Score > 100
def test_4_score_greater_than_100():
    raw = '{"decision": "shortlist", "score": 101, "reason": "Exceptional fit."}'
    with pytest.raises(ValidationError):
        parse_and_validate_llm_output(raw)


# 5. Invalid decision
def test_5_invalid_decision():
    raw = '{"decision": "maybe", "score": 75, "reason": "Borderline profile."}'
    with pytest.raises(ValidationError):
        parse_and_validate_llm_output(raw)


# 6. Empty reason
def test_6_empty_reason():
    raw = '{"decision": "reject", "score": 30, "reason": "   "}'
    with pytest.raises(ValidationError):
        parse_and_validate_llm_output(raw)


# 7. Retry after invalid output
def test_7_retry_after_invalid_output():
    invalid_first = "Candidate looks promising with score 80."
    valid_retry = '{"decision": "shortlist", "score": 80, "reason": "Strong Python knowledge."}'
    client = MockLLMClient(responses=[invalid_first, valid_retry])

    result = screen(
        resume="Experienced Python Developer with FastAPI and Postgres skills.",
        prompt_version="v1",
        client=client,
        max_retries=1,
    )

    assert result.valid is True
    assert result.retried is True
    assert result.decision == "shortlist"
    assert result.score == 80
    assert client.call_count == 2


# 8. Retry failure
def test_8_retry_failure():
    invalid_first = "Malformed response 1"
    invalid_second = "Malformed response 2"
    client = MockLLMClient(responses=[invalid_first, invalid_second])

    result = screen(
        resume="Experienced Python Developer with FastAPI and Postgres skills.",
        prompt_version="v1",
        client=client,
        max_retries=1,
    )

    # CRITICAL: Failed retry MUST NOT turn into decision="reject"
    assert result.valid is False
    assert result.decision is None
    assert result.score is None
    assert result.retried is True
    assert result.error == "validation_failure"
    assert "LLM returned invalid output after retry" in result.reason


# 9. Empty resume
def test_9_empty_resume():
    client = MockLLMClient()
    result = screen(
        resume="   \n\t ",
        prompt_version="v1",
        client=client,
    )
    assert result.valid is False
    assert result.decision is None
    assert result.error == "empty_input"
    assert client.call_count == 0  # Blocked before LLM invocation


# 10. Off-topic input
def test_10_off_topic_input():
    client = MockLLMClient()
    result = screen(
        resume="Write a poem about cats playing in the rain.",
        prompt_version="v1",
        client=client,
    )
    assert result.valid is False
    assert result.decision is None
    assert result.error == "off_topic"
    assert client.call_count == 0  # Blocked before LLM invocation


# 11. Valid resume
def test_11_valid_resume():
    valid_resume = (
        "Alice Smith\n"
        "Education: B.S. in Computer Science, State University\n"
        "Skills: Python, Django, REST APIs, PostgreSQL, Docker\n"
        "Experience: Software Engineer at Acme Corp (2022-2024)\n"
        "Projects: Built microservice scaling to 10k RPS."
    )
    check = validate_screening_guardrails(valid_resume)
    assert check.passed is True

    client = MockLLMClient(
        responses=['{"decision": "shortlist", "score": 88, "reason": "Matches all required engineering criteria."}']
    )
    result = screen(resume=valid_resume, prompt_version="v1", client=client)
    assert result.valid is True
    assert result.decision == "shortlist"
    assert result.score == 88


# 12. v1 prompt selection
def test_12_v1_prompt_selection():
    client = MockLLMClient()
    screen("Python developer resume with fullstack experience.", prompt_version="v1", client=client)
    prompt_sent = client.recorded_prompts[0]
    assert "JOB DESCRIPTION:" in prompt_sent
    assert "CANDIDATE RESUME:" in prompt_sent
    assert "shortlist" in prompt_sent


# 13. v2 prompt selection
def test_13_v2_prompt_selection():
    client = MockLLMClient()
    screen("Python developer resume with fullstack experience.", prompt_version="v2", client=client)
    prompt_sent = client.recorded_prompts[0]
    assert "ATTRIBUTE BLINDNESS" in prompt_sent
    assert "RUBRIC-BASED SCORING" in prompt_sent
    assert "BIAS SELF-CHECK & SANITY AUDIT" in prompt_sent
