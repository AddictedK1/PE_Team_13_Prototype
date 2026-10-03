"""Tests for prompt selection and template construction."""

import pytest
from app.prompts import (
    DEFAULT_JOB_DESCRIPTION,
    PROMPTS,
    load_prompt_template,
    get_prompt,
    get_retry_prompt,
)


def test_v1_prompt_selection():
    resume = "Candidate Resume: Python, Django, PostgreSQL, 2 years exp."
    prompt = get_prompt("v1", resume)
    assert "CANDIDATE RESUME:" in prompt
    assert "Python, Django" in prompt
    assert "JOB DESCRIPTION:" in prompt
    assert "shortlist" in prompt


def test_v2_prompt_selection():
    resume = "Candidate Resume: Java, Spring Boot, AWS, Distributed Systems."
    prompt = get_prompt("v2", resume)
    assert "ATTRIBUTE BLINDNESS" in prompt
    assert "RUBRIC-BASED SCORING" in prompt
    assert "Java, Spring Boot" in prompt


def test_prompt_selection_case_insensitive():
    resume = "Sample Resume"
    p1 = get_prompt("V1", resume)
    p2 = get_prompt("v1", resume)
    assert p1 == p2


def test_unknown_prompt_version_raises():
    with pytest.raises(ValueError, match="Unknown prompt version"):
        get_prompt("v3", "Some resume")


def test_custom_job_description():
    custom_jd = "Role: Machine Learning Engineer. PyTorch required."
    prompt = get_prompt("v1", "Resume text", job_description=custom_jd)
    assert "Machine Learning Engineer" in prompt
    assert "PyTorch required" in prompt


def test_custom_prompt_file_is_loaded():
    template = load_prompt_template()
    assert "The candidate's resume is provided as a PDF/document." in template
    assert "{{JOB_DESCRIPTION}}" in template
    assert "OUTPUT" in template


def test_custom_prompt_formats_context_without_interpreting_json_braces():
    resume = "Resume with Python and production API experience."
    job_description = "Backend engineer with Python experience."
    prompt = get_prompt("custom", resume, job_description=job_description)

    assert job_description in prompt
    assert resume in prompt
    assert '"ats_score": 0' in prompt


def test_retry_prompt_construction():
    orig = "Screen candidate prompt"
    err = "Score was outside 0-100"
    raw = '{"decision": "shortlist", "score": 150}'
    retry_p = get_retry_prompt(orig, raw, err)
    assert "CRITICAL CORRECTION REQUIRED" in retry_p
    assert err in retry_p
    assert raw in retry_p
