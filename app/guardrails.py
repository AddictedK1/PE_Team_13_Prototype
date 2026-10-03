"""Application-level guardrails for candidate screening.

Protects against:
1. Empty input
2. Off-topic input (e.g. 'Write a poem about cats')
3. Harmful/biased screening instructions (e.g. 'Reject all women', 'Prefer IIT students')
4. LLM invalid-output protection (handled in screen.py)
5. API error protection (handled in screen.py and llm_client.py)
"""

from __future__ import annotations

import re
from typing import List, Optional
from pydantic import BaseModel, Field


class GuardrailCheckResult(BaseModel):
    """Result of a guardrail validation check."""

    passed: bool = Field(..., description="Whether the input passed the guardrail check")
    guardrail_type: Optional[str] = Field(default=None, description="Identifier of the guardrail")
    error_message: Optional[str] = Field(default=None, description="Human-readable reason for failure")


# Discriminatory / biased patterns that should never be accepted in hiring criteria or resume input
HARMFUL_BIAS_PATTERNS = [
    # Gender discrimination patterns
    r"\b(?:reject|exclude|disqualify|drop)\s+(?:all\s+)?(?:women|female|females|men|male|males)\b",
    r"\b(?:only\s+shortlist|only\s+hire|prefer)\s+(?:male|males|men|female|females|women)\b",
    r"\bprefer\s+(?:male|female)\s+candidates\b",
    # Institutional elitism / college favoritism
    r"\b(?:prefer|only\s+hire|only\s+shortlist)\s+(?:iit|nit|ivy\s*league|tier[\s-]*1)\s+(?:students|graduates|candidates)?\b",
    r"\b(?:reject|exclude)\s+(?:tier[\s-]*[23]|non-iit|state\s+college)\b",
    # Racial / demographic / caste / age discrimination
    r"\b(?:only\s+hire|reject)\s+(?:white|black|asian|hispanic|indian|dalit|brahmin)\b",
    r"\b(?:reject|exclude)\s+(?:candidates\s+over|older|pregnant)\b",
]

# Off-topic intent keywords / queries
OFF_TOPIC_PATTERNS = [
    r"^(?:write\s+a\s+(?:poem|story|song|essay|joke)|tell\s+me\s+a\s+joke|how\s+to\s+make\s+a\s+cake|recipe\s+for)",
    r"^(?:translate\s+(?:this|to)|what\s+is\s+the\s+capital\s+of|who\s+won\s+the|solve\s+this\s+math)",
    r"^(?:ignore\s+(?:all\s+)?previous\s+instructions|system\s+prompt|dan\s+mode)",
]

# Standard resume indicator tokens (at least a couple should appear in genuine resume text)
RESUME_INDICATOR_KEYWORDS = [
    "experience", "education", "skills", "projects", "work", "university",
    "college", "bachelor", "master", "degree", "developer", "engineer",
    "technologies", "certifications", "intern", "coursework", "gpa",
    "responsibilities", "programming", "python", "java", "c++", "software",
    "technical", "frontend", "backend", "fullstack", "github", "linkedin",
    "graduated", "proficient", "frameworks", "tools", "database", "api",
]


def check_empty_input(text: Optional[str], field_name: str = "Input") -> GuardrailCheckResult:
    """Validate that the input is not empty or trivially short."""
    if not text or not text.strip():
        return GuardrailCheckResult(
            passed=False,
            guardrail_type="empty_input",
            error_message=f"{field_name} cannot be empty.",
        )
    if len(text.strip()) < 15:
        return GuardrailCheckResult(
            passed=False,
            guardrail_type="empty_input",
            error_message=f"{field_name} is too short to be evaluated (minimum 15 characters required).",
        )
    return GuardrailCheckResult(passed=True)


def check_harmful_bias_instructions(text: str, context_label: str = "Input") -> GuardrailCheckResult:
    """Detect discriminatory or illegal screening instructions targeting protected attributes."""
    cleaned = text.lower()
    for pattern in HARMFUL_BIAS_PATTERNS:
        match = re.search(pattern, cleaned)
        if match:
            matched_phrase = match.group(0)
            return GuardrailCheckResult(
                passed=False,
                guardrail_type="harmful_bias_instruction",
                error_message=(
                    f"Harmful screening instruction detected in {context_label}: "
                    f"'{matched_phrase}'. Discriminating based on gender, race, "
                    f"or institutional favoritism violates fairness guardrails."
                ),
            )
    return GuardrailCheckResult(passed=True)


def check_off_topic_input(text: str) -> GuardrailCheckResult:
    """Detect off-topic text that is clearly not a resume or CV."""
    cleaned = text.strip().lower()

    # 1. Check explicit off-topic query patterns
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, cleaned):
            return GuardrailCheckResult(
                passed=False,
                guardrail_type="off_topic",
                error_message="Off-topic input detected: Input appears to be an unrelated query or prompt injection rather than a candidate resume.",
            )

    # 2. Check for presence of genuine resume indicators
    # Tokenize words
    words = set(re.findall(r"\b[a-z]{3,}\b", cleaned))
    matches = [kw for kw in RESUME_INDICATOR_KEYWORDS if kw in words]

    # If the input has zero resume keywords and is relatively short, flag as off-topic
    if len(matches) == 0 and len(cleaned) < 500:
        return GuardrailCheckResult(
            passed=False,
            guardrail_type="off_topic",
            error_message="Off-topic input: The submitted text does not contain typical resume sections, technical skills, or work experience.",
        )

    return GuardrailCheckResult(passed=True)


def validate_screening_guardrails(resume: str, job_description: Optional[str] = None) -> GuardrailCheckResult:
    """Run all pre-screening guardrails on the candidate input and job description."""
    # 1. Resume empty check
    empty_resume = check_empty_input(resume, field_name="Candidate resume")
    if not empty_resume.passed:
        return empty_resume

    # 2. Harmful bias instruction check on resume text
    harmful_resume = check_harmful_bias_instructions(resume, context_label="Candidate resume")
    if not harmful_resume.passed:
        return harmful_resume

    # 3. Harmful bias instruction check on job description (if provided)
    if job_description:
        harmful_jd = check_harmful_bias_instructions(job_description, context_label="Job description")
        if not harmful_jd.passed:
            return harmful_jd

    # 4. Off-topic check on resume
    off_topic = check_off_topic_input(resume)
    if not off_topic.passed:
        return off_topic

    return GuardrailCheckResult(passed=True)
