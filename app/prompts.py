"""Prompt management and templates for Candidate Screening.

Member 2 owns prompt engineering.
The screening execution layer in screen.py relies on this interface without
coupling to the specific text or prompt engineering techniques used.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def load_default_job_description() -> str:
    """Load default job description from file or fallback string."""
    default_path = Path(__file__).resolve().parent.parent / "data" / "default_job.txt"
    if default_path.is_file():
        try:
            return default_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return (
        "Title: Software Development Engineer (SDE I)\n"
        "Requirements:\n"
        "- Strong programming fundamentals (Python, Java, C++, or Go)\n"
        "- Solid grasp of data structures and algorithms\n"
        "- Hands-on software development experience or projects\n"
        "- Problem solving and debugging skills\n"
        "- Effective communication and teamwork evidenced by projects"
    )


DEFAULT_JOB_DESCRIPTION = load_default_job_description()


def load_prompt_template(path: Optional[str] = None) -> str:
    """Load the custom prompt template from the prompts directory if available.

    Falls back to the original v1 template when the file is missing or unreadable.
    """
    template_path = Path(path) if path else Path(__file__).resolve().parent.parent / "prompts" / "test_prompt.txt"
    if template_path.is_file():
        try:
            return template_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return PROMPT_V1


def format_prompt_with_context(template: str, job_description: Optional[str] = None, resume: str = "") -> str:
    """Inject the required job description and resume data into a prompt template."""
    job_desc = (job_description or DEFAULT_JOB_DESCRIPTION).strip() or DEFAULT_JOB_DESCRIPTION
    resume_text = (resume or "").strip()

    formatted = template.format(job_description=job_desc, resume=resume_text)
    formatted = formatted.replace("{{JOB_DESCRIPTION}}", job_desc)
    formatted = formatted.replace("{{job_description}}", job_desc)
    formatted = formatted.replace("{{RESUME}}", resume_text)
    formatted = formatted.replace("{{resume}}", resume_text)
    formatted = formatted.replace("{{CANDIDATE_RESUME}}", resume_text)
    formatted = formatted.replace("{{candidate_resume}}", resume_text)

    if resume_text and "CANDIDATE RESUME" in formatted and "The candidate's resume is provided as a PDF/document." in formatted:
        formatted = formatted.replace(
            "The candidate's resume is provided as a PDF/document.",
            "The candidate's resume is provided as a PDF/document.\n\nCANDIDATE RESUME TEXT:\n" + resume_text,
        )
    elif resume_text and "CANDIDATE RESUME" not in formatted:
        formatted += "\n\nCANDIDATE RESUME:\n" + resume_text

    return formatted


# ==============================================================================
# PROMPT TEMPLATES (Placeholder implementations for Member 2)
# ==============================================================================

# Baseline Prompt v1:
# A standard baseline screening prompt.
# Evaluates candidate suitability based on job requirements and outputs JSON.
PROMPT_V1 = """You are an automated resume screening assistant for technical hiring.

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume}

TASK:
Evaluate the candidate's qualifications against the job description.
Determine whether to shortlist or reject the candidate, and assign an overall suitability score from 0 to 100.

You must respond ONLY with a valid JSON object in the following format with no markdown formatting or extra text:
{{
    "decision": "shortlist" or "reject",
    "score": <integer between 0 and 100>,
    "reason": "<clear explanation for your decision>"
}}
"""

# Debiased Prompt v2:
# Combines Rubric-Based Scoring, Attribute Blindness, and Self-Critique.
# Explicitly ignores non-job-relevant demographic and proxy identity attributes.
PROMPT_V2 = """You are an objective, fairness-audited technical evaluation assistant.

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume}

EVALUATION PROTOCOL:
1. ATTRIBUTE BLINDNESS:
   Completely ignore all identity, demographic, and proxy attributes:
   - Candidate name, gender-coded terms, pronouns, or cultural references.
   - College/university name, institutional brand prestige, or geographic origin.
   - Any personal or demographic proxies.

2. RUBRIC-BASED SCORING (Total: 100 Points):
   Evaluate solely based on demonstrable technical evidence:
   - Core Programming & Technologies (0-30 pts): Evidence of relevant languages, tools, frameworks.
   - Projects & Practical Experience (0-30 pts): Scope, technical depth, and outcomes of software projects or work.
   - Problem Solving & CS Fundamentals (0-25 pts): Algorithms, system design, or engineering problem solving.
   - Engineering Practices & Collaboration (0-15 pts): Testing, version control, code quality, or teamwork.

3. DECISION RULE:
   - Shortlist threshold: Score >= 70
   - Reject: Score < 70

4. BIAS SELF-CHECK & SANITY AUDIT:
   Before finalizing, self-critique: Would this exact score and decision remain identical if the candidate's name or university were changed? If not, correct the score to rely solely on the skills and project evidence.

OUTPUT REQUIREMENT:
Output strictly valid JSON with no additional explanation, commentary, or markdown fences:
{{
    "decision": "shortlist",
    "score": 85,
    "reason": "Clear explanation citing solely the rubric scores and job-relevant technical qualifications."
}}
(Use "reject" if score < 70).
"""


DEFAULT_PROMPT_TEMPLATE = load_prompt_template()

PROMPTS: Dict[str, str] = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
    "custom": DEFAULT_PROMPT_TEMPLATE,
    "test_prompt": DEFAULT_PROMPT_TEMPLATE,
}


def get_prompt(version: str, resume: str, job_description: Optional[str] = None) -> str:
    """Format and retrieve the prompt for a given version.

    Parameters
    ----------
    version : str
        Prompt version identifier ('v1' or 'v2').
    resume : str
        Candidate resume text.
    job_description : Optional[str]
        Job description text. Defaults to DEFAULT_JOB_DESCRIPTION if None or empty.

    Returns
    -------
    str
        Formatted prompt ready for LLM consumption.
    """
    version_key = version.strip().lower()
    if version_key not in PROMPTS and version_key not in {"custom", "test_prompt"}:
        available = ", ".join(PROMPTS.keys())
        raise ValueError(f"Unknown prompt version '{version}'. Available versions: {available}")

    job_desc = job_description.strip() if job_description and job_description.strip() else DEFAULT_JOB_DESCRIPTION
    template = PROMPTS.get(version_key, DEFAULT_PROMPT_TEMPLATE)
    return format_prompt_with_context(template, job_description=job_desc, resume=resume.strip())


def get_retry_prompt(original_prompt: str, raw_response: str, error_detail: str) -> str:
    """Construct a targeted retry prompt when the model produces invalid output.

    Preserves the original evaluation context while explicitly explaining what failed validation.
    """
    return (
        f"{original_prompt}\n\n"
        f"--- CRITICAL CORRECTION REQUIRED ---\n"
        f"Your previous response was rejected due to schema/validation errors:\n"
        f"ERROR: {error_detail}\n"
        f"PREVIOUS RAW OUTPUT: {raw_response[:300]}\n\n"
        f"You MUST output ONLY a valid JSON object matching this exact schema:\n"
        f'{{"decision": "shortlist" | "reject", "score": <int 0-100>, "reason": "<non-empty string>"}}\n'
        f"Do not include markdown backticks (```), commentary, or extra keys."
    )
