"""Core screening engine with structured validation and controlled retry logic.

Implements:
    screen(resume, prompt_version, job_description=None, client=None)

Ensures that API or schema validation failures are NEVER converted into a 'reject'
decision, preventing corruption of bias evaluations.
"""

from __future__ import annotations

import logging
from typing import Optional
from app.llm_client import BaseLLMClient, LLMAPIError, get_llm_client
from app.prompts import get_prompt, get_retry_prompt
from app.schema import CandidateEvaluation, ScreeningResult, parse_and_validate_llm_output

logger = logging.getLogger(__name__)


def screen(
    resume: str,
    prompt_version: str = "v1",
    job_description: Optional[str] = None,
    client: Optional[BaseLLMClient] = None,
    max_retries: int = 1,
) -> ScreeningResult:
    """Screen a candidate resume using the specified prompt version.

    Parameters
    ----------
    resume : str
        Candidate resume text.
    prompt_version : str
        Prompt version: 'v1' (baseline) or 'v2' (debiased).
    job_description : Optional[str]
        Job description to screen against. Uses default SDE role if None.
    client : Optional[BaseLLMClient]
        LLM provider client. Auto-configured if None.
    max_retries : int
        Number of controlled retries upon invalid LLM output (default 1).

    Returns
    -------
    ScreeningResult
        Structured screening result with valid=True for valid evaluations,
        or valid=False with decision=None, score=None for validation/API failures.
    """
    version_key = prompt_version.strip().lower()

    # Resolve LLM client
    try:
        llm = client if client is not None else get_llm_client()
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        return ScreeningResult(
            valid=False,
            decision=None,
            score=None,
            reason=f"LLM client initialization error: {e}",
            error="client_init_error",
            prompt_version=version_key,
        )

    # Format the prompt
    try:
        initial_prompt = get_prompt(version=version_key, resume=resume, job_description=job_description)
    except ValueError as e:
        return ScreeningResult(
            valid=False,
            decision=None,
            score=None,
            reason=f"Invalid prompt configuration: {e}",
            error="invalid_prompt_version",
            prompt_version=version_key,
        )

    # Primary LLM invocation
    raw_response = ""
    try:
        raw_response = llm.generate(prompt=initial_prompt, temperature=0.0)
    except LLMAPIError as e:
        logger.warning(f"LLM API error on initial attempt: {e}")
        return ScreeningResult(
            valid=False,
            decision=None,
            score=None,
            reason=f"API error: {e.message}",
            raw_response=None,
            error="api_error",
            prompt_version=version_key,
        )
    except Exception as e:
        logger.exception("Unexpected error during LLM generation")
        return ScreeningResult(
            valid=False,
            decision=None,
            score=None,
            reason=f"Unexpected system error: {e}",
            raw_response=None,
            error="unexpected_error",
            prompt_version=version_key,
        )

    # Validate primary response
    try:
        parsed = parse_and_validate_llm_output(raw_response)
        return ScreeningResult(
            valid=True,
            decision=parsed.decision,
            score=parsed.score,
            reason=parsed.reason,
            raw_response=raw_response,
            retried=False,
            prompt_version=version_key,
        )
    except Exception as primary_val_err:
        validation_error_msg = str(primary_val_err)
        logger.info(f"Primary output failed validation ({validation_error_msg}). Attempting retry...")

    # If primary validation failed and retries are enabled, execute ONE controlled retry
    if max_retries > 0:
        retry_prompt = get_retry_prompt(
            original_prompt=initial_prompt,
            raw_response=raw_response,
            error_detail=validation_error_msg,
        )
        try:
            retry_raw_response = llm.generate(prompt=retry_prompt, temperature=0.0)
            parsed_retry = parse_and_validate_llm_output(retry_raw_response)
            return ScreeningResult(
                valid=True,
                decision=parsed_retry.decision,
                score=parsed_retry.score,
                reason=parsed_retry.reason,
                raw_response=retry_raw_response,
                retried=True,
                prompt_version=version_key,
            )
        except LLMAPIError as e:
            return ScreeningResult(
                valid=False,
                decision=None,
                score=None,
                reason=f"API error during retry: {e.message}",
                raw_response=None,
                retried=True,
                error="api_error",
                prompt_version=version_key,
            )
        except Exception as retry_val_err:
            return ScreeningResult(
                valid=False,
                decision=None,
                score=None,
                reason=f"LLM returned invalid output after retry: {retry_val_err}",
                raw_response=raw_response,
                retried=True,
                error="validation_failure",
                prompt_version=version_key,
            )

    # If no retries allowed, return validation failure
    return ScreeningResult(
        valid=False,
        decision=None,
        score=None,
        reason=f"LLM returned invalid output: {validation_error_msg}",
        raw_response=raw_response,
        retried=False,
        error="validation_failure",
        prompt_version=version_key,
    )
