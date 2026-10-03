"""Schema definitions and validation for Candidate Screening."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field, field_validator


DecisionType = Literal["shortlist", "reject"]


class CandidateEvaluation(BaseModel):
    """Schema representing the expected valid output from the LLM."""

    decision: DecisionType = Field(
        ...,
        description="Candidate screening decision: either 'shortlist' or 'reject'",
    )
    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Candidate suitability score as an integer from 0 to 100",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Non-empty explanation for the screening decision",
    )

    @field_validator("reason")
    @classmethod
    def validate_reason_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Reason must be a non-empty string.")
        return stripped

    @field_validator("decision", mode="before")
    @classmethod
    def normalize_decision(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_lower = v.strip().lower()
            if v_lower == "selected":
                return "shortlist"
            if v_lower in ("shortlist", "reject", "rejected"):
                return "reject" if v_lower == "rejected" else v_lower
        return v


class ScreeningResult(BaseModel):
    """Uniform result object returned by screen().

    Distinguishes genuine evaluations (valid=True) from validation/system failures (valid=False).
    A system failure MUST NOT be recorded as decision='reject'.
    """

    valid: bool = Field(..., description="Whether the screening succeeded and passed validation")
    decision: Optional[DecisionType] = Field(
        None, description="'shortlist' or 'reject' if valid, None if invalid"
    )
    score: Optional[int] = Field(
        None, ge=0, le=100, description="0-100 suitability score if valid, None if invalid"
    )
    reason: str = Field(..., description="Explanation if valid, or error message if invalid")
    raw_response: Optional[str] = Field(None, description="Raw model text before parsing")
    retried: bool = Field(False, description="Whether a retry was executed")
    error: Optional[str] = Field(None, description="Error category or message if applicable")
    prompt_version: Optional[str] = Field(None, description="Prompt version used ('v1' or 'v2')")

    def to_evaluation_dict(self, pair_id: Optional[Any] = None, candidate_id: Optional[str] = None) -> Dict[str, Any]:
        """Convert to flat dictionary matching Member 4 evaluation expectations."""
        out: Dict[str, Any] = {
            "pair_id": pair_id,
            "candidate_id": candidate_id,
            "prompt_version": self.prompt_version,
            "decision": self.decision,
            "score": self.score,
            "reason": self.reason,
            "valid": self.valid,
        }
        if self.error:
            out["error"] = self.error
        return out


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract and parse JSON object from LLM response text.

    Handles:
    - Raw JSON strings
    - Markdown code fences (```json ... ``` or ``` ... ```)
    - Surrounding conversational text
    """
    if not text or not text.strip():
        raise ValueError("Empty response received from LLM.")

    cleaned = text.strip()

    # 1. Try direct JSON parsing
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 2. Extract from markdown code fences: ```json ... ``` or ``` ... ```
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(fence_pattern, cleaned, re.IGNORECASE)
    if match:
        fence_content = match.group(1).strip()
        try:
            data = json.loads(fence_content)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # 3. Look for innermost/outermost JSON object bounds '{ ... }'
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate_json = cleaned[start_idx : end_idx + 1]
        try:
            data = json.loads(candidate_json)
            if isinstance(data, dict):
                return data
        except Exception as e:
            raise ValueError(f"Found JSON object syntax, but parsing failed: {e}") from e

    raise ValueError(f"Could not extract a valid JSON object from model output: {cleaned[:100]}...")


def parse_and_validate_llm_output(raw_text: str) -> CandidateEvaluation:
    """Parse raw LLM output and validate against CandidateEvaluation schema.

    Raises ValueError or pydantic.ValidationError if invalid.
    """
    json_obj = extract_json_from_text(raw_text)
    if "score" not in json_obj:
        if "fitness_score" in json_obj:
            json_obj["score"] = json_obj["fitness_score"]
        elif "ats_score" in json_obj:
            json_obj["score"] = json_obj["ats_score"]
    return CandidateEvaluation(**json_obj)
