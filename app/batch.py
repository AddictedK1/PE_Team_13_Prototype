"""Batch execution layer for counterfactual candidate evaluation.

Processes 30+ counterfactual pairs through screen() and produces structured,
serializable JSON/CSV results ready for Member 4's statistical evaluation.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.llm_client import BaseLLMClient
from app.screen import screen

logger = logging.getLogger(__name__)


class BatchCandidate(BaseModel):
    """Candidate representation for batch execution."""

    candidate_id: str
    resume: str
    pair_id: Optional[Union[int, str]] = None
    attribute_value: Optional[str] = None
    bias_category: Optional[str] = None
    attribute_modified: Optional[str] = None


def load_counterfactual_pairs(file_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Load counterfactual pairs from JSON file.

    Defaults to data/pairs.json (labelled pairs, schema a/b) if not specified.
    Pairs in the a/b schema are normalised to the candidate_a/candidate_b schema.
    """
    if file_path is None:
        target = Path(__file__).resolve().parent.parent / "data" / "pairs.json"
    else:
        target = Path(file_path)

    if not target.is_file():
        raise FileNotFoundError(f"Counterfactual pairs dataset not found at: {target}")

    with open(target, "r", encoding="utf-8") as f:
        return [_normalise_pair(p) for p in json.load(f)]


def _normalise_pair(pair: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a pairs.json entry (a/b with text) to candidate_a/candidate_b form."""
    if "a" not in pair or "b" not in pair:
        return pair
    out = {k: v for k, v in pair.items() if k not in ("a", "b")}
    out["bias_category"] = pair.get("type")
    out["attribute_modified"] = pair.get("type")
    for side in ("a", "b"):
        c = pair[side]
        out[f"candidate_{side}"] = {
            "candidate_id": f"{pair['pair_id']}{side.upper()}",
            "attribute_value": c.get("group") or c.get("name"),
            "resume": c["text"],
            "name": c.get("name"),
            "college": c.get("college"),
        }
    return out


def flatten_pairs_to_candidates(pairs: List[Dict[str, Any]]) -> List[BatchCandidate]:
    """Convert paired counterfactual objects into flat list of BatchCandidate items."""
    candidates: List[BatchCandidate] = []
    for pair in pairs:
        pair_id = pair.get("pair_id")
        category = pair.get("bias_category")
        attr_modified = pair.get("attribute_modified")

        for key in ("candidate_a", "candidate_b"):
            c_data = pair.get(key, {})
            candidates.append(
                BatchCandidate(
                    candidate_id=str(c_data.get("candidate_id", "")),
                    resume=str(c_data.get("resume", "")),
                    pair_id=pair_id,
                    attribute_value=c_data.get("attribute_value"),
                    bias_category=category,
                    attribute_modified=attr_modified,
                )
            )
    return candidates


def run_batch(
    candidates: List[Union[BatchCandidate, Dict[str, Any]]],
    prompt_version: str = "v1",
    job_description: Optional[str] = None,
    client: Optional[BaseLLMClient] = None,
    delay_seconds: float = 0.0,
) -> List[Dict[str, Any]]:
    """Execute batch screening across candidates and produce structured evaluation records.

    Parameters
    ----------
    candidates : List[BatchCandidate | Dict]
        Candidates to screen.
    prompt_version : str
        'v1' (baseline) or 'v2' (debiased).
    job_description : Optional[str]
        Job requirements (defaults to SDE I role).
    client : Optional[BaseLLMClient]
        LLM provider client.
    delay_seconds : float
        Optional rate-limit delay between invocations.

    Returns
    -------
    List[Dict[str, Any]]
        Flat list of evaluation-ready results.
    """
    results: List[Dict[str, Any]] = []

    for idx, raw_cand in enumerate(candidates):
        if isinstance(raw_cand, dict):
            cand = BatchCandidate(
                candidate_id=str(raw_cand.get("candidate_id", idx)),
                resume=raw_cand.get("resume", ""),
                pair_id=raw_cand.get("pair_id"),
                attribute_value=raw_cand.get("attribute_value"),
                bias_category=raw_cand.get("bias_category"),
                attribute_modified=raw_cand.get("attribute_modified"),
            )
        else:
            cand = raw_cand

        # Screen candidate
        screen_res = screen(
            resume=cand.resume,
            prompt_version=prompt_version,
            job_description=job_description,
            client=client,
        )

        # Build flat record matching Member 4 expectations
        record: Dict[str, Any] = {
            "pair_id": cand.pair_id,
            "candidate_id": cand.candidate_id,
            "prompt_version": prompt_version.strip().lower(),
            "decision": screen_res.decision,
            "score": screen_res.score,
            "reason": screen_res.reason,
            "valid": screen_res.valid,
            "retried": screen_res.retried,
            "error": screen_res.error,
            "bias_category": cand.bias_category,
            "attribute_modified": cand.attribute_modified,
            "attribute_value": cand.attribute_value,
        }
        results.append(record)

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return results


def run_counterfactual_batch(
    pairs: Optional[List[Dict[str, Any]]] = None,
    prompt_version: str = "v1",
    job_description: Optional[str] = None,
    client: Optional[BaseLLMClient] = None,
    delay_seconds: float = 0.0,
) -> List[Dict[str, Any]]:
    """Convenience helper to run all counterfactual pairs through the specified prompt version."""
    if pairs is None:
        pairs = load_counterfactual_pairs()

    flat_candidates = flatten_pairs_to_candidates(pairs)
    return run_batch(
        candidates=flat_candidates,
        prompt_version=prompt_version,
        job_description=job_description,
        client=client,
        delay_seconds=delay_seconds,
    )


def batch_results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert batch results to pandas DataFrame for tabular display or analysis."""
    return pd.DataFrame(results)


def export_batch_results_to_json(results: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """Export batch screening results to JSON file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def export_batch_results_to_csv(results: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """Export batch screening results to CSV file."""
    df = batch_results_to_dataframe(results)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8")
