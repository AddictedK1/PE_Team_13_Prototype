"""Tests for batch execution and counterfactual evaluation layer."""

import json
from pathlib import Path
import pytest
from app.batch import (
    BatchCandidate,
    batch_results_to_dataframe,
    export_batch_results_to_csv,
    export_batch_results_to_json,
    flatten_pairs_to_candidates,
    load_counterfactual_pairs,
    run_batch,
    run_counterfactual_batch,
)
from app.llm_client import MockLLMClient


def test_load_counterfactual_pairs_at_least_30():
    pairs = load_counterfactual_pairs()
    assert isinstance(pairs, list)
    assert len(pairs) >= 30, f"Expected at least 30 counterfactual pairs, got {len(pairs)}"

    # Check structure of the first pair
    first = pairs[0]
    assert "pair_id" in first
    assert "candidate_a" in first
    assert "candidate_b" in first
    assert "resume" in first["candidate_a"]
    assert "resume" in first["candidate_b"]


def test_flatten_pairs_to_candidates():
    pairs = [
        {
            "pair_id": 101,
            "bias_category": "gender",
            "attribute_modified": "name",
            "candidate_a": {"candidate_id": "101A", "attribute_value": "Alice", "resume": "Software engineer"},
            "candidate_b": {"candidate_id": "101B", "attribute_value": "Bob", "resume": "Software engineer"},
        }
    ]
    candidates = flatten_pairs_to_candidates(pairs)
    assert len(candidates) == 2
    assert candidates[0].candidate_id == "101A"
    assert candidates[0].attribute_value == "Alice"
    assert candidates[0].pair_id == 101
    assert candidates[1].candidate_id == "101B"


def test_run_batch_with_mock():
    mock_client = MockLLMClient(
        responses=[
            '{"decision": "shortlist", "score": 85, "reason": "Candidate A meets criteria."}',
            '{"decision": "reject", "score": 45, "reason": "Candidate B missing skills."}',
        ]
    )

    candidates = [
        BatchCandidate(
            candidate_id="1A",
            resume="Experienced python software developer with 4 years experience.",
            pair_id=1,
            bias_category="gender",
            attribute_modified="name",
            attribute_value="Candidate A",
        ),
        BatchCandidate(
            candidate_id="1B",
            resume="Entry level software developer with python knowledge.",
            pair_id=1,
            bias_category="gender",
            attribute_modified="name",
            attribute_value="Candidate B",
        ),
    ]

    results = run_batch(candidates, prompt_version="v1", client=mock_client)

    assert len(results) == 2
    assert results[0]["candidate_id"] == "1A"
    assert results[0]["decision"] == "shortlist"
    assert results[0]["score"] == 85
    assert results[0]["valid"] is True
    assert results[0]["pair_id"] == 1
    assert results[0]["prompt_version"] == "v1"

    assert results[1]["candidate_id"] == "1B"
    assert results[1]["decision"] == "reject"
    assert results[1]["score"] == 45
    assert results[1]["valid"] is True


def test_batch_results_serialization_and_dataframe(tmp_path: Path):
    mock_client = MockLLMClient()
    candidates = [
        BatchCandidate(candidate_id="1A", resume="Python software engineer", pair_id=1),
        BatchCandidate(candidate_id="1B", resume="Python software engineer", pair_id=1),
    ]

    results = run_batch(candidates, prompt_version="v2", client=mock_client)
    df = batch_results_to_dataframe(results)
    assert len(df) == 2
    assert "candidate_id" in df.columns
    assert "decision" in df.columns
    assert "score" in df.columns

    # Test JSON export
    json_path = tmp_path / "batch_out.json"
    export_batch_results_to_json(results, json_path)
    assert json_path.is_file()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert len(loaded) == 2

    # Test CSV export
    csv_path = tmp_path / "batch_out.csv"
    export_batch_results_to_csv(results, csv_path)
    assert csv_path.is_file()
    content = csv_path.read_text(encoding="utf-8")
    assert "candidate_id" in content
    assert "1A" in content
