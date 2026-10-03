"""Regression and UI tests for Streamlit interface and PDF upload stability."""

import os
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from tests.test_pdf_parser import make_clean_pdf


APP_FILE = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


@pytest.fixture(autouse=True)
def setup_mock_llm(monkeypatch):
    """Ensure tests run offline using Mock provider."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")


def test_ui_initial_render():
    """Verify initial UI render contains title, uploader, and controls."""
    at = AppTest.from_file(APP_FILE)
    at.run()

    assert not at.exception
    assert len(at.title) > 0
    assert at.title[0].value == "Bias Audit for Candidate Screening"
    assert len(at.file_uploader) == 1
    assert at.file_uploader[0].label == "Upload candidate resume as PDF"
    assert len(at.button) > 0


def test_ui_pdf_upload_persists_without_blank_screen():
    """Verify that uploading a valid PDF keeps the page visible and shows extracted preview."""
    at = AppTest.from_file(APP_FILE)
    at.run()

    pdf_bytes = make_clean_pdf([
        "Candidate Name: Rahul Sharma",
        "Skills: Python, FastAPI, Docker, SQL",
        "Experience: Software Engineer Intern",
    ])

    uploader = at.file_uploader[0]
    uploader.upload("candidate_resume.pdf", pdf_bytes)
    at.run()

    assert not at.exception
    # Ensure page is NOT blank
    assert len(at.button) > 0
    assert "Run Side-by-Side Audit" in [b.label for b in at.button]

    # Verify upload indicators
    uploaded_labels = [m.value for m in at.markdown if "Uploaded file" in m.value]
    assert len(uploaded_labels) == 1
    assert "candidate_resume.pdf" in uploaded_labels[0]

    # Status success message
    assert any("Resume loaded successfully" in s.value for s in at.success)

    # Collapsible expander present
    assert len(at.expander) >= 1
    assert at.expander[0].label == "Extracted Resume Text"


def test_ui_corrupt_pdf_handled_gracefully():
    """Verify that uploading an invalid PDF shows clear user error and does not blank the screen."""
    at = AppTest.from_file(APP_FILE)
    at.run()

    uploader = at.file_uploader[0]
    uploader.upload("corrupted.pdf", b"Random garbage bytes that do not form a PDF")
    at.run()

    assert not at.exception
    assert len(at.button) > 0

    # User friendly error shown
    error_texts = [e.value for e in at.error]
    assert any("Could not read this PDF" in err for err in error_texts)


def test_ui_audit_execution_from_uploaded_pdf():
    """Verify screening execution from uploaded PDF."""
    at = AppTest.from_file(APP_FILE)
    at.run()

    pdf_bytes = make_clean_pdf([
        "Candidate Name: Priyah Sharma",
        "Skills: Python, FastAPI, Docker, SQL",
        "Experience: Software Engineer Intern",
    ])
    at.file_uploader[0].upload("priyah_resume.pdf", pdf_bytes)
    at.run()

    # Trigger audit
    audit_btn = at.button[0]
    assert audit_btn.label == "Run Side-by-Side Audit"
    audit_btn.click()
    at.run()

    assert not at.exception
    # Decision elements rendered
    decisions = [s.value for s in at.success if "Decision" in s.value or "SHORTLIST" in s.value]
    assert len(decisions) >= 1

    # Audit insights metrics rendered
    metric_labels = [m.label for m in at.metric]
    assert "Suitability Score" in metric_labels


def test_ui_mode_switching_stability():
    """Verify switching between evaluation modes remains stable without crashes."""
    at = AppTest.from_file(APP_FILE)
    at.run()

    # Mode 1: Single Version Screening
    at.radio[0].set_value("Single Version Screening")
    at.run()
    assert not at.exception
    assert "Screen Candidate" in [b.label for b in at.button]

    # Mode 2: Batch Counterfactual Audit
    at.radio[0].set_value("Batch Counterfactual Audit (30+ Pairs)")
    at.run()
    assert not at.exception
    assert "Run Batch Evaluation" in [b.label for b in at.button]
