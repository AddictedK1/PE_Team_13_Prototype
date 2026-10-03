"""Tests for PDF resume extraction and error handling."""

import io
from pathlib import Path
import pytest
import pypdf

from app.llm_client import MockLLMClient
from app.pdf_parser import (
    CorruptedPDFError,
    EmptyPDFError,
    PDFExtractionError,
    PDFSizeLimitError,
    extract_resume_text,
    extract_resume_text_safe,
)
from app.screen import screen


def make_clean_pdf(lines: list[str]) -> bytes:
    """Helper to generate valid PDF bytes in memory."""
    stream_cmds = ["BT", "/F1 11 Tf", "50 740 Td", "14 TL"]
    for idx, line in enumerate(lines):
        safe = line.replace("(", r"\(").replace(")", r"\)")
        if idx == 0:
            stream_cmds.append(f"({safe}) Tj")
        else:
            stream_cmds.append(f"T* ({safe}) Tj")
    stream_cmds.append("ET")
    content = "\n".join(stream_cmds).encode("latin1")
    stream_len = len(content)

    parts = [b"%PDF-1.4\n"]
    offsets = []
    offsets.append(len(b"".join(parts)))
    parts.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    offsets.append(len(b"".join(parts)))
    parts.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    offsets.append(len(b"".join(parts)))
    parts.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )
    offsets.append(len(b"".join(parts)))
    parts.append(f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode("latin1"))
    parts.append(content)
    parts.append(b"\nendstream\nendobj\n")
    offsets.append(len(b"".join(parts)))
    parts.append(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    xref_pos = len(b"".join(parts))
    parts.append(b"xref\n0 6\n0000000000 65535 f \n")
    for off in offsets:
        parts.append(f"{off:010d} 00000 n \n".encode("latin1"))
    parts.append(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin1"))
    return b"".join(parts)


def test_1_valid_pdf_text_extraction():
    lines = [
        "Jane Doe",
        "Education: B.S. in Computer Science",
        "Experience: Software Engineer at Acme",
        "Skills: Python, Go, Docker, PostgreSQL",
    ]
    pdf_bytes = make_clean_pdf(lines)
    extracted = extract_resume_text(io.BytesIO(pdf_bytes))
    assert "Jane Doe" in extracted
    assert "Software Engineer" in extracted
    assert "Python, Go" in extracted


def test_2_empty_pdf_blank_pages():
    # PDF with blank page and no text
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)

    with pytest.raises(EmptyPDFError, match="Could not extract text from this PDF"):
        extract_resume_text(buf)


def test_3_empty_pdf_zero_bytes():
    with pytest.raises(EmptyPDFError, match="Please upload a candidate resume PDF"):
        extract_resume_text(b"")

    with pytest.raises(EmptyPDFError, match="Please upload a candidate resume PDF"):
        extract_resume_text(None)


def test_4_invalid_corrupted_pdf():
    garbage_bytes = b"This is not a real PDF file at all, just corrupt data."
    with pytest.raises(CorruptedPDFError, match="Invalid or corrupted PDF"):
        extract_resume_text(io.BytesIO(garbage_bytes))


def test_5_pdf_page_limit_exceeded():
    # PDF with more than 10 pages
    writer = pypdf.PdfWriter()
    for _ in range(12):
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)

    with pytest.raises(PDFSizeLimitError, match="exceeds maximum page limit"):
        extract_resume_text(buf, max_pages=10)


def test_6_resume_screening_from_extracted_pdf_text():
    lines = [
        "Rahul Sharma",
        "Education: B.Tech in Computer Science, IIT Bombay (GPA: 8.9/10)",
        "Experience: Software Engineering Intern at TechFlow",
        "- Built RESTful microservices in Python and FastAPI.",
        "Skills: Python, Go, Docker, PostgreSQL, Redis, Git.",
        "Projects: Distributed key-value cache in Go with Raft consensus.",
    ]
    pdf_bytes = make_clean_pdf(lines)
    extracted_text = extract_resume_text(io.BytesIO(pdf_bytes))

    mock_client = MockLLMClient(
        responses=['{"decision": "shortlist", "score": 92, "reason": "Candidate has excellent distributed systems skills."}']
    )

    result = screen(
        resume=extracted_text,
        prompt_version="v1",
        client=mock_client,
    )

    assert result.valid is True
    assert result.decision == "shortlist"
    assert result.score == 92
    assert "Rahul Sharma" in mock_client.recorded_prompts[0]


def test_7_extract_resume_text_safe_wrapper():
    # Valid
    lines = ["Test Candidate", "Skills: Python, SQL"]
    pdf_bytes = make_clean_pdf(lines)
    res_valid = extract_resume_text_safe(pdf_bytes)
    assert res_valid.success is True
    assert "Test Candidate" in res_valid.text
    assert res_valid.error_message is None

    # Corrupt
    res_corrupt = extract_resume_text_safe(b"random bytes")
    assert res_corrupt.success is False
    assert "corrupted" in res_corrupt.error_message.lower()
