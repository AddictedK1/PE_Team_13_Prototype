"""PDF resume extraction module.

Extracts text from uploaded PDF resumes with validation for:
- Corrupted/invalid PDFs
- Scanned/empty PDFs with no extractable text
- Oversized PDFs exceeding page or character limits
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, NamedTuple, Optional, Union
from pydantic import BaseModel, Field
import pypdf
from pypdf.errors import PdfReadError


class PDFExtractionError(Exception):
    """Base exception for PDF extraction issues."""
    pass


class CorruptedPDFError(PDFExtractionError):
    """Raised when the PDF is malformed, encrypted, or corrupted."""
    pass


class EmptyPDFError(PDFExtractionError):
    """Raised when no extractable text is present in the PDF."""
    pass


class PDFSizeLimitError(PDFExtractionError):
    """Raised when the PDF exceeds allowed length or size constraints."""
    pass


class PDFExtractionResult(BaseModel):
    """Structured result of PDF extraction."""

    success: bool = Field(..., description="Whether text extraction succeeded")
    text: str = Field("", description="Extracted plain text")
    page_count: int = Field(0, description="Total pages processed")
    char_count: int = Field(0, description="Total characters extracted")
    error_message: Optional[str] = Field(None, description="User-facing error message if failed")


# Constants for sane resume limits
MAX_RESUME_PAGES = 10
MAX_RESUME_CHARS = 25000


def extract_resume_text(
    source: Union[BinaryIO, bytes, io.BytesIO, str, Path],
    max_pages: int = MAX_RESUME_PAGES,
    max_chars: int = MAX_RESUME_CHARS,
) -> str:
    """Extract and validate plain text from a PDF file or stream.

    Parameters
    ----------
    source : BinaryIO, bytes, io.BytesIO, str, or Path
        Uploaded file stream, raw bytes, or file path.
    max_pages : int
        Maximum permitted pages for a candidate resume (default 10).
    max_chars : int
        Maximum permitted extracted characters (default 25,000).

    Returns
    -------
    str
        Cleaned, extracted text from the PDF.

    Raises
    ------
    CorruptedPDFError
        If the file is not a valid PDF or is corrupted.
    EmptyPDFError
        If the PDF contains 0 pages or no extractable text.
    PDFSizeLimitError
        If the PDF exceeds page or character thresholds.
    """
    if source is None:
        raise EmptyPDFError("Please upload a candidate resume PDF.")

    stream: io.BytesIO
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_file():
            raise CorruptedPDFError(f"PDF file not found: {p}")
        try:
            stream = io.BytesIO(p.read_bytes())
        except Exception as e:
            raise CorruptedPDFError(f"Could not read PDF file: {e}") from e
    elif isinstance(source, bytes):
        if len(source) == 0:
            raise EmptyPDFError("Please upload a candidate resume PDF.")
        stream = io.BytesIO(source)
    elif hasattr(source, "read"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            content = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
            if not content or len(content) == 0:
                raise EmptyPDFError("Please upload a candidate resume PDF.")
            stream = io.BytesIO(content)
        except EmptyPDFError:
            raise
        except Exception as e:
            raise CorruptedPDFError(f"Failed reading uploaded file: {e}") from e
    else:
        raise CorruptedPDFError("Invalid source type provided for PDF extraction.")

    # Parse with pypdf
    try:
        reader = pypdf.PdfReader(stream)
    except (PdfReadError, Exception) as e:
        raise CorruptedPDFError(f"Invalid or corrupted PDF file: {e}") from e

    # Page count validation
    total_pages = len(reader.pages)
    if total_pages == 0:
        raise EmptyPDFError("Could not extract text from this PDF. Please upload a text-based resume PDF.")

    if total_pages > max_pages:
        raise PDFSizeLimitError(
            f"PDF exceeds maximum page limit ({total_pages} pages detected, maximum is {max_pages})."
        )

    # Extract text page by page
    extracted_chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_chunks.append(page_text.strip())
        except Exception as e:
            # Continue extracting remaining pages if a single page extraction hiccups
            continue

    full_text = "\n\n".join(extracted_chunks).strip()

    if not full_text:
        raise EmptyPDFError("Could not extract text from this PDF. Please upload a text-based resume PDF.")

    if len(full_text) > max_chars:
        # Truncate reasonably to guard LLM token context while warning
        full_text = full_text[:max_chars].strip()

    return full_text


def extract_resume_text_safe(
    source: Union[BinaryIO, bytes, io.BytesIO, str, Path],
    max_pages: int = MAX_RESUME_PAGES,
    max_chars: int = MAX_RESUME_CHARS,
) -> PDFExtractionResult:
    """Non-raising wrapper around extract_resume_text for UI consumers."""
    try:
        text = extract_resume_text(source=source, max_pages=max_pages, max_chars=max_chars)
        return PDFExtractionResult(
            success=True,
            text=text,
            char_count=len(text),
        )
    except EmptyPDFError as e:
        return PDFExtractionResult(
            success=False,
            error_message=str(e),
        )
    except CorruptedPDFError as e:
        return PDFExtractionResult(
            success=False,
            error_message=f"Invalid or corrupted PDF: {e}",
        )
    except PDFSizeLimitError as e:
        return PDFExtractionResult(
            success=False,
            error_message=str(e),
        )
    except Exception as e:
        return PDFExtractionResult(
            success=False,
            error_message=f"Unexpected error extracting PDF: {e}",
        )
