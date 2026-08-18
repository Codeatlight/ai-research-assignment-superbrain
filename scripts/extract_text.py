"""Text extraction for the ingestion pipeline.

Supported inputs:
  - plain text / markdown files (.txt, .md)
  - PDFs with an embedded text layer via pypdf (optional dependency)

The output is a structured document preserving section and (where available)
page information. This is the ONLY text-extraction path; a future uploaded PDF
flows through the same function, keeping the ingestion architecture uniform.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TextBlock:
    text: str
    section: str = "Body"
    page: int | None = None


@dataclass
class ExtractedDocument:
    paper_id: str
    blocks: list[TextBlock] = field(default_factory=list)


_HEADING_RE = re.compile(r"^(#{1,3}\s+|(?:\d{1,2}(?:\.\d{1,2})*)[.\s]+[A-Z])")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split flat text into (section, body) pairs by heading-like lines."""
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    current_heading = "Body"
    current_body: list[str] = []

    def flush() -> None:
        body = "\n".join(current_body).strip()
        if body:
            sections.append((current_heading, body))

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if stripped and _HEADING_RE.match(stripped):
            flush()
            current_heading = stripped.strip("# ").strip()
            current_body = []
        else:
            current_body.append(line)
    flush()
    return sections


def extract_text(path: Path) -> ExtractedDocument:
    """Extract a structured document from a supported source file."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return _extract_plain(path)
    if suffix == ".pdf":
        return _extract_pdf(path)
    raise ValueError(f"Unsupported source file type: {suffix}")


def _extract_plain(path: Path) -> ExtractedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    doc = ExtractedDocument(paper_id=path.stem)
    for section, body in _split_sections(text):
        for paragraph in _paragraphs(body):
            doc.blocks.append(TextBlock(text=paragraph, section=section, page=None))
    return doc


def _extract_pdf(path: Path) -> ExtractedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pypdf is required to extract text from PDFs. Install with "
            "'pip install pypdf' or provide a .txt/.md copy of the paper."
        ) from exc

    reader = PdfReader(str(path))
    doc = ExtractedDocument(paper_id=path.stem)
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for section, body in _split_sections(text):
            for paragraph in _paragraphs(body):
                doc.blocks.append(TextBlock(text=paragraph, section=section, page=i))
    return doc


def _paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]
