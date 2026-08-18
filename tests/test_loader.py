"""Unit tests for PDFDocLoader and PDF chunking logic."""

import io
from pathlib import Path
import pytest
from pypdf import PdfWriter
from app.core.loader import PDFDocLoader, DocumentChunk


@pytest.fixture
def loader():
    return PDFDocLoader(chunk_size=300, chunk_overlap=30)


@pytest.fixture
def sample_pdf_bytes():
    """Create an in-memory 2-page PDF for test assertions."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    
    # We can also test with actual file in docs_source if present
    bio = io.BytesIO()
    writer.write(bio)
    return bio.getvalue()


def test_load_pdf_bytes_empty(loader):
    """Test loader handles empty bytes gracefully."""
    assert loader.load_pdf_bytes(b"") == []


def test_load_sample_pdf(loader):
    """Test loading the generated technical manual PDF from docs_source."""
    pdf_path = Path("docs_source/manual_tecnico_backend.pdf")
    if not pdf_path.exists():
        pytest.skip("Sample PDF not found in docs_source")

    chunks = loader.load_file(pdf_path)
    assert len(chunks) > 0
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    assert all(c.id for c in chunks)
    assert all(c.metadata.get("source") == "manual_tecnico_backend.pdf" for c in chunks)
    assert all("page" in c.metadata for c in chunks)
    assert all(c.metadata["total_pages"] >= 1 for c in chunks)


def test_load_directory(loader):
    """Test directory scanning and batch PDF chunking."""
    docs_dir = Path("docs_source")
    if not docs_dir.exists():
        pytest.skip("docs_source directory not found")

    chunks = loader.load_directory(docs_dir)
    assert len(chunks) > 0
    pages = {c.metadata.get("page") for c in chunks}
    assert 1 in pages
