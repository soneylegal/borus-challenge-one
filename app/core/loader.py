"""PDF document loader and text chunker for technical documentation."""

from dataclasses import dataclass, field
import hashlib
import io
import logging
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a discrete chunk of text extracted from a document with page metadata."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PDFDocLoader:
    """Loads PDF files from disk or byte streams, splitting them into context-aware chunks with page tracking."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Text splitter tailored for structured paragraphs, code blocks and lists
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _generate_chunk_id(self, source: str, page: int, index: int, content: str) -> str:
        """Generate a deterministic unique ID for each chunk based on source, page and content hash."""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        clean_source = Path(source).stem.replace(" ", "_")
        return f"{clean_source}_p{page}_{index}_{content_hash}"

    def load_pdf_bytes(
        self, pdf_bytes: bytes, filename: str = "document.pdf"
    ) -> list[DocumentChunk]:
        """Extract text page-by-page from raw PDF bytes and split into chunks with metadata."""
        if not pdf_bytes:
            return []

        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            total_pages = len(reader.pages)
            logger.info(f"Processing PDF '{filename}' with {total_pages} pages.")

            chunks: list[DocumentChunk] = []
            chunk_counter = 0

            for page_idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if not page_text:
                    continue

                # Split page text into manageable chunks
                page_splits = self.text_splitter.split_text(page_text)

                for split_idx, split_content in enumerate(page_splits):
                    clean_content = split_content.strip()
                    if not clean_content:
                        continue

                    metadata = {
                        "source": Path(filename).name,
                        "file_path": str(filename),
                        "page": page_idx,
                        "total_pages": total_pages,
                        "chunk_index": chunk_counter,
                        "page_chunk_index": split_idx,
                    }

                    chunk_id = self._generate_chunk_id(
                        filename, page_idx, split_idx, clean_content
                    )
                    chunks.append(
                        DocumentChunk(
                            id=chunk_id,
                            content=clean_content,
                            metadata=metadata,
                        )
                    )
                    chunk_counter += 1

            logger.info(
                f"Generated {len(chunks)} chunks from '{filename}' ({total_pages} pages)."
            )
            return chunks

        except Exception as e:
            logger.error(f"Failed to read PDF stream '{filename}': {e}")
            return []

    def load_file(self, file_path: str | Path) -> list[DocumentChunk]:
        """Load an individual PDF file from disk."""
        path = Path(file_path)
        if not path.is_file():
            logger.warning(f"File not found: {path}")
            return []

        try:
            with open(path, "rb") as f:
                pdf_bytes = f.read()
            return self.load_pdf_bytes(pdf_bytes, filename=str(path.name))
        except Exception as e:
            logger.error(f"Error reading file '{path}': {e}")
            return []

    def load_directory(self, dir_path: str | Path) -> list[DocumentChunk]:
        """Recursively scan a directory for all .pdf files and extract chunks."""
        directory = Path(dir_path)
        if not directory.exists() or not directory.is_dir():
            logger.warning(f"Directory not found: {directory}")
            return []

        all_chunks: list[DocumentChunk] = []
        pdf_files = sorted(list(directory.glob("**/*.pdf")))

        logger.info(f"Found {len(pdf_files)} PDF file(s) in {directory}")
        for pdf_file in pdf_files:
            chunks = self.load_file(pdf_file)
            all_chunks.extend(chunks)

        logger.info(f"Generated {len(all_chunks)} total chunks from PDF directory {directory}")
        return all_chunks


# Alias for compatibility
DocumentLoader = PDFDocLoader
