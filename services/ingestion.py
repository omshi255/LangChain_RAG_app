import io
import uuid
import logging
from typing import List, Tuple
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def extract_text_from_pdf(file_bytes: bytes) -> List[Tuple[str, int]]:
    """Extract text per page. Returns list of (text, page_number)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    if len(reader.pages) == 0:
        raise ValueError("PDF has no pages.")

    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((text, i + 1))

    if not pages:
        raise ValueError("No extractable text found in PDF. It may be scanned/image-only.")

    return pages


def chunk_pages(pages: List[Tuple[str, int]]) -> List[dict]:
    """Split pages into overlapping chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for text, page_num in pages:
        splits = splitter.split_text(text)
        for split in splits:
            split = split.strip()
            if len(split) > 50:  # skip tiny fragments
                chunks.append({"text": split, "page": page_num})

    if not chunks:
        raise ValueError("Text chunking produced no usable chunks.")

    return chunks


def process_document(file_bytes: bytes, filename: str) -> Tuple[List[dict], str]:
    """Full ingestion pipeline. Returns (chunks, document_id)."""
    if not filename.lower().endswith(".pdf"):
        raise ValueError(f"Unsupported file type: '{filename}'. Only PDF is supported.")

    if len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty.")

    if len(file_bytes) > 50 * 1024 * 1024:  # 50MB limit
        raise ValueError("File too large. Maximum size is 50MB.")

    document_id = str(uuid.uuid4())
    logger.info(f"Processing document: {filename} → ID: {document_id}")

    pages = extract_text_from_pdf(file_bytes)
    logger.info(f"Extracted {len(pages)} pages")

    chunks = chunk_pages(pages)
    logger.info(f"Generated {len(chunks)} chunks")

    # Attach document metadata to each chunk
    for i, chunk in enumerate(chunks):
        chunk["document_id"] = document_id
        chunk["filename"] = filename
        chunk["chunk_index"] = i

    return chunks, document_id