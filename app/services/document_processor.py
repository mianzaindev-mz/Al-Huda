"""
services/document_processor.py
-------------------------------
Handles all file ingestion for the Islamic knowledge base:
  - Text extraction from PDF, DOCX, and TXT files
  - Smart semantic chunking
  - Scanning the DataBase folder and populating the vector index
"""

import io
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import PyPDF2

from app.core.config import DATABASE_FOLDER, DOCX_AVAILABLE, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

# Optional DOCX support — flag is authoritative from config.py
try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None  # type: ignore[assignment]
    if not DOCX_AVAILABLE:
        logger.warning("python-docx not installed — DOCX support disabled. Run: pip install python-docx")


# ---------------------------------------------------------------------------
# Text Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_content: bytes) -> Tuple[str, bool, Optional[str]]:
    """Extract and clean text from a PDF byte string.

    Returns (text, success, error_message).
    """
    try:
        logger.info("Extracting text from PDF…")
        reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        ok_pages = 0

        for page_num, page in enumerate(reader.pages):
            try:
                raw = page.extract_text()
                if raw and len(raw.strip()) > 10:
                    cleaned = " ".join(raw.split())       # Normalise whitespace
                    text += f"\n\n--- Page {page_num + 1} ---\n{cleaned}"
                    ok_pages += 1
            except Exception as exc:
                logger.warning(f"PDF page {page_num + 1} extraction failed: {exc}")

        if not text.strip():
            return "", False, "No meaningful text could be extracted from PDF"

        logger.info(
            f"PDF extraction complete — {len(text)} chars, "
            f"{ok_pages}/{len(reader.pages)} pages"
        )
        return text, True, None

    except Exception as exc:
        error = f"PDF extraction error: {exc}"
        logger.error(error)
        return "", False, error


def extract_text_from_docx(file_path: str) -> Tuple[str, bool, Optional[str]]:
    """Extract paragraph text from a DOCX file.

    Returns (text, success, error_message).
    """
    if not DOCX_AVAILABLE:
        return "", False, "DOCX support unavailable — install python-docx"

    try:
        logger.info(f"Extracting text from DOCX: {file_path}")
        doc = DocxDocument(file_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        if not text.strip():
            return "", False, "No text could be extracted from DOCX"

        logger.info(f"DOCX extraction complete — {len(text)} chars")
        return text, True, None

    except Exception as exc:
        error = f"DOCX extraction error: {exc}"
        logger.error(error)
        return "", False, error


def extract_text_from_txt(file_path: str) -> Tuple[str, bool, Optional[str]]:
    """Read a plain-text file, trying common encodings in order.

    Returns (text, success, error_message).
    """
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    try:
        logger.info(f"Reading TXT: {file_path}")
        text = ""
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as fh:
                    text = fh.read()
                logger.info(f"TXT read with '{enc}' encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            return "", False, "Could not decode file with any supported encoding"

        if not text.strip():
            return "", False, "TXT file is empty"

        logger.info(f"TXT extraction complete — {len(text)} chars")
        return text, True, None

    except Exception as exc:
        error = f"TXT read error: {exc}"
        logger.error(error)
        return "", False, error


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split *text* into semantic chunks respecting paragraph and sentence boundaries.

    Args:
        text:       Raw document text.
        chunk_size: Target chunk size in words.
        overlap:    Words of overlap kept between consecutive chunks.

    Returns:
        List of non-empty text chunk strings (each with at least 30 chars).
    """
    if not text or not text.strip():
        return []

    # Validate overlap size
    overlap = min(overlap, chunk_size // 2)  # Prevent overlap from being too large

    # Primary split on double newlines (paragraph boundaries)
    paragraphs = re.split(r"\n\n+|\r\n\r\n+", text)
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Secondary split on sentence boundaries (with Arabic question mark support)
        sentences = re.split(r"(?<=[.!?؟])\s+", paragraph)

        for sentence in sentences:
            words = sentence.split()
            if not words:
                continue

            word_count = len(words)

            # Check if adding this sentence would exceed chunk size
            if current_length + word_count > chunk_size and current_chunk:
                # Save current chunk
                chunk_text_str = " ".join(current_chunk)
                if len(chunk_text_str.strip()) > 30:
                    chunks.append(chunk_text_str.strip())
                
                # Prepare overlap for next chunk
                num_overlap = min(overlap, len(current_chunk))
                current_chunk = current_chunk[-num_overlap:] + words
                current_length = len(current_chunk)
            else:
                # Add to current chunk
                current_chunk.extend(words)
                current_length += word_count

    # Flush remaining words
    if current_chunk:
        final_chunk = " ".join(current_chunk)
        if len(final_chunk.strip()) > 30:
            chunks.append(final_chunk.strip())

    logger.info(f"Chunking complete — {len(chunks)} chunks created")
    return chunks


# ---------------------------------------------------------------------------
# File dispatcher
# ---------------------------------------------------------------------------

def process_file(file_path: str) -> Tuple[str, bool, Optional[str], str]:
    """Route a file to the correct extractor based on its extension.

    Returns (text, success, error_message, file_type_label).
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        with open(file_path, "rb") as fh:
            content = fh.read()
        text, ok, err = extract_text_from_pdf(content)
        return text, ok, err, "pdf"

    if ext == ".txt":
        text, ok, err = extract_text_from_txt(file_path)
        return text, ok, err, "txt"

    if ext in (".docx", ".doc"):
        text, ok, err = extract_text_from_docx(file_path)
        return text, ok, err, "docx"

    return "", False, f"Unsupported file type: {ext}", "unknown"


# ---------------------------------------------------------------------------
# Database folder scanner
# ---------------------------------------------------------------------------

def scan_database_folder() -> Dict[str, Any]:
    """Scan the DataBase/ folder, extract all supported files, and add them to
    the vector database.

    Returns a summary dict with counts and per-file results.
    """
    # Import here to avoid circular dependency (vector_db imports config)
    from app.services.vector_db import vector_db

    logger.info("=" * 60)
    logger.info(f"Scanning DataBase folder: {DATABASE_FOLDER}")
    logger.info("=" * 60)

    db_path = Path(DATABASE_FOLDER)

    if not db_path.exists():
        logger.warning(f"DataBase folder not found — creating: {DATABASE_FOLDER}")
        db_path.mkdir(parents=True, exist_ok=True)
        return {"status": "created", "message": f"Created empty folder at {DATABASE_FOLDER}", "files_processed": 0}

    if not db_path.is_dir():
        msg = f"{DATABASE_FOLDER} exists but is not a directory"
        logger.error(msg)
        return {"status": "error", "message": msg, "files_processed": 0}

    # Gather all supported files
    all_files: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        all_files.extend(db_path.glob(f"*{ext}"))

    if not all_files:
        logger.info("No supported files found in DataBase folder")
        return {"status": "empty", "message": "No supported files found", "files_processed": 0}

    logger.info(f"Found {len(all_files)} file(s) to process")

    successful: List[Dict] = []
    failed: List[Dict] = []
    all_texts: List[str] = []
    all_metadata: List[Dict] = []

    for file_path in all_files:
        logger.info(f"Processing: {file_path.name}")
        text, ok, error, file_type = process_file(str(file_path))

        if ok and text:
            chunks = chunk_text(text, chunk_size=400, overlap=50)
            if chunks:
                meta = {
                    "source": file_path.name,
                    "type": f"database_{file_type}",
                    "file_path": str(file_path),
                    "processed_at": datetime.now().isoformat(),
                }
                all_texts.extend(chunks)
                all_metadata.extend([meta] * len(chunks))
                successful.append({"file": file_path.name, "chunks": len(chunks), "characters": len(text)})
                logger.info(f"  ✓ {file_path.name} — {len(chunks)} chunks")
            else:
                failed.append({"file": file_path.name, "error": "No meaningful content extracted"})
        else:
            failed.append({"file": file_path.name, "error": error or "Unknown error"})
            logger.error(f"  ✗ {file_path.name}: {error}")

    # Add all chunks in a single batch call
    if all_texts:
        try:
            vector_db.add_texts(all_texts, all_metadata)
        except Exception as exc:
            logger.error(f"Error adding texts to vector database: {exc}")

    total_chunks = sum(r["chunks"] for r in successful)
    logger.info("=" * 60)
    logger.info(f"Scan summary — files: {len(all_files)}, ok: {len(successful)}, failed: {len(failed)}, chunks: {total_chunks}")
    logger.info("=" * 60)

    return {
        "status": "success",
        "files_found": len(all_files),
        "files_processed": len(successful),
        "files_failed": len(failed),
        "total_chunks": total_chunks,
        "successful_files": successful,
        "failed_files": failed,
        "total_db_entries": len(vector_db.texts),
    }
