"""
text_extractor.py
------------------
Responsible for ONE job: given a file (PDF, DOCX) or raw text,
return clean plain text. Nothing about embeddings or ranking lives here.

Keeping this separate means if tomorrow you need to support .txt files,
or fix a PDF parsing bug, you touch ONLY this file.
"""

import io
import pdfplumber
import docx  # this is the python-docx package


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF file given as raw bytes."""
    text_chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n".join(text_chunks)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract all text from a .docx file given as raw bytes."""
    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def clean_text(text: str) -> str:
    """
    Basic cleanup, inspired by the preprocessing steps in the Resume2Vec paper:
    - collapse extra whitespace
    - strip leading/trailing space

    NOTE: We deliberately do NOT lowercase or remove stopwords/punctuation here.
    Modern sentence-transformer models are trained on natural, cased, punctuated
    text — aggressive cleaning (lowercasing, stopword removal) actually HURTS
    semantic embedding quality. That kind of cleaning made sense for the older
    keyword-based ATS systems being compared against in the paper, not for
    transformer embeddings.
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # drop empty lines
    return " ".join(lines)


def extract_text(filename: str, file_bytes: bytes) -> str:
    """
    Main entrypoint: figure out file type from extension and dispatch
    to the right extractor. Raises ValueError for unsupported types.
    """
    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        raw = extract_text_from_pdf(file_bytes)
    elif lower_name.endswith(".docx"):
        raw = extract_text_from_docx(file_bytes)
    elif lower_name.endswith(".txt"):
        raw = file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(
            f"Unsupported file type for '{filename}'. Use PDF, DOCX, or TXT."
        )

    return clean_text(raw)
