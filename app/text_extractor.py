import io
import pdfplumber
import docx  


def extract_text_from_pdf(file_bytes: bytes) -> str:

    text_chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n".join(text_chunks)


def extract_text_from_docx(file_bytes: bytes) -> str:
    
    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def clean_text(text: str) -> str:
    
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # drop empty lines
    return " ".join(lines)


def extract_text(filename: str, file_bytes: bytes) -> str:
   
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
