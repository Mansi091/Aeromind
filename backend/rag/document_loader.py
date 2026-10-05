import pymupdf


def load_pdf(filepath: str) -> str:
    """Extract text from a PDF, page by page."""
    text = ""
    with pymupdf.open(filepath) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text
