import pymupdf
import boto3
import tempfile
import os

def load_pdf(filepath:str) ->str:
    """open a pdf file, extract text page by page and return the full text"""
    text=""
    with pymupdf.open(filepath) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text

def load_pdf_from_s3(bucket_name: str, object_key: str) -> str:
    """Download a PDF from S3 to a temporary file and extract its text."""
    s3 = boto3.client('s3')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        s3.download_file(bucket_name, object_key, tmp_file.name)
        tmp_filepath = tmp_file.name
    
    try:
        text = load_pdf(tmp_filepath)
    finally:
        os.remove(tmp_filepath)
        
    return text