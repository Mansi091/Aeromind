import os
import sys
import boto3
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
load_dotenv()

from backend.rag.document_loader import load_pdf, load_pdf_from_s3
from backend.rag.chunker import chunk_text
from backend.rag.vector_store import add_to_vector_store

DOCS_DIR = r"C:\AEROMIND\AVIATION_TECH_DOCUMENTS"
S3_BUCKET = os.getenv("AWS_S3_BUCKET_NAME")

def ingest_from_local():
    if not os.path.exists(DOCS_DIR):
        print(f"Directory not found: {DOCS_DIR}")
        return

    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} local PDF files to process.")

    for filename in pdf_files:
        filepath = os.path.join(DOCS_DIR, filename)
        print(f"\nProcessing {filename} from local...")
        
        try:
            raw_text = load_pdf(filepath)
            chunks = chunk_text(raw_text)
            metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
            add_to_vector_store(chunks, metadatas=metadatas)
        except Exception as e:
            print(f"Failed to process {filename}: {str(e)}")

def ingest_from_s3(bucket_name):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name)
    
    if 'Contents' not in response:
        print(f"No objects found in S3 bucket: {bucket_name}")
        return

    pdf_keys = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.pdf')]
    print(f"Found {len(pdf_keys)} PDF files in S3 to process.")

    for key in pdf_keys:
        print(f"\nProcessing {key} from S3...")
        try:
            raw_text = load_pdf_from_s3(bucket_name, key)
            chunks = chunk_text(raw_text)
            metadatas = [{"source": f"s3://{bucket_name}/{key}", "chunk_index": i} for i in range(len(chunks))]
            add_to_vector_store(chunks, metadatas=metadatas)
        except Exception as e:
            print(f"Failed to process {key} from S3: {str(e)}")

def ingest():
    if S3_BUCKET:
        print(f"S3 Bucket configured ({S3_BUCKET}). Starting cloud ingestion...")
        ingest_from_s3(S3_BUCKET)
    else:
        print("No S3 Bucket configured. Falling back to local ingestion...")
        ingest_from_local()

if __name__ == "__main__":
    ingest()
