import os
import sys
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
load_dotenv()

from backend.rag.document_loader import load_pdf
from backend.rag.chunker import chunk_text
from backend.rag.vector_store import add_to_vector_store

DOCS_DIR = os.getenv("DOCUMENTS_DIR", os.path.join(PROJECT_ROOT, "data", "documents"))

def ingest_from_local():
    if not os.path.exists(DOCS_DIR):
        print(f"Documents directory not found: {DOCS_DIR}. Create it and add PDF files.")
        return

    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]
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

def ingest():
    ingest_from_local()

if __name__ == "__main__":
    ingest()
