import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chroma")

def get_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_store():
    embeddings = get_embeddings()
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def add_to_vector_store(chunks: list[str], metadatas: list[dict] = None):
    vector_store = get_vector_store()
    vector_store.add_texts(texts=chunks, metadatas=metadatas)
    print(f"Added {len(chunks)} chunks to vector store at {DB_DIR}")

def retrieve_context(query: str, k: int = 5):
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=k)
    return docs
