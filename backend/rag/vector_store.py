from functools import lru_cache
import os
from asyncpg.exceptions import DuplicateObjectError, DuplicateTableError
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.hybrid_search_config import (
    HybridSearchConfig,
    reciprocal_rank_fusion,
)
from langchain_postgres.v2.vectorstores import PGVectorStore
from sqlalchemy.exc import SQLAlchemyError

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://aeromind:aeromind@localhost:5432/aeromind",
)
TABLE_NAME = os.getenv("PGVECTOR_TABLE", "aviation_chunks")
VECTOR_SIZE = 384
RESULT_COUNT = 5


def _asyncpg_database_url() -> str:
    if DATABASE_URL.startswith("postgresql+psycopg://"):
        return DATABASE_URL.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
    if DATABASE_URL.startswith("postgresql://"):
        return DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    return DATABASE_URL


def _hybrid_search_config(query: str = "") -> HybridSearchConfig:
    return HybridSearchConfig(
        tsv_column="content_tsv",
        tsv_lang="pg_catalog.english",
        fts_query=query,
        fusion_function=reciprocal_rank_fusion,
        fusion_function_parameters={"fetch_top_k": RESULT_COUNT},
        primary_top_k=20,
        secondary_top_k=20,
    )


def _is_duplicate_error(error: Exception, expected_error: type[Exception]) -> bool:
    return isinstance(error, expected_error) or isinstance(
        getattr(error, "orig", None), expected_error
    )


@lru_cache(maxsize=1)
def get_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def get_pg_engine():
    return PGEngine.from_connection_string(_asyncpg_database_url())

@lru_cache(maxsize=1)
def get_vector_store():
    engine = get_pg_engine()
    hybrid_config = _hybrid_search_config()
    try:
        engine.init_vectorstore_table(
            table_name=TABLE_NAME,
            vector_size=VECTOR_SIZE,
            hybrid_search_config=hybrid_config,
        )
    except (DuplicateTableError, SQLAlchemyError) as error:
        if not _is_duplicate_error(error, DuplicateTableError):
            raise

    vector_store = PGVectorStore.create_sync(
        engine=engine,
        embedding_service=get_embeddings(),
        table_name=TABLE_NAME,
        k=RESULT_COUNT,
        hybrid_search_config=hybrid_config,
    )
    try:
        vector_store.apply_hybrid_search_index()
    except (DuplicateObjectError, DuplicateTableError, SQLAlchemyError) as error:
        if not (
            _is_duplicate_error(error, DuplicateObjectError)
            or _is_duplicate_error(error, DuplicateTableError)
        ):
            raise
    return vector_store

def add_to_vector_store(chunks: list[str], metadatas: list[dict] | None = None):
    vector_store = get_vector_store()
    vector_store.add_texts(texts=chunks, metadatas=metadatas)
    print(f"Added {len(chunks)} chunks to PostgreSQL table '{TABLE_NAME}'")

def retrieve_context(query: str, k: int = 5):
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(
        query,
        k=k,
        hybrid_search_config=_hybrid_search_config(query),
    )
    return docs
