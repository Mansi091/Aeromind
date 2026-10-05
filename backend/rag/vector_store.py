"""Local SQLite storage and hybrid retrieval for aviation manual chunks."""

from __future__ import annotations

import json
import math
import os
import sqlite3
from functools import lru_cache
from pathlib import Path
from contextlib import closing

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

DATABASE_PATH = Path(
    os.getenv("SQLITE_DB_PATH", Path(__file__).resolve().parents[2] / "data" / "aeromind.sqlite3")
).expanduser()
RESULT_COUNT = 5


@lru_cache(maxsize=1)
def get_embeddings() -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def _connect() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS aviation_chunks (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS aviation_chunks_fts
        USING fts5(content, content='aviation_chunks', content_rowid='id')
        """
    )
    connection.executescript(
        """
        CREATE TRIGGER IF NOT EXISTS aviation_chunks_ai AFTER INSERT ON aviation_chunks BEGIN
            INSERT INTO aviation_chunks_fts(rowid, content) VALUES (new.id, new.content);
        END;
        CREATE TRIGGER IF NOT EXISTS aviation_chunks_ad AFTER DELETE ON aviation_chunks BEGIN
            INSERT INTO aviation_chunks_fts(aviation_chunks_fts, rowid, content)
            VALUES ('delete', old.id, old.content);
        END;
        CREATE TRIGGER IF NOT EXISTS aviation_chunks_au AFTER UPDATE ON aviation_chunks BEGIN
            INSERT INTO aviation_chunks_fts(aviation_chunks_fts, rowid, content)
            VALUES ('delete', old.id, old.content);
            INSERT INTO aviation_chunks_fts(rowid, content) VALUES (new.id, new.content);
        END;
        """
    )
    return connection


def add_to_vector_store(chunks: list[str], metadatas: list[dict] | None = None) -> None:
    if not chunks:
        return
    if metadatas is None:
        metadatas = [{} for _ in chunks]
    if len(metadatas) != len(chunks):
        raise ValueError("There must be one metadata object per chunk.")

    embeddings = get_embeddings().embed_documents(chunks)
    rows = [
        (content, json.dumps(metadata), json.dumps(embedding))
        for content, metadata, embedding in zip(chunks, metadatas, embeddings)
    ]
    with closing(_connect()) as connection:
        with connection:
            connection.executemany(
                "INSERT INTO aviation_chunks(content, metadata, embedding) VALUES (?, ?, ?)",
                rows,
            )
    print(f"Added {len(chunks)} chunks to SQLite database '{DATABASE_PATH}'")


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _fts_query(query: str) -> str:
    # FTS5 treats punctuation and operators specially; quoting each term keeps
    # ordinary user questions from becoming malformed MATCH expressions.
    terms = [term.replace('"', '""') for term in query.split() if term.strip()]
    return " OR ".join(f'"{term}"' for term in terms)


def retrieve_context(query: str, k: int = RESULT_COUNT) -> list[Document]:
    query_vector = get_embeddings().embed_query(query)
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT id, content, metadata, embedding FROM aviation_chunks"
        ).fetchall()
        if not rows:
            return []

        query_text = _fts_query(query)
        keyword_rows = []
        if query_text:
            try:
                keyword_rows = connection.execute(
                    """
                    SELECT rowid AS id, bm25(aviation_chunks_fts) AS score
                    FROM aviation_chunks_fts
                    WHERE aviation_chunks_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (query_text, max(k * 4, 20)),
                ).fetchall()
            except sqlite3.OperationalError:
                # Queries containing unsupported token syntax still get vector search.
                keyword_rows = []

    semantic_ranked = sorted(
        rows,
        key=lambda row: _cosine_similarity(query_vector, json.loads(row["embedding"])),
        reverse=True,
    )[: max(k * 4, 20)]
    scores: dict[int, float] = {}
    rrf_constant = 60
    for ranking in (semantic_ranked, keyword_rows):
        for rank, row in enumerate(ranking, start=1):
            row_id = int(row["id"])
            scores[row_id] = scores.get(row_id, 0.0) + 1.0 / (rrf_constant + rank)

    by_id = {int(row["id"]): row for row in rows}
    selected = sorted(scores, key=scores.get, reverse=True)[:k]
    return [
        Document(
            page_content=by_id[row_id]["content"],
            metadata=json.loads(by_id[row_id]["metadata"]),
        )
        for row_id in selected
    ]
