from fastapi.testclient import TestClient

from backend import main


client = TestClient(main.app)


def test_upload_pdf_saves_and_indexes_chunks(tmp_path, monkeypatch):
    indexed = {}
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)
    monkeypatch.setattr(main, "load_pdf", lambda path: "extracted manual text")
    monkeypatch.setattr(main, "chunk_text", lambda text: ["chunk one", "chunk two"])

    def record_chunks(chunks, metadatas=None):
        indexed["chunks"] = chunks
        indexed["metadatas"] = metadatas

    monkeypatch.setattr(main, "add_to_vector_store", record_chunks)

    response = client.post(
        "/documents",
        files={"file": ("manual.pdf", b"%PDF-valid-content", "application/pdf")},
    )

    assert response.status_code == 201
    result = response.json()
    assert result["filename"].startswith("manual-")
    assert result["chunks_added"] == 2
    assert (tmp_path / result["filename"]).read_bytes() == b"%PDF-valid-content"
    assert indexed["chunks"] == ["chunk one", "chunk two"]
    assert all(item["source"] == result["filename"] for item in indexed["metadatas"])


def test_upload_rejects_non_pdf_file():
    response = client.post(
        "/documents",
        files={"file": ("manual.txt", b"not a PDF", "text/plain")},
    )

    assert response.status_code == 400


def test_upload_rejects_invalid_pdf_content():
    response = client.post(
        "/documents",
        files={"file": ("manual.pdf", b"not a PDF", "application/pdf")},
    )

    assert response.status_code == 400


def test_query_returns_actionable_error_when_embedding_model_fails(monkeypatch):
    def fail_to_invoke(state):
        raise ImportError("blocked embedding runtime")

    monkeypatch.setattr(main.rag_app, "invoke", fail_to_invoke)
    response = client.post(
        "/query",
        json={"query": "What are the types of airport?", "thread_id": "test"},
    )

    assert response.status_code == 503
    assert "embedding model could not be loaded" in response.json()["detail"]