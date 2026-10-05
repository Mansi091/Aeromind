from backend.rag import vector_store
from langchain_postgres.v2.hybrid_search_config import reciprocal_rank_fusion


def test_retrieve_context_uses_current_query_for_full_text_search(monkeypatch):
    captured = {}

    class FakeVectorStore:
        def similarity_search(self, query, k, hybrid_search_config):
            captured["query"] = query
            captured["k"] = k
            captured["fts_query"] = hybrid_search_config.fts_query
            captured["fusion"] = hybrid_search_config.fusion_function
            return ["retrieved chunk"]

    monkeypatch.setattr(vector_store, "get_vector_store", lambda: FakeVectorStore())

    docs = vector_store.retrieve_context("runway category", k=3)

    assert docs == ["retrieved chunk"]
    assert captured == {
        "query": "runway category",
        "k": 3,
        "fts_query": "runway category",
        "fusion": reciprocal_rank_fusion,
    }


def test_reciprocal_rank_fusion_rewards_results_found_by_both_searches():
    vector_results = [
        {"langchain_id": "shared", "distance": 0.2},
        {"langchain_id": "semantic-only", "distance": 0.3},
    ]
    keyword_results = [
        {"langchain_id": "shared", "distance": 0.9},
        {"langchain_id": "keyword-only", "distance": 0.8},
    ]

    fused = reciprocal_rank_fusion(
        vector_results,
        keyword_results,
        fetch_top_k=3,
    )

    assert [item["langchain_id"] for item in fused][0] == "shared"
    assert {item["langchain_id"] for item in fused} == {
        "shared",
        "semantic-only",
        "keyword-only",
    }