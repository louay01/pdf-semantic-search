from fastapi.testclient import TestClient

from pdf_semantic_search.api import create_app
from pdf_semantic_search.ingest import ChunkRecord
from pdf_semantic_search.search import SearchMatch


class FakeSearchService:
    model_name = "fake-multilingual-model"

    def search(self, query: str, top_k: int = 5) -> list[SearchMatch]:
        if not query.strip():
            raise ValueError("query must not be empty")

        return [
            SearchMatch(
                score=0.91,
                content="match one",
                metadata=ChunkRecord(
                    document_name="doc-a.pdf",
                    page_number=2,
                    chunk_index=1,
                    content="match one",
                ),
            ),
            SearchMatch(
                score=0.77,
                content="match two",
                metadata=ChunkRecord(
                    document_name="doc-b.pdf",
                    page_number=1,
                    chunk_index=0,
                    content="match two",
                ),
            ),
        ][:top_k]


def test_search_endpoint_returns_ranked_matches():
    client = TestClient(create_app(search_service=FakeSearchService()))

    response = client.post("/search", json={"query": "bonjour", "top_k": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "bonjour"
    assert payload["model_name"] == "fake-multilingual-model"
    assert len(payload["results"]) == 2
    assert payload["results"][0]["score"] == 0.91
    assert payload["results"][0]["content"] == "match one"
    assert payload["results"][0]["metadata"] == {
        "document_name": "doc-a.pdf",
        "page_number": 2,
        "chunk_index": 1,
    }


def test_search_endpoint_rejects_blank_query():
    client = TestClient(create_app(search_service=FakeSearchService()))

    response = client.post("/search", json={"query": "   ", "top_k": 2})

    assert response.status_code == 400
    assert "query must not be empty" in response.json()["detail"]
