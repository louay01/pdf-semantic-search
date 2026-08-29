from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .search import SearchMatch, SearchService


DEFAULT_INDEX_DIR = Path("data/index")


class Searcher(Protocol):
    model_name: str

    def search(self, query: str, top_k: int = 5) -> list[SearchMatch]: ...


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Search query text")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum number of matches to return")


class SearchChunkMetadata(BaseModel):
    document_name: str
    page_number: int
    chunk_index: int


class SearchResult(BaseModel):
    score: float
    content: str
    metadata: SearchChunkMetadata


class SearchResponse(BaseModel):
    query: str
    model_name: str
    results: list[SearchResult]


def resolve_index_dir(index_dir: Path | None = None) -> Path:
    if index_dir is not None:
        return index_dir
    env_value = os.getenv("PDF_SEARCH_INDEX_DIR")
    return Path(env_value) if env_value else DEFAULT_INDEX_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    if app.state.search_service is None:
        app.state.search_service = SearchService.from_index_dir(app.state.index_dir)
    yield


def create_app(*, index_dir: Path | None = None, search_service: Searcher | None = None) -> FastAPI:
    app = FastAPI(title="PDF Semantic Search", version="0.1.0", lifespan=lifespan)
    app.state.search_service = search_service
    app.state.index_dir = resolve_index_dir(index_dir)

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "status": "ok",
            "message": "PDF semantic search API is running",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/search", response_model=SearchResponse)
    def search_endpoint(payload: SearchRequest) -> SearchResponse:
        service = app.state.search_service
        if service is None:
            raise HTTPException(status_code=503, detail="Search service is not available")

        try:
            matches = service.search(payload.query, top_k=payload.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        results = [
            SearchResult(
                score=match.score,
                content=match.content,
                metadata=SearchChunkMetadata(
                    document_name=match.metadata.document_name,
                    page_number=match.metadata.page_number,
                    chunk_index=match.metadata.chunk_index,
                ),
            )
            for match in matches
        ]
        return SearchResponse(query=payload.query, model_name=service.model_name, results=results)

    return app


app = create_app()
