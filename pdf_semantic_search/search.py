from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .embeddings import EmbeddingModel, embed_texts, load_embedding_model
from .ingest import ChunkRecord


@dataclass(slots=True)
class SearchMatch:
    score: float
    content: str
    metadata: ChunkRecord


def load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    records: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw_line = line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            records.append(record)

    if not records:
        raise ValueError(f"Metadata file is empty: {metadata_path}")

    sorted_records = sorted(records, key=lambda record: int(record["vector_id"]))
    for expected_id, record in enumerate(sorted_records):
        if int(record["vector_id"]) != expected_id:
            raise ValueError("Metadata vector_id values must be contiguous and zero-based")

    return sorted_records


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_faiss_index(index_path: Path) -> Any:
    import faiss

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index file does not exist: {index_path}")
    return faiss.read_index(str(index_path))


class SearchService:
    def __init__(
        self,
        *,
        index: Any,
        metadata_records: list[dict[str, Any]],
        model_name: str,
        embedding_model: EmbeddingModel | None = None,
        batch_size: int = 32,
    ) -> None:
        self.index = index
        self.metadata_records = metadata_records
        self.model_name = model_name
        self._embedding_model = embedding_model
        self.batch_size = batch_size

        if getattr(self.index, "ntotal", 0) != len(self.metadata_records):
            raise ValueError("FAISS index size does not match metadata record count")

    @classmethod
    def from_index_dir(
        cls,
        index_dir: Path,
        *,
        embedding_model: EmbeddingModel | None = None,
        batch_size: int = 32,
    ) -> "SearchService":
        manifest = load_manifest(index_dir / "manifest.json")
        metadata_records = load_metadata(index_dir / "metadata.jsonl")
        index = load_faiss_index(index_dir / "index.faiss")
        model_name = str(manifest["model_name"])

        return cls(
            index=index,
            metadata_records=metadata_records,
            model_name=model_name,
            embedding_model=embedding_model,
            batch_size=batch_size,
        )

    @property
    def embedding_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            self._embedding_model = load_embedding_model(self.model_name)
        return self._embedding_model

    def search(self, query: str, top_k: int = 5) -> list[SearchMatch]:
        import numpy as np

        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not self.metadata_records:
            return []

        query_vector = embed_texts(
            [normalized_query],
            model=self.embedding_model,
            batch_size=self.batch_size,
        )[0]
        query_matrix = np.asarray([query_vector], dtype="float32")
        limit = min(top_k, len(self.metadata_records))
        scores, vector_ids = self.index.search(query_matrix, limit)

        matches: list[SearchMatch] = []
        for score, vector_id in zip(scores[0], vector_ids[0], strict=False):
            if vector_id < 0:
                continue
            metadata = self.metadata_records[vector_id]
            matches.append(
                SearchMatch(
                    score=float(score),
                    content=str(metadata["content"]),
                    metadata=ChunkRecord(
                        document_name=str(metadata["document_name"]),
                        page_number=int(metadata["page_number"]),
                        chunk_index=int(metadata["chunk_index"]),
                        content=str(metadata["content"]),
                    ),
                )
            )

        return matches
