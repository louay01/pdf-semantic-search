from __future__ import annotations

from typing import Protocol


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingModel(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> object: ...


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> EmbeddingModel:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def embed_texts(
    texts: list[str],
    *,
    model: EmbeddingModel,
    batch_size: int = 32,
) -> list[list[float]]:
    if not texts:
        return []

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    normalized_vectors = [list(map(float, row)) for row in vectors]
    if len(normalized_vectors) != len(texts):
        raise ValueError("embedding count does not match text count")
    if not normalized_vectors or not normalized_vectors[0]:
        raise ValueError("embedding model returned empty vectors")

    dimensions = len(normalized_vectors[0])
    if any(len(row) != dimensions for row in normalized_vectors):
        raise ValueError("embedding vectors must have a consistent size")

    return normalized_vectors

