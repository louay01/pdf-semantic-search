from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .chunking import chunk_text
from .embeddings import DEFAULT_MODEL_NAME, EmbeddingModel, embed_texts, load_embedding_model


SUPPORTED_SUFFIXES = {".pdf"}


@dataclass(slots=True)
class ChunkRecord:
    document_name: str
    page_number: int
    chunk_index: int
    content: str


@dataclass(slots=True)
class IngestionResult:
    documents_indexed: int
    chunks_indexed: int
    embedding_dimensions: int
    index_path: Path
    metadata_path: Path
    manifest_path: Path


PageExtractor = Callable[[Path], Iterable[tuple[int, str]]]
IndexWriter = Callable[[list[list[float]], Path], None]


def discover_pdf_files(pdf_dir: Path) -> list[Path]:
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"PDF path is not a directory: {pdf_dir}")

    files = [path for path in pdf_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(files)


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((page_number, text))
    return pages


def build_chunk_records(
    pdf_dir: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    page_extractor: PageExtractor = extract_pdf_pages,
) -> tuple[list[ChunkRecord], int]:
    pdf_files = discover_pdf_files(pdf_dir)
    if not pdf_files:
        raise ValueError(f"No PDF files were found in {pdf_dir}")

    chunk_records: list[ChunkRecord] = []

    for pdf_path in pdf_files:
        document_name = str(pdf_path.relative_to(pdf_dir))
        for page_number, page_text in page_extractor(pdf_path):
            for chunk_index, content in enumerate(
                chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            ):
                chunk_records.append(
                    ChunkRecord(
                        document_name=document_name,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        content=content,
                    )
                )

    if not chunk_records:
        raise ValueError("No text chunks were produced from the provided PDFs")

    return chunk_records, len(pdf_files)


def write_faiss_index(vectors: list[list[float]], index_path: Path) -> None:
    import faiss
    import numpy as np

    if not vectors:
        raise ValueError("Cannot create a FAISS index without vectors")

    matrix = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(index_path))


def write_metadata(records: list[ChunkRecord], metadata_path: Path) -> None:
    with metadata_path.open("w", encoding="utf-8") as handle:
        for vector_id, record in enumerate(records):
            payload = {"vector_id": vector_id, **asdict(record)}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_manifest(
    *,
    manifest_path: Path,
    source_dir: Path,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    documents_indexed: int,
    chunks_indexed: int,
    embedding_dimensions: int,
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir.resolve()),
        "model_name": model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "documents_indexed": documents_indexed,
        "chunks_indexed": chunks_indexed,
        "embedding_dimensions": embedding_dimensions,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ensure_output_paths(index_dir: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "metadata.jsonl"
    manifest_path = index_dir / "manifest.json"

    existing_paths = [path for path in (index_path, metadata_path, manifest_path) if path.exists()]
    if existing_paths and not overwrite:
        names = ", ".join(path.name for path in existing_paths)
        raise FileExistsError(f"Index output already exists in {index_dir}: {names}. Use --overwrite to replace it.")

    for path in existing_paths:
        path.unlink()

    return index_path, metadata_path, manifest_path


def ingest_pdfs(
    pdf_dir: Path,
    *,
    index_dir: Path,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    overwrite: bool = False,
    page_extractor: PageExtractor = extract_pdf_pages,
    embedding_model: EmbeddingModel | None = None,
    index_writer: IndexWriter = write_faiss_index,
) -> IngestionResult:
    chunk_records, documents_indexed = build_chunk_records(
        pdf_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        page_extractor=page_extractor,
    )
    texts = [record.content for record in chunk_records]

    model = embedding_model or load_embedding_model(model_name)
    vectors = embed_texts(texts, model=model, batch_size=batch_size)
    index_path, metadata_path, manifest_path = ensure_output_paths(index_dir, overwrite)

    index_writer(vectors, index_path)
    write_metadata(chunk_records, metadata_path)
    write_manifest(
        manifest_path=manifest_path,
        source_dir=pdf_dir,
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        documents_indexed=documents_indexed,
        chunks_indexed=len(chunk_records),
        embedding_dimensions=len(vectors[0]),
    )

    return IngestionResult(
        documents_indexed=documents_indexed,
        chunks_indexed=len(chunk_records),
        embedding_dimensions=len(vectors[0]),
        index_path=index_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
    )

