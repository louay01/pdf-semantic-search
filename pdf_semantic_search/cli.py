from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embeddings import DEFAULT_MODEL_NAME
from .ingest import ingest_pdfs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local PDF semantic search tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a directory of PDFs into a FAISS index")
    ingest_parser.add_argument("pdf_dir", type=Path, help="Directory containing PDFs")
    ingest_parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/index"),
        help="Directory where index.faiss and metadata files will be written",
    )
    ingest_parser.add_argument("--chunk-size", type=int, default=800, help="Maximum characters per chunk")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=150, help="Overlap between chunks")
    ingest_parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    ingest_parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model used for CPU embeddings",
    )
    ingest_parser.add_argument("--overwrite", action="store_true", help="Replace an existing index in the output folder")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        result = ingest_pdfs(
            args.pdf_dir,
            index_dir=args.index_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            model_name=args.model_name,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "documents_indexed": result.documents_indexed,
                    "chunks_indexed": result.chunks_indexed,
                    "embedding_dimensions": result.embedding_dimensions,
                    "index_path": str(result.index_path.resolve()),
                    "metadata_path": str(result.metadata_path.resolve()),
                    "manifest_path": str(result.manifest_path.resolve()),
                },
                indent=2,
            )
        )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2

