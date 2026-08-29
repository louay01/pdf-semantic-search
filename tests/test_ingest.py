import json
from pathlib import Path

from pdf_semantic_search.ingest import ingest_pdfs


class FakeEmbeddingModel:
    def encode(self, sentences, **_: object):
        return [[float(index + 1), float(len(sentence))] for index, sentence in enumerate(sentences)]


def test_ingest_pdfs_writes_index_metadata_and_manifest(tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "doc-a.pdf").write_bytes(b"%PDF-1.4")
    (pdf_dir / "doc-b.pdf").write_bytes(b"%PDF-1.4")

    def fake_page_extractor(pdf_path: Path):
        if pdf_path.name == "doc-a.pdf":
            return [(1, "bonjour a tous depuis paris"), (2, "second page text")]
        return [(1, "hola mundo desde madrid")]

    def fake_index_writer(vectors: list[list[float]], index_path: Path):
        serialized = "\n".join(",".join(str(value) for value in row) for row in vectors)
        index_path.write_text(serialized, encoding="utf-8")

    result = ingest_pdfs(
        pdf_dir,
        index_dir=tmp_path / "index",
        chunk_size=14,
        chunk_overlap=4,
        overwrite=False,
        page_extractor=fake_page_extractor,
        embedding_model=FakeEmbeddingModel(),
        index_writer=fake_index_writer,
    )

    assert result.documents_indexed == 2
    assert result.chunks_indexed == 8
    assert result.embedding_dimensions == 2
    assert result.index_path.exists()
    assert result.metadata_path.exists()
    assert result.manifest_path.exists()

    metadata_lines = result.metadata_path.read_text(encoding="utf-8").splitlines()
    parsed_metadata = [json.loads(line) for line in metadata_lines]
    assert parsed_metadata[0]["document_name"] == "doc-a.pdf"
    assert parsed_metadata[0]["page_number"] == 1
    assert parsed_metadata[0]["chunk_index"] == 0
    assert parsed_metadata[0]["content"] == "bonjour a tous"
    assert parsed_metadata[-1]["document_name"] == "doc-b.pdf"

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["documents_indexed"] == 2
    assert manifest["chunks_indexed"] == 8
    assert manifest["embedding_dimensions"] == 2


def test_ingest_pdfs_requires_overwrite_for_existing_outputs(tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "doc-a.pdf").write_bytes(b"%PDF-1.4")

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_text("existing", encoding="utf-8")

    def fake_page_extractor(_: Path):
        return [(1, "simple page text")]

    def fake_index_writer(vectors: list[list[float]], index_path: Path):
        index_path.write_text(str(vectors), encoding="utf-8")

    try:
        ingest_pdfs(
            pdf_dir,
            index_dir=index_dir,
            page_extractor=fake_page_extractor,
            embedding_model=FakeEmbeddingModel(),
            index_writer=fake_index_writer,
        )
    except FileExistsError as exc:
        assert "--overwrite" in str(exc)
    else:
        raise AssertionError("Expected FileExistsError")
