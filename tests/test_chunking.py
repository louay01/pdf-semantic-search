from pdf_semantic_search.chunking import chunk_text, normalize_whitespace


def test_normalize_whitespace_collapses_runs():
    assert normalize_whitespace("Bonjour\n\n   tout   le monde\t") == "Bonjour tout le monde"


def test_chunk_text_returns_overlapping_chunks_without_empty_values():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"

    chunks = chunk_text(text, chunk_size=18, chunk_overlap=5)

    assert chunks == [
        "alpha beta gamma",
        "gamma delta",
        "delta epsilon zeta",
        "epsilon zeta eta",
        "zeta eta theta",
        "theta iota kappa",
        "kappa lambda",
    ]


def test_chunk_text_validates_overlap():
    try:
        chunk_text("sample", chunk_size=10, chunk_overlap=10)
    except ValueError as exc:
        assert "chunk_overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
