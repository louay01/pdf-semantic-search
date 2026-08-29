from __future__ import annotations


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = normalize_whitespace(text)
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            boundary = normalized.rfind(" ", start + 1, end + 1)
            if boundary > start:
                end = boundary

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(end - chunk_overlap, 0)
        while next_start > 0 and not normalized[next_start - 1].isspace():
            next_start -= 1
        while next_start < text_length and normalized[next_start].isspace():
            next_start += 1
        if next_start <= start:
            next_start = end
            while next_start < text_length and normalized[next_start].isspace():
                next_start += 1
        start = next_start

    return chunks
