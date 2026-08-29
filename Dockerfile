FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY pdf_semantic_search /app/pdf_semantic_search
COPY tests /app/tests

RUN pip install --no-cache-dir .[dev]

EXPOSE 8000

CMD ["uvicorn", "pdf_semantic_search.api:app", "--host", "0.0.0.0", "--port", "8000"]
