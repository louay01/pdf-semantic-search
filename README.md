# PDF Semantic Search

This project indexes local PDF files and exposes a FastAPI endpoint for semantic search. The same Docker image is used for ingestion, testing, and serving the API. No paid API is required.

The PDF files and generated index remain on the host machine. They are mounted into the containers when needed, so the index persists after the ingestion container exits and can be reused by the API container.

## 1. Choose the local PDF folder

Place the PDF files in any local folder. The PDFs must contain extractable text; scanned image-only PDFs require OCR before ingestion.

Run the commands below from the repository root. In PowerShell, set the full path to the PDF folder:

```powershell
$PdfDir = "C:\full\path\to\your\PDFs"
```

The generated index is stored under `data\index` in the repository.

## 2. Build the image

Run this command from the repository root, where the `Dockerfile` is located:

```powershell
docker build -t pdf-semantic-search .
```

## 3. Run ingestion

The PDF folder is mounted read-only at `/pdfs`. The repository's `data` folder is mounted at `/data`, allowing the generated files to persist on the host machine. Docker creates the local `data` folder if necessary, and the ingestion program creates its `index` subfolder.

```powershell
docker run --rm `
  -v "${PdfDir}:/pdfs:ro" `
  -v "${PWD}\data:/data" `
  pdf-semantic-search `
  python -m pdf_semantic_search ingest /pdfs `
  --index-dir /data/index `
  --overwrite
```

The first run downloads the free multilingual embedding model. The command prints the number of indexed documents and chunks when it completes.

## 4. Verify the generated files

```powershell
Get-ChildItem .\data\index
Get-Content .\data\index\manifest.json
```

The index directory should contain:

- `index.faiss`: the vector index
- `metadata.jsonl`: one metadata record per indexed text chunk
- `manifest.json`: the model, chunk settings, and ingestion counts

## 5. Start the API

The API container mounts the same index directory created by ingestion. `PDF_SEARCH_INDEX_DIR` tells the application where that directory is located inside the container.

```powershell
docker run --rm `
  --name pdf-search-api `
  -p 8000:8000 `
  -e PDF_SEARCH_INDEX_DIR=/data/index `
  -v "${PWD}\data:/data:ro" `
  pdf-semantic-search
```

Keep this terminal open while using the API. Interactive API documentation is available at <http://127.0.0.1:8000/docs>.

## 6. Search the PDFs

Run the following in another PowerShell terminal:

```powershell
$Body = @{
  query = "marches publics"
  top_k = 3
} | ConvertTo-Json

$Response = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/search" `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($Body))

$Response | ConvertTo-Json -Depth 6
```

A result contains its similarity score, matching text, PDF name, page number, and chunk number. The health endpoint can be checked separately:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## 7. Rebuild after changing the PDFs

After adding, removing, or replacing PDF files, stop the API with `Ctrl+C` and rerun the ingestion command from step 3. The `--overwrite` option replaces the previous index files. Then start the API again so it loads the rebuilt index.

## Run the tests

The tests use fake embedding and indexing components, so they do not require model downloads:

```powershell
docker run --rm pdf-semantic-search pytest -q
```

## Implementation notes

- PDF text is extracted with `pypdf`.
- Text is split into overlapping chunks before embedding.
- Embeddings use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` on CPU.
- Embeddings are normalized, and FAISS inner-product search therefore acts as cosine similarity.
- Search returns matching chunks rather than full document summaries or page renders.

Possible improvements include OCR support, metadata filters, hybrid keyword/vector retrieval, and incremental indexing.
