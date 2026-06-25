# API Design

## Health

`GET /health`

Returns service status and version.

## Legacy Trust-Safety Workflow

`POST /v1/analyze`

Runs the original deterministic trust-safety workflow. This remains as a baseline eval workflow while RAG features are added.

## RAG Ingestion

`POST /v1/rag/ingest`

JSON ingestion endpoint for tests, demos, and batch ingestion.

Request:

```json
{
  "document_name": "handbook.txt",
  "content_type": "text/plain",
  "content": "Document text...",
  "source": "api",
  "metadata": {"team": "support"},
  "chunk_size_tokens": 180,
  "chunk_overlap_tokens": 30
}
```

Response:

```json
{
  "document_id": "uuid",
  "name": "handbook.txt",
  "content_type": "text/plain",
  "status": "ingested",
  "chunk_count": 4,
  "token_count": 612,
  "embedding_model": "local-hash-embedding-v1",
  "parser_version": "text-parser-v1",
  "chunker_version": "token-window-v1",
  "ingestion_latency_ms": 12.4,
  "chunks": [
    {
      "chunk_id": "uuid",
      "chunk_index": 0,
      "token_count": 180,
      "preview": "..."
    }
  ]
}
```

`POST /v1/rag/ingest/upload?document_name=handbook.txt`

Raw-body upload endpoint for PDF, DOCX, or TXT. The file type is inferred from `Content-Type` and filename.

## Retrieval

`POST /retrieve`

Planned Sprint 2 endpoint for top-k retrieval, similarity scores, metadata filters, backend name, and retrieval latency.

## RAG Answer Generation

`POST /v1/rag/answer`

Planned Sprint 3 endpoint for question-answering with retrieved context, token usage, latencies, trace ID, and quality scores.

## Offline Evaluation

`POST /v1/evals/rag/run`

Planned Sprint 4-6 endpoint for JSONL dataset evaluation, ranking metrics, generation metrics, MLflow logging, and report artifacts.
