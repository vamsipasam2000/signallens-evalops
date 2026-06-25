# Sprint Plan

## Sprint 0: Architecture and Repo Shape

Deliverables:
- 2.0 architecture document.
- Database schema.
- MVP roadmap.
- Sprint plan.
- Target folder structure under `app/`.

Acceptance criteria:
- A reviewer can explain the platform architecture from docs alone.
- Code package boundaries match the architecture.

## Sprint 1: RAG Ingestion Pipeline

Deliverables:
- TXT parser and adapter hooks for PDF/DOCX.
- Chunking strategy with overlap.
- Deterministic embedding fallback.
- Document/chunk repository interface and in-memory implementation.
- API endpoint for JSON and raw upload ingestion.
- Unit/integration tests.

Acceptance criteria:
- Ingesting a document returns document ID, chunk count, token count, embedding model, parser version, chunker version, and ingestion latency.
- Chunk metadata and embeddings are stored through repository interfaces.
- Tests run without external services.

## Sprint 2: Retrieval API

Deliverables:
- Similarity search API.
- Metadata filters.
- In-memory, ChromaDB, and pgvector vector-store adapters.
- Retrieval traces and latency metrics.

Acceptance criteria:
- `/retrieve` returns top-k chunks, scores, backend name, and latency.
- Precision-focused tests validate deterministic ranking.

## Sprint 3: LangGraph RAG Generation

Deliverables:
- LangGraph RAG workflow.
- LangChain LLM adapter.
- Context assembly.
- Token usage accounting.
- Trace repository.

Acceptance criteria:
- `/rag/answer` returns answer, retrieved context, token usage, latencies, and trace ID.
- Generation can run with local deterministic LLM fallback and real provider settings.

## Sprint 4: Evaluation Engine

Deliverables:
- Custom RAG metrics.
- RAGAS adapter.
- Evaluation result storage.
- Pass/fail thresholds.

Acceptance criteria:
- Each answer has factuality, faithfulness, answer relevance, context relevance, and hallucination signals.
- Metric breakdown is stored and returned by API.

## Sprint 5: Ranking and Experiment Tracking

Deliverables:
- Precision@K, Recall@K, MRR, NDCG.
- Retrieval strategy comparison runner.
- MLflow logging adapter.
- Leaderboard API.

Acceptance criteria:
- Offline run compares at least two chunking or retrieval strategies.
- MLflow logs params, metrics, and artifacts.

## Sprint 6: Offline Reports and Docs

Deliverables:
- JSONL offline runner.
- HTML, CSV, and JSON reports.
- MkDocs documentation.
- Docker Compose.

Acceptance criteria:
- One command runs an offline eval and writes all reports.
- Docs explain RAG, embeddings, vector DBs, ranking metrics, eval metrics, and experiment tracking.
