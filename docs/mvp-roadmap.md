# MVP Roadmap

The 2.0 roadmap is organized around interview-worthy vertical slices. Each slice must leave behind runnable APIs, tests, docs, and implementation details that can be discussed confidently.

## MVP 1: RAG Ingestion Foundation

- Parser router for TXT, DOCX, and PDF.
- Token-aware chunking with overlap.
- Embedding-provider interface with deterministic local fallback and SentenceTransformers adapter.
- Document and chunk repositories.
- Ingestion API with chunk count, token count, embedding model, and latency.
- Unit and integration tests.

## MVP 2: Retrieval Backends

- `/retrieve` API.
- Top-k similarity search.
- Metadata filtering.
- ChromaDB adapter.
- pgvector adapter.
- Retrieval latency instrumentation.

## MVP 3: RAG Generation Workflow

- LangGraph RAG workflow.
- Query normalization, retrieval, context assembly, generation, output scoring.
- LangChain LLM abstraction.
- Token usage and latency tracking.
- Full trace persistence.

## MVP 4: LLM/RAG Evaluation Engine

- Custom factuality, faithfulness, answer relevance, context relevance, and hallucination metrics.
- RAGAS adapter.
- DeepEval adapter if it adds useful signal without duplicate complexity.
- Eval score storage and pass/fail thresholds.

## MVP 5: Ranking Evaluation and Leaderboards

- Precision@K, Recall@K, MRR, NDCG.
- Compare retrieval backends, embedding models, chunk sizes, chunk overlap, and reranking strategies.
- Leaderboard API and CSV/JSON report generation.

## MVP 6: MLflow Experiment Tracking

- Log chunking, retrieval, embedding, and LLM parameters.
- Log retrieval, ranking, generation, and latency metrics.
- Log HTML/CSV/JSON reports as artifacts.
- Comparison dashboard docs.

## MVP 7: Offline Evaluation Suite

- JSONL dataset loader.
- Batch runner for retrieval, generation, and ranking.
- HTML, CSV, and JSON reports.
- Dockerized local run with PostgreSQL and MLflow.

## MVP 8: Production Hardening

- PostgreSQL repository implementation.
- Alembic migrations.
- Docker Compose with API, PostgreSQL, ChromaDB, and MLflow.
- 80% test coverage target.
- MkDocs site with architecture, API, schema, and runbooks.
