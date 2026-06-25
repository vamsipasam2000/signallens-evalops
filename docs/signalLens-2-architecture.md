# SignalLens EvalOps 2.0 Architecture

SignalLens EvalOps 2.0 is an AI evaluation and observability platform for RAG, retrieval quality, generation quality, ranking metrics, and experiment tracking.

## Architecture Diagram

```mermaid
flowchart LR
    Client[API Client / Offline Runner] --> API[FastAPI API Layer]
    API --> DI[Dependency Injection]
    API --> RetrieveAPI[POST /retrieve]
    API --> RetrievalEvalAPI[POST /evaluate/retrieval]
    DI --> RAG[RAG Services]
    DI --> Eval[Evaluation Services]
    DI --> Rank[Ranking Services]
    DI --> Obs[Observability Services]
    DI --> Exp[Experiment Tracking]

    RAG --> Parser[Document Parsers<br/>PDF / DOCX / TXT]
    RAG --> Chunker[Chunking Strategy]
    RAG --> Embed[Embedding Providers<br/>SentenceTransformers / OpenAI / Gemini]
    RAG --> Retriever[Retrieval Engine]

    RetrieveAPI --> RetrievalService[RetrievalService]
    RetrievalEvalAPI --> Eval
    Eval --> RetrievalService
    RetrievalService --> Retriever
    RetrievalService --> TraceWriter[Automatic Trace Writer<br/>payload / results / scores / latency / status / errors]

    Retriever --> Chroma[ChromaDB Adapter]
    Retriever --> PGVec[pgvector Adapter]
    Retriever --> LocalVec[Local Deterministic Store]

    Eval --> RAGAS[RAGAS Adapter]
    Eval --> Custom[Custom Eval Framework]
    Eval --> DeepEval[DeepEval Adapter]

    Rank --> Metrics[Precision@K / Recall@K / MRR / NDCG]
    Exp --> MLflow[MLflow Tracking]
    TraceWriter --> Traces
    Obs --> Traces[Trace Repository]

    Chroma --> Storage[(Vector Index)]
    PGVec --> Postgres[(PostgreSQL + pgvector)]
    LocalVec --> Memory[(Local Dev Store)]
    Traces --> Postgres
    MLflow --> Artifacts[(Metrics / Artifacts)]
```

## Runtime Flow

1. **Ingestion:** user uploads PDF, DOCX, or TXT. The parser extracts text, chunker builds overlapping chunks, embedding provider creates vectors, and repositories store document/chunk metadata.
2. **Retrieval:** API receives a query, embeds it, runs top-k similarity search through the selected vector backend, returns chunks with similarity scores and latency, and automatically persists a trace with the request payload, results, scores, status, and errors.
3. **Generation:** LangGraph orchestrates query normalization, retrieval, context assembly, LLM generation, custom scoring, and trace persistence.
4. **Evaluation:** offline and online eval runners compute retrieval, ranking, generation, factuality, faithfulness, answer relevance, context relevance, and hallucination signals. `/evaluate/retrieval` sends each retrieval query through `RetrievalService`, so evaluation and benchmark-grid runs create retrieval traces automatically.
5. **Experiment tracking:** every offline eval and strategy comparison logs parameters, metrics, and artifacts to MLflow.
6. **Observability:** every request carries a trace ID and records retrieval latency, generation latency, evaluation latency, total latency, token usage, failures, and error rates.

## Package Boundaries

- `app/api`: FastAPI routes and request/response schemas.
- `app/core`: settings, dependency wiring, shared errors.
- `app/rag`: ingestion, parsing, chunking, embedding, answer generation orchestration.
- `app/retrieval`: vector-store abstractions and backend adapters.
- `app/ranking`: ranking metrics and leaderboard construction.
- `app/evals`: RAG and LLM evaluation runners, metrics, report writers.
- `app/mlflow`: MLflow experiment logging adapter.
- `app/storage`: repository interfaces, in-memory development repositories, PostgreSQL schema ownership.
- `tests`: unit and integration tests.
- `docs`: architecture, API, database schema, and sprint plans.

## Design Principles

- Keep FastAPI thin. Business logic lives in service classes.
- Use repository interfaces so PostgreSQL/pgvector can replace in-memory dev storage without API changes.
- Use adapter interfaces for ChromaDB, pgvector, SentenceTransformers, OpenAI/Gemini embeddings, RAGAS, DeepEval, and MLflow.
- Prefer deterministic local fallbacks for tests and demos. External providers are opt-in through settings.
- Trace every pipeline step with typed events and latency metrics.
- Store enough metadata to compare chunking, embedding, retrieval, and generation strategies over time.
