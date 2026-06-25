# Database Schema

SignalLens 2.0 stores operational metadata in PostgreSQL and vector embeddings in either pgvector or ChromaDB. The in-memory repository used in local tests follows the same logical model.

## PostgreSQL Extensions

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

## Tables

```sql
CREATE TABLE documents (
    document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    content_type TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'api',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    token_count INTEGER NOT NULL DEFAULT 0,
    ingestion_latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    parser_version TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ingested', 'failed')),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE document_chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding_model TEXT NOT NULL,
    embedding VECTOR(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX document_chunks_document_id_idx ON document_chunks(document_id);
CREATE INDEX document_chunks_embedding_hnsw_idx
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE rag_traces (
    trace_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    retrieval_backend TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    llm_model TEXT,
    retrieved_context JSONB NOT NULL DEFAULT '[]'::jsonb,
    token_usage JSONB NOT NULL DEFAULT '{}'::jsonb,
    latencies_ms JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE evaluation_runs (
    eval_run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_name TEXT NOT NULL,
    dataset_path TEXT NOT NULL,
    experiment_name TEXT NOT NULL,
    mlflow_run_id TEXT,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifact_paths JSONB NOT NULL DEFAULT '[]'::jsonb,
    status TEXT NOT NULL CHECK (status IN ('completed', 'failed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE evaluation_items (
    eval_item_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_run_id UUID NOT NULL REFERENCES evaluation_runs(eval_run_id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    ground_truth TEXT NOT NULL,
    generated_answer TEXT,
    retrieved_chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    metric_breakdown JSONB NOT NULL DEFAULT '{}'::jsonb,
    passed BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE retrieval_leaderboard (
    leaderboard_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_run_id UUID NOT NULL REFERENCES evaluation_runs(eval_run_id) ON DELETE CASCADE,
    strategy_name TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    chunk_overlap INTEGER NOT NULL,
    top_k INTEGER NOT NULL,
    precision_at_k DOUBLE PRECISION NOT NULL,
    recall_at_k DOUBLE PRECISION NOT NULL,
    mrr DOUBLE PRECISION NOT NULL,
    ndcg DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

## Storage Strategy

- Use PostgreSQL for documents, chunks, traces, eval runs, eval items, and leaderboards.
- Use pgvector when the deployment should keep vector search inside PostgreSQL.
- Use ChromaDB when the deployment should favor a dedicated local vector index and fast iteration.
- Keep metadata duplicated in both relational rows and vector collections so retrieval results can be filtered without extra joins.
