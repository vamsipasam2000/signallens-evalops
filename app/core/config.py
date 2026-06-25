from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = Field(default="SignalLens EvalOps", validation_alias="APP_NAME")
    app_env: str = Field(default="local", validation_alias="APP_ENV")
    log_level: str = Field(default="info", validation_alias="LOG_LEVEL")
    app_version: str = "0.2.0"
    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias="LANGFUSE_HOST",
    )
    default_chunk_size_tokens: int = Field(
        default=180,
        validation_alias="CHUNK_SIZE_TOKENS",
    )
    default_chunk_overlap_tokens: int = Field(
        default=30,
        validation_alias="CHUNK_OVERLAP_TOKENS",
    )
    local_embedding_dimensions: int = Field(
        default=384,
        validation_alias="LOCAL_EMBEDDING_DIMENSIONS",
    )
    vector_backend: str = Field(default="memory", validation_alias="VECTOR_BACKEND")
    embedding_provider: str = Field(default="local-hash", validation_alias="EMBEDDING_PROVIDER")
    chroma_persist_directory: str = Field(
        default=".local/chroma",
        validation_alias="CHROMA_PERSIST_DIRECTORY",
    )
    chroma_collection_name: str = Field(
        default="signallens_chunks",
        validation_alias="CHROMA_COLLECTION_NAME",
    )
    pgvector_dsn: str | None = Field(default=None, validation_alias="PGVECTOR_DSN")
    pgvector_table_name: str = Field(
        default="signallens_chunks",
        validation_alias="PGVECTOR_TABLE_NAME",
    )
    retrieval_leaderboard_path: str = Field(
        default="reports/retrieval_leaderboard.json",
        validation_alias="RETRIEVAL_LEADERBOARD_PATH",
    )
    mlflow_enabled: bool = Field(default=False, validation_alias="MLFLOW_ENABLED")
    mlflow_tracking_uri: str | None = Field(
        default=None,
        validation_alias="MLFLOW_TRACKING_URI",
    )
    mlflow_experiment_name: str = Field(
        default="SignalLens Retrieval Evaluation",
        validation_alias="MLFLOW_EXPERIMENT_NAME",
    )
    trace_storage_backend: str = Field(default="memory", validation_alias="TRACE_STORAGE_BACKEND")
    trace_postgres_dsn: str | None = Field(default=None, validation_alias="TRACE_POSTGRES_DSN")
    trace_postgres_table_name: str = Field(
        default="signallens_traces",
        validation_alias="TRACE_POSTGRES_TABLE_NAME",
    )
    analysis_precision_threshold: float = Field(
        default=0.7,
        validation_alias="ANALYSIS_PRECISION_THRESHOLD",
    )
    analysis_recall_threshold: float = Field(
        default=0.7,
        validation_alias="ANALYSIS_RECALL_THRESHOLD",
    )
    analysis_mrr_threshold: float = Field(
        default=0.7,
        validation_alias="ANALYSIS_MRR_THRESHOLD",
    )
    analysis_ndcg_threshold: float = Field(
        default=0.7,
        validation_alias="ANALYSIS_NDCG_THRESHOLD",
    )
    analysis_latency_threshold_ms: float = Field(
        default=1_000.0,
        validation_alias="ANALYSIS_LATENCY_THRESHOLD_MS",
    )
    analysis_similarity_threshold: float = Field(
        default=0.5,
        validation_alias="ANALYSIS_SIMILARITY_THRESHOLD",
    )
    quality_precision_threshold: float = Field(
        default=0.8,
        validation_alias="QUALITY_PRECISION_THRESHOLD",
    )
    quality_recall_threshold: float = Field(
        default=0.8,
        validation_alias="QUALITY_RECALL_THRESHOLD",
    )
    quality_mrr_threshold: float = Field(
        default=0.8,
        validation_alias="QUALITY_MRR_THRESHOLD",
    )
    quality_ndcg_threshold: float = Field(
        default=0.8,
        validation_alias="QUALITY_NDCG_THRESHOLD",
    )
    quality_latency_threshold_ms: float = Field(
        default=1_000.0,
        validation_alias="QUALITY_LATENCY_THRESHOLD_MS",
    )
    quality_similarity_threshold: float = Field(
        default=0.5,
        validation_alias="QUALITY_SIMILARITY_THRESHOLD",
    )

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
