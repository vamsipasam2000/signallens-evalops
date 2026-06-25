from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.schemas import RAGChunkResponse, RAGIngestRequest, RAGIngestResponse
from app.core.dependencies import get_rag_ingestion_service
from app.core.errors import (
    DependencyUnavailableError,
    IngestionValidationError,
    UnsupportedDocumentTypeError,
)
from app.rag.ingestion import IngestionService
from app.rag.models import IngestionResult

router = APIRouter(prefix="/v1/rag", tags=["rag"])


@router.post("/ingest", response_model=RAGIngestResponse)
def ingest_document(
    request: RAGIngestRequest,
    service: Annotated[IngestionService, Depends(get_rag_ingestion_service)],
) -> RAGIngestResponse:
    result = _run_ingestion(
        service=service,
        document_name=request.document_name,
        content=request.content.encode("utf-8"),
        content_type=request.content_type,
        source=request.source,
        metadata=request.metadata,
        chunk_size_tokens=request.chunk_size_tokens,
        chunk_overlap_tokens=request.chunk_overlap_tokens,
    )
    return _build_ingest_response(result)


@router.post("/ingest/upload", response_model=RAGIngestResponse)
async def ingest_raw_upload(
    request: Request,
    service: Annotated[IngestionService, Depends(get_rag_ingestion_service)],
    document_name: str = Query(..., min_length=1, max_length=255),
    source: str = Query(default="api", min_length=1, max_length=100),
    content_type: str | None = Query(default=None, max_length=200),
    chunk_size_tokens: int | None = Query(default=None, ge=1, le=4_000),
    chunk_overlap_tokens: int | None = Query(default=None, ge=0, le=2_000),
) -> RAGIngestResponse:
    body = await request.body()
    result = _run_ingestion(
        service=service,
        document_name=document_name,
        content=body,
        content_type=content_type or request.headers.get("content-type"),
        source=source,
        metadata={"upload_mode": "raw-body"},
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )
    return _build_ingest_response(result)


def _run_ingestion(
    *,
    service: IngestionService,
    document_name: str,
    content: bytes,
    content_type: str | None,
    source: str,
    metadata: dict[str, object],
    chunk_size_tokens: int | None,
    chunk_overlap_tokens: int | None,
) -> IngestionResult:
    try:
        return service.ingest_bytes(
            document_name=document_name,
            content=content,
            content_type=content_type,
            source=source,
            metadata=metadata,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except DependencyUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except IngestionValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _build_ingest_response(result: IngestionResult) -> RAGIngestResponse:
    document = result.document
    return RAGIngestResponse(
        document_id=document.document_id,
        name=document.name,
        content_type=document.content_type,
        status=document.status,
        chunk_count=document.chunk_count,
        token_count=document.token_count,
        embedding_model=document.embedding_model,
        parser_version=document.parser_version,
        chunker_version=document.chunker_version,
        ingestion_latency_ms=document.ingestion_latency_ms,
        chunks=[
            RAGChunkResponse(
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                preview=_preview(chunk.text),
            )
            for chunk in result.chunks
        ],
        message="Document ingested, chunked, embedded, and stored.",
    )


def _preview(text: str, limit: int = 140) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."
