from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from app.core.errors import IngestionValidationError
from app.main import app
from app.rag.chunking import TokenWindowChunker
from app.rag.embeddings import HashEmbeddingProvider
from app.rag.ingestion import IngestionService
from app.rag.parsers import DocxDocumentParser, default_parser_router
from app.storage.registry import get_document_repository
from app.storage.repositories import InMemoryDocumentRepository


@pytest.fixture(autouse=True)
def clear_api_repository() -> None:
    get_document_repository().clear()


def _service(repository: InMemoryDocumentRepository | None = None) -> IngestionService:
    return IngestionService(
        parser_router=default_parser_router(),
        chunker_factory=lambda chunk_size, overlap: TokenWindowChunker(
            chunk_size_tokens=chunk_size,
            overlap_tokens=overlap,
        ),
        embedding_provider=HashEmbeddingProvider(dimensions=32),
        repository=repository or InMemoryDocumentRepository(),
        default_chunk_size_tokens=6,
        default_chunk_overlap_tokens=2,
    )


def _docx_bytes(paragraphs: list[str]) -> bytes:
    xml_paragraphs = "".join(
        f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>" for paragraph in paragraphs
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{xml_paragraphs}</w:body></w:document>"
    )
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
    return payload.getvalue()


def test_token_window_chunker_uses_overlap() -> None:
    chunker = TokenWindowChunker(chunk_size_tokens=4, overlap_tokens=1)

    chunks = chunker.chunk("one two three four five six seven")

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
    ]
    assert chunks[1].metadata["start_token"] == 3


def test_token_window_chunker_rejects_invalid_overlap() -> None:
    with pytest.raises(IngestionValidationError, match="smaller"):
        TokenWindowChunker(chunk_size_tokens=4, overlap_tokens=4)


def test_hash_embedding_provider_is_deterministic_and_normalized() -> None:
    provider = HashEmbeddingProvider(dimensions=16)

    first = provider.embed_documents(["retrieval quality signals"])[0]
    second = provider.embed_documents(["retrieval quality signals"])[0]

    assert first == second
    assert len(first) == 16
    assert round(sum(value * value for value in first), 4) == 1.0


def test_docx_parser_extracts_paragraph_text_with_stdlib_zip_xml() -> None:
    parser = DocxDocumentParser()

    parsed = parser.parse(
        filename="handbook.docx",
        content=_docx_bytes(["First paragraph.", "Second paragraph."]),
        content_type=None,
    )

    assert parsed.text == "First paragraph.\nSecond paragraph."
    assert parsed.parser_version == "docx-zipxml-parser-v1"
    assert parsed.metadata["paragraph_count"] == 2


def test_ingestion_service_stores_document_chunks_and_embeddings() -> None:
    repository = InMemoryDocumentRepository()
    service = _service(repository)

    result = service.ingest_bytes(
        document_name="handbook.txt",
        content=b"alpha beta gamma delta epsilon zeta eta theta",
        content_type="text/plain",
        source="unit-test",
        metadata={"domain": "support"},
    )

    stored_document = repository.get_document(result.document.document_id)
    stored_chunks = repository.list_chunks(result.document.document_id)

    assert stored_document is not None
    assert stored_document.name == "handbook.txt"
    assert stored_document.chunk_count == 2
    assert stored_document.token_count == 8
    assert stored_document.metadata["domain"] == "support"
    assert stored_document.embedding_model == "local-hash-embedding-v1"
    assert len(stored_chunks) == 2
    assert len(stored_chunks[0].embedding) == 32
    assert stored_chunks[0].metadata["domain"] == "support"
    assert stored_chunks[0].metadata["start_token"] == 0


def test_rag_ingest_json_endpoint_returns_ingestion_metrics() -> None:
    client = TestClient(app)

    response = client.post(
        "/v1/rag/ingest",
        json={
            "document_name": "eval-handbook.txt",
            "content_type": "text/plain",
            "content": (
                "RAG evaluation tracks context relevance, answer relevance, and faithfulness."
            ),
            "metadata": {"owner": "eval-team"},
            "chunk_size_tokens": 4,
            "chunk_overlap_tokens": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ingested"
    assert body["name"] == "eval-handbook.txt"
    assert body["content_type"] == "text/plain"
    assert body["chunk_count"] == 3
    assert body["token_count"] == 9
    assert body["embedding_model"] == "local-hash-embedding-v1"
    assert body["parser_version"] == "text-parser-v1"
    assert body["chunker_version"] == "token-window-v1"
    assert body["ingestion_latency_ms"] >= 0
    assert body["chunks"][0]["preview"].startswith("RAG evaluation tracks")


def test_rag_ingest_raw_upload_endpoint_accepts_text_body() -> None:
    client = TestClient(app)

    response = client.post(
        "/v1/rag/ingest/upload?document_name=upload.txt&chunk_size_tokens=4&chunk_overlap_tokens=1",
        content=b"alpha beta gamma delta epsilon",
        headers={"content-type": "text/plain"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ingested"
    assert body["chunk_count"] == 2
    assert body["chunks"][1]["preview"] == "delta epsilon"


def test_rag_ingest_rejects_unsupported_document_type() -> None:
    client = TestClient(app)

    response = client.post(
        "/v1/rag/ingest/upload?document_name=data.csv",
        content=b"a,b,c",
        headers={"content-type": "application/octet-stream"},
    )

    assert response.status_code == 415
