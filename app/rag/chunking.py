from __future__ import annotations

from typing import Any

from app.core.errors import IngestionValidationError
from app.rag.models import TextChunk


class TokenWindowChunker:
    version = "token-window-v1"

    def __init__(self, *, chunk_size_tokens: int, overlap_tokens: int) -> None:
        if chunk_size_tokens <= 0:
            raise IngestionValidationError("chunk_size_tokens must be greater than zero")
        if overlap_tokens < 0:
            raise IngestionValidationError("chunk_overlap_tokens cannot be negative")
        if overlap_tokens >= chunk_size_tokens:
            raise IngestionValidationError(
                "chunk_overlap_tokens must be smaller than chunk_size_tokens"
            )

        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text: str, *, metadata: dict[str, Any] | None = None) -> list[TextChunk]:
        tokens = text.split()
        if not tokens:
            raise IngestionValidationError("cannot chunk empty text")

        chunks: list[TextChunk] = []
        start = 0
        index = 0
        while start < len(tokens):
            end = min(start + self.chunk_size_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(
                TextChunk(
                    chunk_index=index,
                    text=" ".join(chunk_tokens),
                    token_count=len(chunk_tokens),
                    metadata={
                        **(metadata or {}),
                        "start_token": start,
                        "end_token": end,
                    },
                )
            )
            if end == len(tokens):
                break
            start = end - self.overlap_tokens
            index += 1

        return chunks
