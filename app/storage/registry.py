from functools import lru_cache

from app.storage.repositories import InMemoryDocumentRepository


@lru_cache
def get_document_repository() -> InMemoryDocumentRepository:
    return InMemoryDocumentRepository()
