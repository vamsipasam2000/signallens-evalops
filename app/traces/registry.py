from functools import lru_cache

from app.traces.repositories import InMemoryTraceRepository


@lru_cache
def get_in_memory_trace_repository() -> InMemoryTraceRepository:
    return InMemoryTraceRepository()
