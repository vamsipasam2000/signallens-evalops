from functools import lru_cache

from app.analysis.repositories import InMemoryFailureAnalysisRepository


@lru_cache
def get_in_memory_failure_analysis_repository() -> InMemoryFailureAnalysisRepository:
    return InMemoryFailureAnalysisRepository()
