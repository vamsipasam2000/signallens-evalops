from functools import lru_cache

from app.quality.repositories import InMemoryQualityGateRepository


@lru_cache
def get_in_memory_quality_gate_repository() -> InMemoryQualityGateRepository:
    return InMemoryQualityGateRepository()
