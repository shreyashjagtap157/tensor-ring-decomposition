"""Shared pytest fixtures and configuration."""

import pytest

from tensor_ring_decomposition.core.contraction import ContractionPathCache
from tensor_ring_decomposition.models.registry import ModelRegistry
from tensor_ring_decomposition.models.profiles import BUILTIN_PROFILES


@pytest.fixture(autouse=True)
def _clear_contraction_cache():
    """Clear the global contraction path cache before each test."""
    ContractionPathCache.clear()


@pytest.fixture(autouse=True)
def _ensure_registry_loaded():
    """Ensure all built-in profiles are registered before each test."""
    if len(ModelRegistry.list_all()) < 60:
        for _p in BUILTIN_PROFILES:
            ModelRegistry.register(_p)
