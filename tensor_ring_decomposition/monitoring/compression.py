"""Compression statistics tracking during training."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ..core.embedding import TensorRingEmbedding

logger = logging.getLogger(__name__)


class CompressionTracker:
    """Track compression statistics during training."""

    def __init__(self, embedding: "TensorRingEmbedding", log_interval: int = 1000):
        self.embedding = embedding
        self.log_interval = log_interval
        self.dense_params = embedding.vocab_size * embedding.embedding_dim

    def log_metrics(self, step: int) -> Dict[str, float]:
        """Log compression metrics every log_interval steps."""
        if step % self.log_interval != 0:
            return {}

        spectral_norms = self.embedding.spectral_norms()
        metrics: Dict[str, float] = {
            "tr/compression_ratio": self.embedding.compression_ratio,
            "tr/num_parameters": float(self.embedding.num_parameters),
            "tr/params_saved": float(self.dense_params - self.embedding.num_parameters),
        }
        for k, v in spectral_norms.items():
            metrics[f"tr/spectral_norms/{k}"] = v

        for key, value in metrics.items():
            logger.info(f"{key}: {value}")

        return metrics

    def memory_bytes(self) -> int:
        """Estimate memory footprint in bytes (assuming fp32)."""
        return self.embedding.num_parameters * 4
