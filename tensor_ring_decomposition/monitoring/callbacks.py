"""Training loop callback for monitoring TR embeddings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

from .compression import CompressionTracker
from .quality import QualityGate

if TYPE_CHECKING:
    from ..core.embedding import TensorRingEmbedding

logger = logging.getLogger(__name__)


class TensorRingCallback:
    """Training loop callback for monitoring TR embeddings.

    Compatible with PyTorch Lightning and HuggingFace Trainer.
    """

    def __init__(
        self,
        embedding: "TensorRingEmbedding",
        quality_gate: Optional[QualityGate] = None,
        gauge_fix: bool = True,
        log_interval: int = 1000,
    ):
        self.embedding = embedding
        self.quality_gate = quality_gate
        self.gauge_fix = gauge_fix
        self.tracker = CompressionTracker(embedding, log_interval)

    def on_train_batch_end(self, batch_idx: int, loss: float = 0.0, **kwargs):
        """Called after each training batch."""
        self.tracker.log_metrics(batch_idx)

        if self.gauge_fix:
            self.embedding.cores._apply_gauge_fix()

    def on_validation_end(self, metrics: Dict[str, float], **kwargs):
        """Called after validation. Check quality gate."""
        if self.quality_gate is not None:
            if not self.quality_gate.check(metrics):
                logger.warning("Quality gate triggered - consider rollback")
