"""Automated quality monitoring with rollback trigger."""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class QualityGate:
    """Automated quality monitoring with rollback trigger."""

    def __init__(self, baseline_metrics: Dict[str, float], threshold: float = 0.02):
        """Initialize quality gate.

        Args:
            baseline_metrics: Baseline quality metrics (e.g., from dense model)
            threshold: Maximum allowed relative drop before rollback (default: 2%)
        """
        self.baseline = baseline_metrics
        self.threshold = threshold
        self.triggered = False

    def check(self, current_metrics: Dict[str, float]) -> bool:
        """Returns True if quality is acceptable.

        Checks each metric against baseline. If any drops > threshold,
        returns False and logs the failure.

        Handles edge cases:
        - Zero baseline: uses absolute drop instead of relative
        - Negative baseline: compares absolute change direction
        """
        for key, baseline_value in self.baseline.items():
            if key not in current_metrics:
                continue
            current_value = current_metrics[key]
            abs_base = abs(baseline_value)
            if abs_base > 1e-12:
                drop = (baseline_value - current_value) / abs_base
            else:
                drop = abs(baseline_value - current_value)

            if drop > self.threshold:
                logger.error(
                    f"Quality gate FAILED: {key} dropped {drop:.1%} "
                    f"(baseline={baseline_value:.4f}, current={current_value:.4f})"
                )
                self.triggered = True
                return False

        return True

    def should_rollback(self) -> bool:
        """Whether rollback should be triggered."""
        return self.triggered
