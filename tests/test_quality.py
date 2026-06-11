"""Tests for quality monitoring module."""

from tensor_ring_decomposition.monitoring.quality import QualityGate
from tensor_ring_decomposition.monitoring.compression import CompressionTracker
from tensor_ring_decomposition.core.embedding import TensorRingEmbedding


class TestQualityGate:
    def test_pass(self):
        gate = QualityGate({"accuracy": 0.95}, threshold=0.02)
        result = gate.check({"accuracy": 0.94})
        assert result is True
        assert not gate.should_rollback()

    def test_fail(self):
        gate = QualityGate({"accuracy": 0.95}, threshold=0.02)
        result = gate.check({"accuracy": 0.90})
        assert result is False
        assert gate.should_rollback()

    def test_missing_metric(self):
        gate = QualityGate({"accuracy": 0.95, "loss": 0.1}, threshold=0.02)
        result = gate.check({"accuracy": 0.94})
        assert result is True

    def test_multiple_metrics(self):
        gate = QualityGate(
            {"accuracy": 0.95, "loss": 0.1}, threshold=0.02
        )
        result = gate.check({"accuracy": 0.94, "loss": 0.0985})
        assert result is True

    def test_one_metric_fails(self):
        gate = QualityGate(
            {"accuracy": 0.95, "loss": 0.1}, threshold=0.02
        )
        result = gate.check({"accuracy": 0.88, "loss": 0.09})
        assert result is False


class TestCompressionTracker:
    def test_log_metrics(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        tracker = CompressionTracker(emb, log_interval=1)
        metrics = tracker.log_metrics(0)
        assert "tr/compression_ratio" in metrics
        assert "tr/num_parameters" in metrics

    def test_log_interval(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        tracker = CompressionTracker(emb, log_interval=10)
        metrics = tracker.log_metrics(5)
        assert metrics == {}

    def test_memory_bytes(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        tracker = CompressionTracker(emb)
        mem = tracker.memory_bytes()
        assert mem > 0
