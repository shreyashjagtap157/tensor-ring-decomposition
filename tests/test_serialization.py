"""Tests for serialization module."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from tensor_ring_decomposition.core.embedding import TensorRingEmbedding
from tensor_ring_decomposition.utils.serialization import save, load, SecurityError


class TestSerialization:
    def test_roundtrip(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        indices = torch.tensor([0, 1, 2])
        original_output = emb(indices)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_model")
            save(emb, path)

            loaded = load(path)
            loaded_output = loaded(indices)

            assert torch.allclose(original_output, loaded_output, atol=1e-6)

    def test_manifest_created(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_model")
            save(emb, path)

            manifest_path = Path(path).with_suffix(".json")
            assert manifest_path.exists()

            manifest = json.loads(manifest_path.read_text())
            assert manifest["schema_version"] == "1.0"
            assert manifest["tr_config"]["vocab_size"] == 100

    def test_safetensors_created(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_model")
            save(emb, path)

            st_path = Path(path).with_suffix(".safetensors")
            assert st_path.exists()

    def test_hash_verification(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        secret = b"test_secret_key"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_model")
            save(emb, path, secret_key=secret)

            loaded = load(path, secret_key=secret)
            assert loaded.vocab_size == 100

    def test_tampered_checkpoint(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_model")
            save(emb, path)

            # Tamper with the safetensors file
            st_path = Path(path).with_suffix(".safetensors")
            with open(st_path, "ab") as f:
                f.write(b"tampered")

            with pytest.raises(SecurityError):
                load(path)

    def test_missing_manifest(self):
        with pytest.raises(FileNotFoundError):
            load("/nonexistent/path")

    def test_extra_metadata(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_model")
            save(emb, path, extra_metadata={"author": "test"})

            manifest_path = Path(path).with_suffix(".json")
            manifest = json.loads(manifest_path.read_text())
            assert manifest["author"] == "test"

    def test_explicit_ranks_roundtrip(self):
        """Verify serialization roundtrip with explicit ranks (not scalar rank)."""
        emb = TensorRingEmbedding(100, 32, rank=None, ranks=[4, 8, 8, 8, 4])
        indices = torch.tensor([0, 1, 2])
        original_output = emb(indices)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_ranks_model")
            save(emb, path)

            loaded = load(path)
            loaded_output = loaded(indices)

            assert loaded.structure.ranks == [4, 8, 8, 8, 4]
            assert torch.allclose(original_output, loaded_output, atol=1e-6)
