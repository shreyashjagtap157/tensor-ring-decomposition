"""Tests for TensorRingEmbedding module."""

import pytest
import torch
import torch.nn as nn

from tensor_ring_decomposition.core.embedding import TensorRingEmbedding


class TestTensorRingEmbedding:
    def test_basic_forward(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        indices = torch.tensor([0, 1, 2, 3])
        output = emb(indices)
        assert output.shape == (4, 32)

    def test_batched_forward(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        indices = torch.tensor([[0, 1], [2, 3]])
        output = emb(indices)
        assert output.shape == (2, 2, 32)

    def test_gradient_flow(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        indices = torch.tensor([0, 1, 2])
        output = emb(indices)
        loss = output.sum()
        loss.backward()
        for param in emb.parameters():
            assert param.grad is not None

    def test_eval_mode_cache(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        emb.to_eval_mode()
        assert emb._cache_valid
        assert emb._emb_cache is not None

    def test_train_mode_clears_cache(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        emb.to_eval_mode()
        assert emb._cache_valid
        emb.train_mode()
        assert not emb._cache_valid
        assert emb._emb_cache is None

    def test_config(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        cfg = emb.config()
        assert cfg["vocab_size"] == 100
        assert cfg["embedding_dim"] == 32
        assert cfg["rank"] == 4
        assert cfg["max_seq_len"] is None

    def test_config_with_max_seq_len(self):
        emb = TensorRingEmbedding(100, 32, rank=4, max_seq_len=512)
        cfg = emb.config()
        assert cfg["max_seq_len"] == 512

    def test_compression_ratio(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        ratio = emb.compression_ratio
        assert ratio > 1.0

    def test_num_parameters(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        assert emb.num_parameters > 0

    def test_from_compression_ratio(self):
        emb = TensorRingEmbedding.from_compression_ratio(100, 32, ratio=5)
        assert emb.compression_ratio > 1.0

    def test_from_target_params(self):
        emb = TensorRingEmbedding.from_target_params(100, 32, params=1000)
        assert emb.num_parameters > 0

    def test_from_pretrained(self):
        matrix = torch.randn(100, 32)
        emb = TensorRingEmbedding.from_pretrained(matrix, rank=4)
        assert emb.vocab_size == 100
        assert emb.embedding_dim == 32

    def test_padding_idx(self):
        emb = TensorRingEmbedding(100, 32, rank=4, padding_idx=0)
        indices = torch.tensor([0, 1, 2])
        output = emb(indices)
        assert output.shape == (3, 32)

    def test_fp16_rejection(self):
        emb = TensorRingEmbedding(100, 32, rank=4, dtype=torch.float16)
        indices = torch.tensor([0, 1])
        with pytest.raises(TypeError, match="fp16"):
            emb(indices)

    def test_invalid_compression_config_multiple(self):
        with pytest.raises(ValueError, match="Exactly one"):
            TensorRingEmbedding(100, 32, rank=4, target_compression=2.0)

    def test_invalid_compression_config_none(self):
        with pytest.raises(ValueError, match="Exactly one"):
            TensorRingEmbedding(100, 32)

    def test_invalid_target_compression(self):
        with pytest.raises(ValueError, match="target_compression"):
            TensorRingEmbedding(100, 32, target_compression=0.5)

    def test_invalid_vocab_size(self):
        with pytest.raises(ValueError, match="positive"):
            TensorRingEmbedding(0, 32, rank=4)

    def test_invalid_embedding_dim(self):
        with pytest.raises(ValueError, match="positive"):
            TensorRingEmbedding(100, 0, rank=4)

    def test_spectral_norms(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        norms = emb.spectral_norms()
        assert len(norms) > 0

    def test_reconstruct(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        matrix = emb.reconstruct()
        assert matrix.shape == (100, 32)

    def test_reconstruction_error(self):
        emb = TensorRingEmbedding(100, 32, rank=8)
        original = torch.randn(100, 32)
        error = emb.reconstruction_error(original)
        assert isinstance(error, float)
        assert error >= 0

    def test_rank_property(self):
        emb = TensorRingEmbedding(100, 32, rank=6)
        assert emb.rank == 6
