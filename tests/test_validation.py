"""Tests for validation module."""

import pytest
import torch
import torch.nn as nn

from tensor_ring_decomposition.core.embedding import TensorRingEmbedding
from tensor_ring_decomposition.utils.validation import (
    validate_indices,
    validate_compatibility,
)


class TestValidateIndices:
    def test_valid_indices(self):
        validate_indices(torch.tensor([0, 1, 2]), vocab_size=10)

    def test_out_of_bounds(self):
        with pytest.raises(IndexError, match="out of range"):
            validate_indices(torch.tensor([0, 1, 10]), vocab_size=10)

    def test_negative_without_padding(self):
        with pytest.raises(IndexError, match="negative"):
            validate_indices(torch.tensor([-1, 0, 1]), vocab_size=10)

    def test_negative_with_padding(self):
        validate_indices(
            torch.tensor([-1, 0, 1]), vocab_size=10, padding_idx=0
        )

    def test_very_negative_with_padding(self):
        with pytest.raises(IndexError, match="less than -1|padding marker"):
            validate_indices(
                torch.tensor([-2, 0, 1]), vocab_size=10, padding_idx=0
            )


class TestValidateCompatibility:
    def test_compatible(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        linear = nn.Linear(32, 16)
        validate_compatibility(emb, linear)

    def test_incompatible(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        linear = nn.Linear(64, 16)
        with pytest.raises(ValueError, match="expects input dim"):
            validate_compatibility(emb, linear)

    def test_no_linear_layer(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        model = nn.Sequential(nn.ReLU(), nn.ReLU())
        validate_compatibility(emb, model)  # Should not raise
