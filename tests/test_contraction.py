"""Tests for contraction and ring closure modules."""

import torch
import pytest

from tensor_ring_decomposition.core.contraction import (
    compute_emb_precontraction,
    ring_closure,
)


class TestEmbPrecontraction:
    def test_single_core(self):
        core = torch.randn(8, 4, 4)
        result = compute_emb_precontraction([core])
        assert result.shape == (4, 8, 4)

    def test_two_cores(self):
        c1 = torch.randn(4, 4, 4)
        c2 = torch.randn(8, 4, 4)
        result = compute_emb_precontraction([c1, c2])
        # After contraction: (D1*D2, R, R) permuted to (R, D1*D2, R)
        assert result.shape[0] == 4
        assert result.shape[2] == 4
        assert result.shape[1] == 32  # 4*8

    def test_four_cores(self):
        cores = [torch.randn(4, 4, 4) for _ in range(4)]
        result = compute_emb_precontraction(cores)
        assert result.shape[0] == 4
        assert result.shape[2] == 4
        assert result.shape[1] == 256  # 4^4

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_emb_precontraction([])


class TestRingClosure:
    def test_basic(self):
        B, R, D = 8, 4, 16
        vocab_result = torch.randn(B, R, R)
        emb_contraction = torch.randn(R, D, R)
        output = ring_closure(vocab_result, emb_contraction)
        assert output.shape == (B, D)

    def test_einsum_consistency(self):
        """Ring closure produces consistent results across multiple calls."""
        B, R, D = 4, 3, 8
        vocab_result = torch.randn(B, R, R)
        emb_contraction = torch.randn(R, D, R)
        output1 = ring_closure(vocab_result, emb_contraction)
        output2 = ring_closure(vocab_result, emb_contraction)
        assert torch.allclose(output1, output2, atol=1e-7)

    def test_gradient_flow(self):
        B, R, D = 4, 3, 8
        vocab_result = torch.randn(B, R, R, requires_grad=True)
        emb_contraction = torch.randn(R, D, R, requires_grad=True)
        output = ring_closure(vocab_result, emb_contraction)
        loss = output.sum()
        loss.backward()
        assert vocab_result.grad is not None
        assert emb_contraction.grad is not None


class TestGatherVocabCores:
    def test_raise_oob_on_out_of_bounds(self):
        from tensor_ring_decomposition.core.contraction import gather_vocab_cores
        vocab_cores = [torch.randn(2, 3, 3), torch.randn(2, 3, 3)]
        factor_sizes = [2, 2]
        flat_indices = torch.tensor([4])  # OOB: valid range is [0, 3]
        with pytest.raises(IndexError):
            gather_vocab_cores(flat_indices, vocab_cores, factor_sizes, raise_oob=True)

    def test_clamp_oob_when_not_raising(self):
        from tensor_ring_decomposition.core.contraction import gather_vocab_cores
        vocab_cores = [torch.randn(2, 3, 3), torch.randn(2, 3, 3)]
        factor_sizes = [2, 2]
        flat_indices = torch.tensor([4])  # OOB: valid range is [0, 3]
        # Should not raise; clamps to valid range
        result = gather_vocab_cores(flat_indices, vocab_cores, factor_sizes, raise_oob=False)
        assert result.shape[0] == 1
        assert result.ndim == 3

