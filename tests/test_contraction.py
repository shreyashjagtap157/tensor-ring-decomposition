"""Tests for contraction and ring closure modules."""

import torch
import pytest

from tensor_ring_decomposition.core.contraction import (
    compute_emb_precontraction,
    ring_closure,
    ContractionPathCache,
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

    def test_efficient_matches_einsum(self):
        B, R, D = 4, 3, 8
        vocab_result = torch.randn(B, R, R)
        emb_contraction = torch.randn(R, D, R)
        output_eff = ring_closure(vocab_result, emb_contraction, use_efficient=True)
        output_ein = ring_closure(vocab_result, emb_contraction, use_efficient=False)
        assert torch.allclose(output_eff, output_ein, atol=1e-5)

    def test_gradient_flow(self):
        B, R, D = 4, 3, 8
        vocab_result = torch.randn(B, R, R, requires_grad=True)
        emb_contraction = torch.randn(R, D, R, requires_grad=True)
        output = ring_closure(vocab_result, emb_contraction)
        loss = output.sum()
        loss.backward()
        assert vocab_result.grad is not None
        assert emb_contraction.grad is not None


class TestContractionPathCache:
    def test_cache_hit(self):
        ContractionPathCache.clear()
        eq = "bm,bn->bmn"
        t1 = torch.randn(2, 3)
        t2 = torch.randn(2, 4)
        path1, _ = ContractionPathCache.get_path(eq, [t1, t2])
        path2, _ = ContractionPathCache.get_path(eq, [t1, t2])
        assert path1 == path2

    def test_cache_clear(self):
        ContractionPathCache.clear()
        eq = "bm,bn->bmn"
        t1 = torch.randn(2, 3)
        t2 = torch.randn(2, 4)
        ContractionPathCache.get_path(eq, [t1, t2])
        ContractionPathCache.clear()
        path, _ = ContractionPathCache.get_path(eq, [t1, t2])
        assert path is not None

    def test_3d_contraction(self):
        ContractionPathCache.clear()
        eq = "bri,idr->bd"
        t1 = torch.randn(4, 3, 3)
        t2 = torch.randn(3, 8, 3)
        path, _ = ContractionPathCache.get_path(eq, [t1, t2])
        assert path is not None
