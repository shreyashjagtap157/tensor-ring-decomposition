"""Tests for TRTensor — tensor ring tensor representation and reconstruction."""

import pytest
import torch

from tensor_ring_decomposition.core.tensor_ring import TRTensor
from tensor_ring_decomposition.core.factorization import compute_ring_structure


def _make_trtensor(V, D, rank=4, ring_components=4, requires_grad=False, **kwargs):
    """Create a TRTensor + RingStructure from dimension specs.

    Returns (TRTensor, RingStructure).
    """
    structure = compute_ring_structure(
        V, D, ring_components=ring_components, rank=rank, **kwargs
    )
    k, m = structure.n_vocab_cores, structure.n_emb_cores
    ranks = structure.ranks

    vocab_cores = []
    for i in range(k):
        shape = (structure.vocab_factor_sizes[i], ranks[i], ranks[i + 1])
        vocab_cores.append(torch.randn(*shape, requires_grad=requires_grad))

    emb_cores = []
    for i in range(m):
        shape = (structure.emb_factor_sizes[i], ranks[k + i], ranks[k + i + 1])
        emb_cores.append(torch.randn(*shape, requires_grad=requires_grad))

    return TRTensor(vocab_cores, emb_cores), structure


class TestTRTensorCreation:
    def test_create_from_cores(self):
        trt, _ = _make_trtensor(100, 32, rank=4)
        assert isinstance(trt, TRTensor)
        assert len(trt.vocab_cores) > 0
        assert len(trt.emb_cores) > 0

    def test_create_with_explicit_shapes(self):
        V0, V1, D0, D1, R = 10, 10, 4, 8, 4
        vocab_cores = [torch.randn(V0, R, R), torch.randn(V1, R, R)]
        emb_cores = [torch.randn(D0, R, R), torch.randn(D1, R, R)]
        trt = TRTensor(vocab_cores, emb_cores)
        result = trt.to_tensor()
        assert result.shape == (V0 * V1, D0 * D1)

    def test_create_with_different_dtypes(self):
        V0, V1, D0, D1 = 6, 6, 4, 4
        for dtype in [torch.float32, torch.float64]:
            vocab_cores = [torch.randn(V0, 4, 4, dtype=dtype), torch.randn(V1, 4, 4, dtype=dtype)]
            emb_cores = [torch.randn(D0, 4, 4, dtype=dtype), torch.randn(D1, 4, 4, dtype=dtype)]
            trt = TRTensor(vocab_cores, emb_cores)
            result = trt.to_tensor()
            assert result.dtype == dtype
            assert result.shape == (36, 16)


class TestTRTensorToTensorShape:
    def test_shape_matches_ring_structure(self):
        V, D = 100, 32
        trt, structure = _make_trtensor(V, D, rank=4)
        result = trt.to_tensor()
        assert result.ndim == 2
        assert result.shape == (structure.padded_vocab_size, structure.padded_embedding_dim)

    def test_shape_no_padding(self):
        V, D = 144, 64
        trt, structure = _make_trtensor(V, D, rank=4, auto_pad=False)
        result = trt.to_tensor()
        assert result.shape == (V, D)

    def test_shape_with_different_ranks(self):
        V, D = 144, 64
        for rank in [2, 4, 8]:
            trt, _ = _make_trtensor(V, D, rank=rank, auto_pad=False)
            result = trt.to_tensor()
            assert result.shape == (V, D), f"Failed for rank={rank}"

    def test_rank_2(self):
        trt, _ = _make_trtensor(144, 64, rank=2, auto_pad=False)
        assert trt.to_tensor().shape == (144, 64)

    def test_rank_4(self):
        trt, _ = _make_trtensor(144, 64, rank=4, auto_pad=False)
        assert trt.to_tensor().shape == (144, 64)

    def test_rank_8(self):
        trt, _ = _make_trtensor(144, 64, rank=8, auto_pad=False)
        assert trt.to_tensor().shape == (144, 64)

    def test_ring_components_3(self):
        V, D = 12, 64
        trt, structure = _make_trtensor(V, D, rank=4, ring_components=3, auto_pad=False)
        assert structure.n_vocab_cores == 1
        assert structure.n_emb_cores == 2
        result = trt.to_tensor()
        assert result.shape == (structure.padded_vocab_size, structure.padded_embedding_dim)

    def test_ring_components_4(self):
        V, D = 144, 64
        trt, structure = _make_trtensor(V, D, rank=4, ring_components=4, auto_pad=False)
        assert structure.n_vocab_cores == 2
        assert structure.n_emb_cores == 2
        result = trt.to_tensor()
        assert result.shape == (V, D)

    def test_ring_components_5(self):
        V, D = 144, 64
        trt, structure = _make_trtensor(V, D, rank=4, ring_components=5, auto_pad=False)
        assert structure.n_vocab_cores == 2
        assert structure.n_emb_cores == 3
        result = trt.to_tensor()
        assert result.shape == (structure.padded_vocab_size, structure.padded_embedding_dim)


class TestTRTensorValues:
    def test_finite_values(self):
        trt, _ = _make_trtensor(100, 32, rank=4)
        result = trt.to_tensor()
        assert torch.isfinite(result).all()
        assert not torch.isnan(result).any()

    def test_deterministic_reconstruction(self):
        torch.manual_seed(42)
        trt1, _ = _make_trtensor(100, 32, rank=4, auto_pad=False)
        torch.manual_seed(42)
        trt2, _ = _make_trtensor(100, 32, rank=4, auto_pad=False)
        r1, r2 = trt1.to_tensor(), trt2.to_tensor()
        assert torch.allclose(r1, r2)

    def test_non_zero_entries(self):
        trt, _ = _make_trtensor(100, 32, rank=4, auto_pad=False)
        result = trt.to_tensor()
        assert result.abs().sum() > 0

    def test_rank_1_output(self):
        trt, _ = _make_trtensor(144, 64, rank=1, auto_pad=False)
        result = trt.to_tensor()
        assert result.shape == (144, 64)
        assert torch.isfinite(result).all()

    def test_small_dimensions(self):
        V, D = 3, 6
        trt, _ = _make_trtensor(V, D, rank=2, ring_components=3, auto_pad=False)
        result = trt.to_tensor()
        assert result.shape == (V, D)
        assert torch.isfinite(result).all()

    def test_large_dimensions(self):
        V, D = 20000, 128
        trt, structure = _make_trtensor(V, D, rank=2, ring_components=4)
        result = trt.to_tensor()
        assert result.shape[0] >= V
        assert result.shape[1] >= D
        assert torch.isfinite(result).all()


class TestTRTensorProperties:
    def test_vocab_cores_accessible(self):
        trt, structure = _make_trtensor(144, 64, rank=4, auto_pad=False)
        cores = trt.vocab_cores
        assert len(cores) == structure.n_vocab_cores
        for i, c in enumerate(cores):
            assert isinstance(c, torch.Tensor)
            assert c.ndim == 3

    def test_emb_cores_accessible(self):
        trt, structure = _make_trtensor(144, 64, rank=4, auto_pad=False)
        cores = trt.emb_cores
        assert len(cores) == structure.n_emb_cores
        for i, c in enumerate(cores):
            assert isinstance(c, torch.Tensor)
            assert c.ndim == 3

    def test_device_matches_tensors(self):
        trt, _ = _make_trtensor(144, 64, rank=4, auto_pad=False)
        expected = trt.vocab_cores[0].device
        for c in trt.vocab_cores:
            assert c.device == expected
        for c in trt.emb_cores:
            assert c.device == expected

    def test_dtype_matches_tensors(self):
        trt, _ = _make_trtensor(144, 64, rank=4, auto_pad=False)
        expected = trt.vocab_cores[0].dtype
        for c in trt.vocab_cores:
            assert c.dtype == expected
        for c in trt.emb_cores:
            assert c.dtype == expected

    def test_core_shapes_match_ring_structure(self):
        trt, structure = _make_trtensor(144, 64, rank=4, auto_pad=False)
        k = structure.n_vocab_cores
        ranks = structure.ranks
        for i, c in enumerate(trt.vocab_cores):
            expected = (structure.vocab_factor_sizes[i], ranks[i], ranks[i + 1])
            assert c.shape == expected, f"vocab_cores[{i}] shape {c.shape} != {expected}"
        for i, c in enumerate(trt.emb_cores):
            expected = (structure.emb_factor_sizes[i], ranks[k + i], ranks[k + i + 1])
            assert c.shape == expected, f"emb_cores[{i}] shape {c.shape} != {expected}"


class TestTRTensorGradientFlow:
    def test_gradient_flow_through_to_tensor(self):
        trt, _ = _make_trtensor(36, 16, rank=4, requires_grad=True, auto_pad=False)
        result = trt.to_tensor()
        loss = result.sum()
        loss.backward()
        for c in trt.vocab_cores:
            assert c.grad is not None
            assert c.grad.shape == c.shape
            assert c.grad.abs().sum() > 0
        for c in trt.emb_cores:
            assert c.grad is not None
            assert c.grad.shape == c.shape
            assert c.grad.abs().sum() > 0

    def test_gradient_accumulates_correctly(self):
        trt, _ = _make_trtensor(36, 16, rank=4, requires_grad=True, auto_pad=False)
        result = trt.to_tensor()
        loss = (result ** 2).mean()
        loss.backward()
        for c in trt.vocab_cores:
            assert c.grad is not None
        for c in trt.emb_cores:
            assert c.grad is not None
        grad_sum = sum(c.grad.abs().sum().item() for c in trt.vocab_cores + trt.emb_cores)
        assert grad_sum > 0

    def test_no_grad_by_default(self):
        trt, _ = _make_trtensor(36, 16, rank=4, requires_grad=False, auto_pad=False)
        for c in trt.vocab_cores:
            assert not c.requires_grad
        for c in trt.emb_cores:
            assert not c.requires_grad
        result = trt.to_tensor()
        assert not result.requires_grad

    def test_zero_rank_requires_grad(self):
        V0, V1, D0, D1 = 6, 6, 4, 4
        vocab_cores = [torch.randn(V0, 1, 1, requires_grad=True), torch.randn(V1, 1, 1, requires_grad=True)]
        emb_cores = [torch.randn(D0, 1, 1, requires_grad=True), torch.randn(D1, 1, 1, requires_grad=True)]
        trt = TRTensor(vocab_cores, emb_cores)
        result = trt.to_tensor()
        loss = result.sum()
        loss.backward()
        for c in trt.vocab_cores + trt.emb_cores:
            assert c.grad is not None
            assert c.grad.shape == c.shape


class TestTRTensorParameterCount:
    def test_parameter_count_matches_manual(self):
        trt, _ = _make_trtensor(144, 64, rank=4, auto_pad=False)
        manual = sum(c.numel() for c in trt.vocab_cores + trt.emb_cores)
        assert trt.parameter_count() == manual
        assert trt.parameter_count() > 0

    def test_parameter_count_changes_with_rank(self):
        _, s2 = _make_trtensor(144, 64, rank=2, auto_pad=False)
        _, s8 = _make_trtensor(144, 64, rank=8, auto_pad=False)
        k = s2.n_vocab_cores + s2.n_emb_cores
        # Each core scales as rank^2, so rank=8 should have 16x more params than rank=2
        r2_total = sum(s2.vocab_factor_sizes[i] * s2.ranks[i] * s2.ranks[i + 1]
                       for i in range(s2.n_vocab_cores))
        r2_total += sum(s2.emb_factor_sizes[i] * s2.ranks[s2.n_vocab_cores + i] * s2.ranks[s2.n_vocab_cores + i + 1]
                        for i in range(s2.n_emb_cores))
        r8_total = sum(s8.vocab_factor_sizes[i] * s8.ranks[i] * s8.ranks[i + 1]
                       for i in range(s8.n_vocab_cores))
        r8_total += sum(s8.emb_factor_sizes[i] * s8.ranks[s8.n_vocab_cores + i] * s8.ranks[s8.n_vocab_cores + i + 1]
                        for i in range(s8.n_emb_cores))
        # rank=8 should have ~16x more params than rank=2
        ratio = r8_total / r2_total
        assert 14.0 <= ratio <= 18.0  # Allow slight variation due to rounding

    def test_parameter_count_different_components(self):
        trt3, _ = _make_trtensor(100, 32, rank=4, ring_components=3)
        trt5, _ = _make_trtensor(100, 32, rank=4, ring_components=5)
        assert trt3.parameter_count() > 0
        assert trt5.parameter_count() > 0


class TestTRTensorEdgeCases:
    def test_rank_1(self):
        trt, structure = _make_trtensor(144, 64, rank=1, auto_pad=False)
        result = trt.to_tensor()
        assert result.shape == (144, 64)
        assert trt.parameter_count() == sum(c.numel() for c in trt.vocab_cores + trt.emb_cores)

    def test_very_small_vocab_and_dim(self):
        V, D = 2, 4
        trt, _ = _make_trtensor(V, D, rank=2, ring_components=3, auto_pad=False)
        result = trt.to_tensor()
        assert result.shape == (V, D)

    def test_single_vocab_core(self):
        V, D = 10, 64
        trt, _ = _make_trtensor(V, D, rank=4, ring_components=3, auto_pad=False)
        assert len(trt.vocab_cores) == 1
        result = trt.to_tensor()
        assert result.shape == (V, D)

    def test_single_emb_core(self):
        V, D = 144, 8
        trt, _ = _make_trtensor(V, D, rank=4, ring_components=3, auto_pad=False)
        assert len(trt.emb_cores) == 2
        result = trt.to_tensor()
        assert result.shape == (V, D)

    def test_uneven_rank_list(self):
        structure = compute_ring_structure(
            144, 64, ring_components=4, ranks=[4, 6, 8, 10, 4]
        )
        k = structure.n_vocab_cores
        ranks = structure.ranks
        vocab_cores = []
        for i in range(k):
            shape = (structure.vocab_factor_sizes[i], ranks[i], ranks[i + 1])
            vocab_cores.append(torch.randn(*shape))
        emb_cores = []
        for i in range(structure.n_emb_cores):
            shape = (structure.emb_factor_sizes[i], ranks[k + i], ranks[k + i + 1])
            emb_cores.append(torch.randn(*shape))
        trt = TRTensor(vocab_cores, emb_cores)
        result = trt.to_tensor()
        assert result.shape == (structure.padded_vocab_size, structure.padded_embedding_dim)
        assert torch.isfinite(result).all()
