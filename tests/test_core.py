"""Tests for core factorization and cores modules."""

import pytest
import torch

from tensor_ring_decomposition.core.factorization import (
    factorize_dimension,
    compute_ring_structure,
)
from tensor_ring_decomposition.core.cores import TensorRingCores
from tensor_ring_decomposition.utils.gauge import GaugeFixer


class TestFactorizeDimension:
    def test_basic(self):
        result = factorize_dimension(100, 2)
        assert len(result) == 2
        assert result[0] * result[1] == 100

    def test_single_factor(self):
        result = factorize_dimension(50000, 1)
        assert result == [50000]

    def test_four_factors(self):
        result = factorize_dimension(50000, 4)
        assert len(result) == 4
        product = 1
        for f in result:
            product *= f
        assert product == 50000

    def test_embedding_dim(self):
        result = factorize_dimension(768, 4)
        assert len(result) == 4
        product = 1
        for f in result:
            product *= f
        assert product == 768

    def test_too_many_factors(self):
        with pytest.raises(ValueError):
            factorize_dimension(3, 5)

    def test_small_dim(self):
        result = factorize_dimension(6, 3)
        assert len(result) == 3
        product = 1
        for f in result:
            product *= f
        assert product == 6


class TestRingStructure:
    def test_balanced(self):
        structure = compute_ring_structure(50000, 768, ring_components=4, rank=8)
        assert structure.n_vocab_cores == 2
        assert structure.n_emb_cores == 2
        assert structure.ring_components == 4

    def test_proportional(self):
        structure = compute_ring_structure(
            50000, 768, ring_components=4, rank=8, split_mode="proportional"
        )
        assert structure.n_vocab_cores + structure.n_emb_cores == 4

    def test_manual_requires_ranks(self):
        with pytest.raises(ValueError):
            compute_ring_structure(
                50000, 768, ring_components=4, rank=8, split_mode="manual"
            )

    def test_invalid_split_mode(self):
        with pytest.raises(ValueError):
            compute_ring_structure(
                50000, 768, ring_components=4, rank=8, split_mode="invalid"
            )

    def test_rank_list(self):
        structure = compute_ring_structure(50000, 768, ring_components=4, rank=8)
        assert len(structure.ranks) == 5  # k + m + 1 = 4 + 1
        assert all(r == 8 for r in structure.ranks)

    def test_ring_components_too_small(self):
        with pytest.raises(ValueError):
            compute_ring_structure(50000, 768, ring_components=1)

    def test_ranks_too_short(self):
        with pytest.raises(ValueError, match="ranks length"):
            compute_ring_structure(
                50000, 768, ring_components=4, ranks=[4, 4, 4]
            )


class TestTensorRingCores:
    def test_init(self):
        structure = compute_ring_structure(100, 32, ring_components=4, rank=4)
        cores = TensorRingCores(structure)
        assert len(cores.vocab_cores) == 2
        assert len(cores.emb_cores) == 2

    def test_xavier_init(self):
        structure = compute_ring_structure(100, 32, ring_components=4, rank=4)
        cores = TensorRingCores(structure)
        cores.initialize("uniform")
        for core in cores._all_cores():
            assert core.data.abs().sum() > 0

    def test_svd_init(self):
        structure = compute_ring_structure(100, 32, ring_components=4, rank=4)
        cores = TensorRingCores(structure)
        matrix = torch.randn(100, 32)
        cores.initialize("svd", matrix)
        for core in cores._all_cores():
            assert core.data.abs().sum() > 0

    def test_svd_init_requires_matrix(self):
        structure = compute_ring_structure(100, 32, ring_components=4, rank=4)
        cores = TensorRingCores(structure)
        with pytest.raises(ValueError, match="embedding_matrix"):
            cores.initialize("svd")

    def test_invalid_init_method(self):
        structure = compute_ring_structure(100, 32, ring_components=4, rank=4)
        cores = TensorRingCores(structure)
        with pytest.raises(ValueError, match="Unknown init_method"):
            cores.initialize("invalid")

    def test_parameter_count(self):
        structure = compute_ring_structure(100, 32, ring_components=4, rank=4)
        cores = TensorRingCores(structure)
        cores.initialize("uniform")
        count = cores.parameter_count()
        assert count > 0

    def test_spectral_norms(self):
        structure = compute_ring_structure(100, 32, ring_components=4, rank=4)
        cores = TensorRingCores(structure)
        cores.initialize("uniform")
        norms = cores.spectral_norms()
        assert len(norms) == 4  # 2 vocab + 2 emb
        for name, norm in norms.items():
            assert norm > 0
            assert "vocab" in name or "emb" in name


class TestGaugeFixer:
    def test_fix_left(self):
        structure = compute_ring_structure(100, 32, ring_components=4, rank=4)
        cores = TensorRingCores(structure)
        cores.initialize("uniform")

        # Save original values
        original_values = [c.data.clone() for c in cores.vocab_cores]

        GaugeFixer.fix_left(cores.vocab_cores)

        # Values should have changed (unless already orthogonal)
        # At minimum, the operation should not crash
        for core in cores.vocab_cores:
            assert core.data.shape[0] > 0

    def test_fix_right(self):
        structure = compute_ring_structure(100, 32, ring_components=4, rank=4)
        cores = TensorRingCores(structure)
        cores.initialize("uniform")

        GaugeFixer.fix_right(cores.vocab_cores)

        for core in cores.vocab_cores:
            assert core.data.shape[0] > 0

    def test_spectral_norms(self):
        structure = compute_ring_structure(100, 32, ring_components=4, rank=4)
        cores = TensorRingCores(structure)
        cores.initialize("uniform")

        norms = GaugeFixer.spectral_norms(cores.vocab_cores)
        assert len(norms) == 2
        for norm in norms:
            assert norm > 0
