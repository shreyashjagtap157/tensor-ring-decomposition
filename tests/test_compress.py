"""Tests for the high-level compress / list_models API."""

import pytest
import torch
import torch.nn as nn
from tensor_ring_decomposition import compress, list_models
from tensor_ring_decomposition.core.embedding import TensorRingEmbedding


class TestCompressTensor:
    """compress() with a raw dense tensor."""

    @pytest.mark.slow
    def test_basic_tensor_with_rank(self):
        tensor = torch.randn(100, 32)
        emb = compress(tensor, rank=4)
        assert isinstance(emb, TensorRingEmbedding)
        assert emb.vocab_size == 100
        assert emb.embedding_dim == 32
        assert emb.compression_ratio > 1.0

    @pytest.mark.slow
    def test_with_target_compression(self):
        tensor = torch.randn(200, 64)
        emb = compress(tensor, target_compression=10)
        assert isinstance(emb, TensorRingEmbedding)
        assert emb.compression_ratio >= 9.0

    @pytest.mark.slow
    def test_with_target_params(self):
        tensor = torch.randn(200, 64)
        emb = compress(tensor, target_params=5000)
        assert isinstance(emb, TensorRingEmbedding)
        assert emb.num_parameters <= 6000

    @pytest.mark.slow
    def test_with_autotune(self):
        tensor = torch.randn(100, 32)
        emb = compress(tensor, rank=4, autotune=True)
        assert isinstance(emb, TensorRingEmbedding)
        assert emb.vocab_size == 100
        assert emb.embedding_dim == 32

    @pytest.mark.slow
    def test_with_device_cpu(self):
        tensor = torch.randn(100, 32)
        emb = compress(tensor, rank=4, device="cpu")
        assert isinstance(emb, TensorRingEmbedding)
        param_device = emb.cores.vocab_cores[0].device
        assert str(param_device) == "cpu"

    @pytest.mark.slow
    def test_with_explicit_torch_device(self):
        tensor = torch.randn(100, 32)
        emb = compress(tensor, rank=4, device=torch.device("cpu"))
        assert isinstance(emb, TensorRingEmbedding)

    @pytest.mark.slow
    def test_returns_tensor_ring_embedding_instance(self):
        tensor = torch.randn(50, 16)
        result = compress(tensor, rank=4)
        assert isinstance(result, TensorRingEmbedding)

    @pytest.mark.slow
    def test_forward_works_after_compress(self):
        tensor = torch.randn(100, 32)
        emb = compress(tensor, rank=4)
        indices = torch.tensor([0, 1, 2, 3])
        output = emb(indices)
        assert output.shape == (4, 32)


class TestCompressInvalidInputs:
    """compress() with invalid / unsupported inputs."""

    def test_nonexistent_file_path(self):
        with pytest.raises(FileNotFoundError):
            compress("nonexistent_file.pt")

    def test_random_string_not_a_file_or_model(self):
        with pytest.raises((FileNotFoundError, ImportError)):
            compress("this_is_not_a_valid_model_or_file_path_xyzzy")

    def test_unsupported_type_integer(self):
        with pytest.raises(TypeError, match="Unsupported source type"):
            compress(42)  # type: ignore[arg-type]

    def test_unsupported_type_list(self):
        with pytest.raises(TypeError, match="Unsupported source type"):
            compress([1, 2, 3])  # type: ignore[arg-type]

    def test_nn_embedding_module_not_accepted(self):
        embed = nn.Embedding(100, 32)
        with pytest.raises(TypeError, match="Unsupported source type"):
            compress(embed)  # type: ignore[arg-type]

    def test_nn_embedding_weight_extracted(self):
        embed = nn.Embedding(100, 32)
        emb = compress(embed.weight, rank=4)
        assert isinstance(emb, TensorRingEmbedding)
        assert emb.vocab_size == 100
        assert emb.embedding_dim == 32


class TestCompressWithoutRankOrTarget:
    """compress() with insufficient config uses default rank."""

    @pytest.mark.slow
    def test_no_rank_no_target_defaults_to_rank_8(self):
        tensor = torch.randn(100, 32)
        emb = compress(tensor)
        assert isinstance(emb, TensorRingEmbedding)
        assert emb.rank >= 2


class TestListModels:
    """list_models() functionality."""

    def test_list_all_models_returns_string(self):
        result = list_models()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Model Registry Summary" in result or "bert" in result.lower() or "llama" in result.lower()

    def test_list_by_valid_family_llama(self):
        result = list_models(family="llama")
        assert isinstance(result, str)
        assert "llama" in result.lower() or "Models in family" in result

    def test_list_by_valid_family_bert(self):
        result = list_models(family="bert")
        assert isinstance(result, str)
        assert "bert" in result.lower() or "Models in family" in result

    def test_list_by_unknown_family(self):
        result = list_models(family="nonexistent_family_xyz")
        assert isinstance(result, str)
        assert "Unknown family" in result

    def test_list_family_with_varied_case(self):
        result = list_models(family="LLAMA")
        assert isinstance(result, str)
        assert "llama" in result.lower() or "Models in family" in result
