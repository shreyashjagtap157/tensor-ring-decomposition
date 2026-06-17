"""Tests for multi-format embedding matrix loader."""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest
import torch

from tensor_ring_decomposition.loaders.loaders import (
    guess_format,
    load_from_torch,
    load_from_safetensors,
    load_from_numpy,
    load_from_gguf,
    load_from_transformers,
    _resolve_key,
    load_embedding_matrix,
)


@pytest.fixture
def tmp_dir():
    """Create a temporary directory and clean it up afterwards."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# guess_format
# ---------------------------------------------------------------------------

class TestGuessFormat:
    def test_pt(self):
        assert guess_format("model.pt") == "torch"

    def test_pth(self):
        assert guess_format("model.pth") == "torch"

    def test_bin(self):
        assert guess_format("model.bin") == "torch"

    def test_safetensors(self):
        assert guess_format("model.safetensors") == "safetensors"

    def test_npy(self):
        assert guess_format("model.npy") == "numpy"

    def test_npz(self):
        assert guess_format("model.npz") == "numpy"

    def test_gguf(self):
        assert guess_format("model.gguf") == "gguf"

    def test_unknown(self):
        assert guess_format("model.ckpt") == "unknown"

    def test_no_extension(self):
        assert guess_format("model") == "unknown"

    def test_case_insensitive(self):
        assert guess_format("MODEL.PT") == "torch"
        assert guess_format("model.SAFETENSORS") == "safetensors"
        assert guess_format("model.NPY") == "numpy"
        assert guess_format("model.GGUF") == "gguf"

    def test_uppercase_unknown(self):
        assert guess_format("model.CKPT") == "unknown"


# ---------------------------------------------------------------------------
# _resolve_key
# ---------------------------------------------------------------------------

class TestResolveKey:
    def test_exact_match(self):
        d = {"weight": torch.randn(10, 5), "bias": torch.randn(5)}
        result = _resolve_key(d, "weight")
        assert result.shape == (10, 5)

    def test_substring_match(self):
        d = {"encoder.weight": torch.randn(10, 5), "decoder.weight": torch.randn(8, 4)}
        result = _resolve_key(d, "weight")
        assert result.shape == (10, 5)

    def test_substring_match_deeper(self):
        d = {"model.layers.0.attention.weight": torch.randn(16, 8)}
        result = _resolve_key(d, "attention.weight")
        assert result.shape == (16, 8)

    def test_key_not_found_raises(self):
        d = {"some.key": torch.randn(4, 4)}
        with pytest.raises(KeyError, match="not found"):
            _resolve_key(d, "nonexistent")

    def test_empty_dict_raises(self):
        with pytest.raises(KeyError):
            _resolve_key({}, "anything")

    def test_returns_exact_over_substring(self):
        d = {"weight": torch.randn(2, 2), "embedding.weight": torch.randn(4, 4)}
        result = _resolve_key(d, "weight")
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# load_from_torch
# ---------------------------------------------------------------------------

class TestLoadFromTorch:
    def test_raw_tensor(self, tmp_dir):
        path = os.path.join(tmp_dir, "raw.pt")
        t = torch.randn(100, 32)
        torch.save(t, path)
        result = load_from_torch(path)
        assert result.shape == (100, 32)
        assert torch.equal(result, t)

    def test_state_dict_with_wte_key(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.pt")
        data = {"wte.weight": torch.randn(100, 32), "lm_head.weight": torch.randn(100, 32)}
        torch.save(data, path)
        result = load_from_torch(path)
        assert result.shape == (100, 32)
        assert torch.equal(result, data["wte.weight"])

    def test_state_dict_with_embed_tokens_key(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.pt")
        data = {"embed_tokens.weight": torch.randn(50, 16)}
        torch.save(data, path)
        result = load_from_torch(path)
        assert result.shape == (50, 16)

    def test_state_dict_with_shared_key(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.pt")
        data = {"shared.weight": torch.randn(60, 24)}
        torch.save(data, path)
        result = load_from_torch(path)
        assert result.shape == (60, 24)

    def test_state_dict_with_embeddings_key(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.pt")
        data = {"embeddings.word_embeddings.weight": torch.randn(70, 28)}
        torch.save(data, path)
        result = load_from_torch(path)
        assert result.shape == (70, 28)

    def test_state_dict_with_explicit_key(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.pt")
        data = {"custom.key": torch.randn(80, 32), "other": torch.randn(10, 5)}
        torch.save(data, path)
        result = load_from_torch(path, key="custom.key")
        assert result.shape == (80, 32)

    def test_state_dict_uses_largest_2d_when_no_standard_key(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.pt")
        data = {
            "a": torch.randn(10, 5),
            "b": torch.randn(200, 64),
            "c": torch.randn(50, 20),
        }
        torch.save(data, path)
        result = load_from_torch(path)
        assert result.shape == (200, 64)

    def test_state_dict_skips_non_2d_tensors(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.pt")
        data = {
            "a": torch.randn(100, 32),
            "b": torch.randn(64),
            "c": torch.randn(4, 4, 4),
        }
        torch.save(data, path)
        result = load_from_torch(path)
        assert result.shape == (100, 32)

    def test_empty_state_dict_no_2d_tensors_raises(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.pt")
        torch.save({"scalar": torch.tensor(1.0)}, path)
        with pytest.raises(ValueError):
            load_from_torch(path)

    def test_empty_dict_raises(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.pt")
        torch.save({}, path)
        with pytest.raises(ValueError):
            load_from_torch(path)

    def test_unsupported_type_raises(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.pt")
        torch.save([1, 2, 3], path)
        with pytest.raises(TypeError, match="Unsupported"):
            load_from_torch(path)

    def test_weights_only_protection(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.pt")
        torch.save(torch.randn(10, 5), path)
        result = load_from_torch(path)
        assert result.shape == (10, 5)

    def test_device_transfer(self, tmp_dir):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        path = os.path.join(tmp_dir, "model.pt")
        t = torch.randn(100, 32)
        torch.save(t, path)
        result = load_from_torch(path, device=torch.device("cuda"))
        assert result.device.type == "cuda"

    def test_key_uses_resolve_key_substring(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.pt")
        data = {"encoder.weight": torch.randn(100, 32)}
        torch.save(data, path)
        result = load_from_torch(path, key="weight")
        assert result.shape == (100, 32)

    def test_bin_extension(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.bin")
        t = torch.randn(100, 32)
        torch.save(t, path)
        result = load_from_torch(path)
        assert result.shape == (100, 32)

    def test_pth_extension(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.pth")
        t = torch.randn(100, 32)
        torch.save(t, path)
        result = load_from_torch(path)
        assert result.shape == (100, 32)


# ---------------------------------------------------------------------------
# load_from_safetensors
# ---------------------------------------------------------------------------

class TestLoadFromSafetensors:
    def _save_sf(self, tensors, path):
        sf = pytest.importorskip("safetensors.torch")
        sf.save_file(tensors, path)

    def test_load_with_key(self, tmp_dir):
        pytest.importorskip("safetensors.torch")
        path = os.path.join(tmp_dir, "model.safetensors")
        tensors = {"weight": torch.randn(100, 32), "bias": torch.randn(32)}
        self._save_sf(tensors, path)
        result = load_from_safetensors(path, key="weight")
        assert result.shape == (100, 32)

    def test_auto_key_wte(self, tmp_dir):
        pytest.importorskip("safetensors.torch")
        path = os.path.join(tmp_dir, "model.safetensors")
        tensors = {"wte.weight": torch.randn(50, 16)}
        self._save_sf(tensors, path)
        result = load_from_safetensors(path)
        assert result.shape == (50, 16)

    def test_auto_key_embed_tokens(self, tmp_dir):
        pytest.importorskip("safetensors.torch")
        path = os.path.join(tmp_dir, "model.safetensors")
        tensors = {"model.embed_tokens.weight": torch.randn(60, 24)}
        self._save_sf(tensors, path)
        result = load_from_safetensors(path)
        assert result.shape == (60, 24)

    def test_auto_key_lm_head(self, tmp_dir):
        pytest.importorskip("safetensors.torch")
        path = os.path.join(tmp_dir, "model.safetensors")
        tensors = {"lm_head.weight": torch.randn(70, 28)}
        self._save_sf(tensors, path)
        result = load_from_safetensors(path)
        assert result.shape == (70, 28)

    def test_auto_key_embeddings(self, tmp_dir):
        pytest.importorskip("safetensors.torch")
        path = os.path.join(tmp_dir, "model.safetensors")
        tensors = {"embeddings.word_embeddings.weight": torch.randn(80, 32)}
        self._save_sf(tensors, path)
        result = load_from_safetensors(path)
        assert result.shape == (80, 32)

    def test_auto_key_embed_tokens_alt(self, tmp_dir):
        pytest.importorskip("safetensors.torch")
        path = os.path.join(tmp_dir, "model.safetensors")
        tensors = {"embeddings.token_embedding.weight": torch.randn(90, 36)}
        self._save_sf(tensors, path)
        result = load_from_safetensors(path)
        assert result.shape == (90, 36)

    def test_largest_2d_fallback(self, tmp_dir):
        pytest.importorskip("safetensors.torch")
        path = os.path.join(tmp_dir, "model.safetensors")
        tensors = {
            "a": torch.randn(10, 5),
            "b": torch.randn(200, 64),
            "c": torch.randn(50, 20),
        }
        self._save_sf(tensors, path)
        result = load_from_safetensors(path)
        assert result.shape == (200, 64)

    def test_no_2d_tensor_raises(self, tmp_dir):
        pytest.importorskip("safetensors.torch")
        path = os.path.join(tmp_dir, "model.safetensors")
        tensors = {"a": torch.randn(64), "b": torch.randn(4, 4, 4)}
        self._save_sf(tensors, path)
        with pytest.raises(ValueError, match="No 2D tensor"):
            load_from_safetensors(path)

    def test_missing_key_raises(self, tmp_dir):
        pytest.importorskip("safetensors.torch")
        path = os.path.join(tmp_dir, "model.safetensors")
        tensors = {"weight": torch.randn(100, 32)}
        self._save_sf(tensors, path)
        with pytest.raises(KeyError):
            load_from_safetensors(path, key="nonexistent")


# ---------------------------------------------------------------------------
# load_from_numpy
# ---------------------------------------------------------------------------

class TestLoadFromNumpy:
    def test_npy(self, tmp_dir):
        np = pytest.importorskip("numpy")
        path = os.path.join(tmp_dir, "matrix.npy")
        arr = np.random.randn(100, 32).astype(np.float32)
        np.save(path, arr)
        result = load_from_numpy(path)
        assert result.shape == (100, 32)
        assert result.dtype == torch.float32

    def test_npz_single_array(self, tmp_dir):
        np = pytest.importorskip("numpy")
        path = os.path.join(tmp_dir, "matrix.npz")
        arr = np.random.randn(50, 16).astype(np.float32)
        np.savez(path, arr)
        result = load_from_numpy(path)
        assert result.shape == (50, 16)

    def test_npz_multiple_arrays_uses_largest(self, tmp_dir):
        np = pytest.importorskip("numpy")
        path = os.path.join(tmp_dir, "matrix.npz")
        np.savez(path, small=np.random.randn(10, 5), large=np.random.randn(200, 64))
        result = load_from_numpy(path)
        assert result.shape == (200, 64)

    def test_npy_file_not_found_raises(self, tmp_dir):
        path = os.path.join(tmp_dir, "nonexistent.npy")
        with pytest.raises(FileNotFoundError):
            load_from_numpy(path)

    def test_npy_device_transfer_cpu(self, tmp_dir):
        np = pytest.importorskip("numpy")
        path = os.path.join(tmp_dir, "matrix.npy")
        arr = np.random.randn(100, 32).astype(np.float32)
        np.save(path, arr)
        result = load_from_numpy(path, device=torch.device("cpu"))
        assert result.device.type == "cpu"


# ---------------------------------------------------------------------------
# load_from_gguf
# ---------------------------------------------------------------------------

class TestLoadFromGguf:
    def test_missing_gguf_package_raises(self):
        with patch.dict("sys.modules", {"gguf": None}):
            with pytest.raises(ImportError, match="gguf"):
                load_from_gguf("dummy.gguf")

    def test_import_error_when_not_installed(self):
        with pytest.raises(ImportError, match="gguf"):
            load_from_gguf("dummy.gguf")

    def test_successful_load_with_mock(self, tmp_dir):
        np = pytest.importorskip("numpy")
        gguf = pytest.importorskip("gguf")
        path = os.path.join(tmp_dir, "model.gguf")
        with open(path, "wb") as f:
            f.write(b"dummy")
        mock_tensor = MagicMock()
        mock_tensor.data = np.random.randn(100, 32).astype(np.float32)
        mock_reader = MagicMock()
        mock_reader.get_tensor.return_value = mock_tensor
        mock_reader.tensors = [mock_tensor]

        with patch("gguf.GGUFReader", return_value=mock_reader):
            result = load_from_gguf(path)
            assert result.shape == (100, 32)

    def test_no_embedding_tensor_raises(self, tmp_dir):
        pytest.importorskip("gguf")
        path = os.path.join(tmp_dir, "model.gguf")
        with open(path, "wb") as f:
            f.write(b"dummy")
        mock_reader = MagicMock()
        mock_reader.get_tensor.side_effect = ValueError("not found")
        mock_tensor = MagicMock()
        mock_tensor.name = "some_other.weight"
        mock_reader.tensors = [mock_tensor]

        with patch("gguf.GGUFReader", return_value=mock_reader):
            with pytest.raises(KeyError, match="No embedding tensor"):
                load_from_gguf(path)


# ---------------------------------------------------------------------------
# load_from_transformers
# ---------------------------------------------------------------------------

class TestLoadFromTransformers:
    def test_missing_transformers_package_raises(self):
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                load_from_transformers("bert-base-uncased")

    def test_import_error_when_not_installed(self):
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                load_from_transformers("bert-base-uncased")

    def test_successful_load_with_mock(self, tmp_dir):
        weight_tensor = torch.randn(100, 32)
        mock_model = MagicMock()
        mock_emb = MagicMock()
        type(mock_emb).weight = PropertyMock(return_value=weight_tensor)
        mock_model.get_input_embeddings.return_value = mock_emb
        mock_auto_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        fake_transformers = MagicMock(spec=[])
        fake_transformers.AutoModel = mock_auto_model
        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            result = load_from_transformers("test-model", cache_dir=tmp_dir)
            assert result.shape == (100, 32)
            mock_auto_model.from_pretrained.assert_called_once_with("test-model", cache_dir=tmp_dir, low_cpu_mem_usage=True)


# ---------------------------------------------------------------------------
# load_embedding_matrix  (fallback chain / format dispatch)
# ---------------------------------------------------------------------------

class TestLoadEmbeddingMatrix:
    def test_dispatch_torch(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.pt")
        t = torch.randn(100, 32)
        torch.save(t, path)
        result = load_embedding_matrix(path)
        assert result.shape == (100, 32)

    def _save_sf(self, tensors, path):
        sf = pytest.importorskip("safetensors.torch")
        sf.save_file(tensors, path)

    def test_dispatch_safetensors(self, tmp_dir):
        pytest.importorskip("safetensors.torch")
        path = os.path.join(tmp_dir, "model.safetensors")
        tensors = {"weight": torch.randn(100, 32)}
        self._save_sf(tensors, path)
        result = load_embedding_matrix(path, format="safetensors", key="weight")
        assert result.shape == (100, 32)

    def test_dispatch_numpy(self, tmp_dir):
        np = pytest.importorskip("numpy")
        path = os.path.join(tmp_dir, "matrix.npy")
        arr = np.random.randn(100, 32).astype(np.float32)
        np.save(path, arr)
        result = load_embedding_matrix(path, format="numpy")
        assert result.shape == (100, 32)

    def test_dispatch_torch_with_explicit_format(self, tmp_dir):
        path = os.path.join(tmp_dir, "myfile.custom")
        t = torch.randn(100, 32)
        torch.save(t, path)
        result = load_embedding_matrix(path, format="torch")
        assert result.shape == (100, 32)

    def test_file_not_found_raises(self, tmp_dir):
        path = os.path.join(tmp_dir, "nonexistent.pt")
        with pytest.raises(FileNotFoundError, match="not found"):
            load_embedding_matrix(path)

    def test_auto_format_from_bin_extension(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.bin")
        t = torch.randn(100, 32)
        torch.save(t, path)
        result = load_embedding_matrix(path)
        assert result.shape == (100, 32)

    def test_unknown_extension_fallback_chain_torch(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.unknown_ext")
        t = torch.randn(100, 32)
        torch.save(t, path)
        result = load_embedding_matrix(path)
        assert result.shape == (100, 32)

    def test_unknown_extension_fallback_chain_numpy(self, tmp_dir):
        np = pytest.importorskip("numpy")
        npy_path = os.path.join(tmp_dir, "model.npy")
        path = os.path.join(tmp_dir, "model.unknown_ext")
        arr = np.random.randn(100, 32).astype(np.float32)
        np.save(npy_path, arr)
        os.rename(npy_path, path)
        result = load_embedding_matrix(path)
        assert result.shape == (100, 32)

    def test_unknown_extension_all_fail_then_transformers(self, tmp_dir):
        path = os.path.join(tmp_dir, "nonexistent.unknown_ext")
        with patch(
            "tensor_ring_decomposition.loaders.loaders.load_from_transformers",
            return_value=torch.randn(100, 32),
        ) as mock:
            result = load_embedding_matrix(path)
            assert result.shape == (100, 32)
            mock.assert_called_once()

    def test_transformers_fallback_when_file_not_exists(self, tmp_dir):
        path = os.path.join(tmp_dir, "nonexistent.unknown")
        with patch(
            "tensor_ring_decomposition.loaders.loaders.load_from_transformers",
            return_value=torch.randn(100, 32),
        ) as mock:
            result = load_embedding_matrix(path)
            assert result.shape == (100, 32)
            mock.assert_called_once()

    def test_format_override_ignores_extension(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.npy")
        t = torch.randn(100, 32)
        torch.save(t, path)
        result = load_embedding_matrix(path, format="torch")
        assert result.shape == (100, 32)

    def test_transformers_via_format(self):
        with patch(
            "tensor_ring_decomposition.loaders.loaders.load_from_transformers",
            return_value=torch.randn(100, 32),
        ) as mock:
            result = load_embedding_matrix("bert-base-uncased", format="transformers")
            assert result.shape == (100, 32)
            mock.assert_called_once_with("bert-base-uncased", None, None)

    def test_key_passed_to_torch_loader(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.pt")
        data = {"custom.embedding.weight": torch.randn(100, 32)}
        torch.save(data, path)
        result = load_embedding_matrix(path, key="custom.embedding.weight")
        assert result.shape == (100, 32)

    def test_key_passthrough_to_safetensors(self, tmp_dir):
        pytest.importorskip("safetensors.torch")
        path = os.path.join(tmp_dir, "model.safetensors")
        tensors = {"my.custom.weight": torch.randn(100, 32)}
        self._save_sf(tensors, path)
        result = load_embedding_matrix(path, format="safetensors", key="my.custom.weight")
        assert result.shape == (100, 32)

    def test_gguf_dispatch(self, tmp_dir):
        np = pytest.importorskip("numpy")
        pytest.importorskip("gguf")
        path = os.path.join(tmp_dir, "model.gguf")
        with open(path, "wb") as f:
            f.write(b"dummy")
        mock_tensor = MagicMock()
        mock_tensor.data = np.random.randn(100, 32).astype(np.float32)
        mock_reader = MagicMock()
        mock_reader.get_tensor.return_value = mock_tensor
        mock_reader.tensors = [mock_tensor]
        with patch("gguf.GGUFReader", return_value=mock_reader):
            result = load_embedding_matrix(path, format="gguf")
            assert result.shape == (100, 32)

    def test_dispatch_safetensors_auto_detect(self, tmp_dir):
        pytest.importorskip("safetensors.torch")
        path = os.path.join(tmp_dir, "model.safetensors")
        tensors = {"wte.weight": torch.randn(100, 32)}
        self._save_sf(tensors, path)
        result = load_embedding_matrix(path)
        assert result.shape == (100, 32)
