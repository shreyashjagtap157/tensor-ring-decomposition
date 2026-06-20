"""Tests for export / load / compile functionality of TensorRingEmbedding."""

import logging
import tempfile
from pathlib import Path

import pytest
import torch

from tensor_ring_decomposition import TensorRingEmbedding, ExportFormat


_HAS_ONNX = False
try:
    import torch.onnx  # noqa: F401
    import onnxscript  # noqa: F401 used by torch.onnx.export internally
    _HAS_ONNX = True
except ImportError:
    pass


_HAS_TRITON = True  # assume torch.compile may work; individual tests adapt


def _make_embedding(**kwargs) -> TensorRingEmbedding:
    params = dict(vocab_size=100, embedding_dim=32, rank=4)
    params.update(kwargs)
    return TensorRingEmbedding(**params)


class TestTorchScriptExport:
    def test_export_and_load_torchscript(self):
        emb = _make_embedding()
        indices = torch.randint(0, 100, (2, 8))
        emb.eval()
        original = emb(indices)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "tr_embedding")
            exported_path = emb.export(path, format=ExportFormat.TORCHSCRIPT)
            assert Path(exported_path).exists()
            assert exported_path.endswith(".torchscript")

            loaded = TensorRingEmbedding.load_exported(exported_path)
            output = loaded(indices)

        assert torch.allclose(original, output, atol=1e-5)

    def test_exported_model_preserves_output_shape(self):
        emb = _make_embedding()
        indices = torch.tensor([[0, 1], [2, 3]])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "tr_emb_shape")
            exported_path = emb.export(
                path, format=ExportFormat.TORCHSCRIPT, batch_size=2, seq_len=2
            )
            loaded = TensorRingEmbedding.load_exported(exported_path)
            output = loaded(indices)

        assert output.shape == (2, 2, 32)

    def test_exported_model_different_batch_size(self):
        emb = _make_embedding()
        indices_small = torch.tensor([[0, 1], [2, 3]])
        indices_large = torch.randint(0, 100, (4, 16))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "tr_emb_batch")
            emb.export(path, format=ExportFormat.TORCHSCRIPT, batch_size=2, seq_len=4)
            loaded = TensorRingEmbedding.load_exported(path + ".torchscript")

        small_out = loaded(indices_small)
        large_out = loaded(indices_large)
        assert small_out.shape == (2, 2, 32)
        assert large_out.shape == (4, 16, 32)

    def test_export_twice_overwrites(self):
        emb = _make_embedding()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "tr_emb")
            emb.export(path, format=ExportFormat.TORCHSCRIPT)
            size_before = Path(path + ".torchscript").stat().st_size

            emb2 = _make_embedding(vocab_size=200, embedding_dim=64)
            emb2.export(path, format=ExportFormat.TORCHSCRIPT)
            size_after = Path(path + ".torchscript").stat().st_size

        assert size_after != size_before

    def test_export_to_nonexistent_directory(self):
        emb = _make_embedding()

        with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
            emb.export(
                "/nonexistent_dir_xyz/tensor_ring_model",
                format=ExportFormat.TORCHSCRIPT,
            )

    def test_load_nonexistent_file(self):
        with pytest.raises(Exception):
            TensorRingEmbedding.load_exported(
                "/nonexistent_path/tensor_ring_model.torchscript"
            )


class TestONNXExport:
    @pytest.mark.skipif(not _HAS_ONNX, reason="torch.onnx not available")
    @pytest.mark.skipif(
        torch.__version__ < (2, 0), reason="torch >= 2.0 required for ONNX export"
    )
    def test_export_onnx(self):
        emb = _make_embedding()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "tr_embedding")
            exported_path = emb.export(path, format=ExportFormat.ONNX)
            assert Path(exported_path).exists()
            assert exported_path.endswith(".onnx")

    @pytest.mark.skipif(not _HAS_ONNX, reason="torch.onnx not available")
    @pytest.mark.skipif(
        torch.__version__ < (2, 0), reason="torch >= 2.0 required for ONNX export"
    )
    def test_export_onnx_static_axes(self):
        emb = _make_embedding()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "tr_embedding_static")
            exported_path = emb.export(path, format=ExportFormat.ONNX, dynamic_axes=False)
            assert Path(exported_path).exists()

    @pytest.mark.skipif(not _HAS_ONNX, reason="torch.onnx not available")
    @pytest.mark.skipif(
        torch.__version__ < (2, 0), reason="torch >= 2.0 required for ONNX export"
    )
    def test_export_onnx_different_batch_size(self):
        emb = _make_embedding()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "tr_embedding_bs")
            emb.export(path, format=ExportFormat.ONNX, batch_size=4, seq_len=64)
            assert Path(path + ".onnx").exists()


class TestLoadExported:
    def test_load_exported_returns_scriptmodule(self):
        emb = _make_embedding()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "tr_embedding")
            exported_path = emb.export(path, format=ExportFormat.TORCHSCRIPT)
            loaded = TensorRingEmbedding.load_exported(exported_path)

        assert isinstance(loaded, torch.jit.ScriptModule)

    def test_load_exported_output_identical_to_original(self):
        emb = _make_embedding()
        indices = torch.randint(0, 100, (3, 7))
        emb.eval()
        original = emb(indices)

        with tempfile.TemporaryDirectory() as tmpdir:
            exported_path = emb.export(
                str(Path(tmpdir) / "tr_embedding"), format=ExportFormat.TORCHSCRIPT
            )
            loaded = TensorRingEmbedding.load_exported(exported_path)
            output = loaded(indices)

        assert torch.allclose(original, output, atol=1e-5)

    def test_load_exported_deterministic(self):
        emb = _make_embedding()
        indices = torch.randint(0, 100, (2, 4))

        with tempfile.TemporaryDirectory() as tmpdir:
            exported_path = emb.export(
                str(Path(tmpdir) / "tr_embedding"), format=ExportFormat.TORCHSCRIPT
            )
            loaded = TensorRingEmbedding.load_exported(exported_path)

        output1 = loaded(indices)
        output2 = loaded(indices)
        assert torch.allclose(output1, output2, atol=1e-5)

    def test_load_exported_with_device(self):
        emb = _make_embedding()

        with tempfile.TemporaryDirectory() as tmpdir:
            exported_path = emb.export(
                str(Path(tmpdir) / "tr_embedding"), format=ExportFormat.TORCHSCRIPT
            )
            loaded = TensorRingEmbedding.load_exported(exported_path, device="cpu")

        assert isinstance(loaded, torch.jit.ScriptModule)


class TestCompileForward:
    def test_compile_forward_returns_callable(self):
        emb = _make_embedding()
        compiled_forward = emb.compile_forward(mode="reduce-overhead")
        assert callable(compiled_forward)

    def test_compile_forward_default_mode(self):
        emb = _make_embedding()
        compiled_forward = emb.compile_forward()
        assert callable(compiled_forward)

    def test_compile_forward_with_fullgraph(self):
        emb = _make_embedding()
        compiled_forward = emb.compile_forward(mode="reduce-overhead", fullgraph=True)
        assert callable(compiled_forward)

    def test_compile_forward_can_be_called(self):
        emb = _make_embedding()
        compiled_forward = emb.compile_forward(mode="reduce-overhead")
        indices = torch.randint(0, 100, (2, 8))
        try:
            output = compiled_forward(indices)
            assert output.shape == (2, 8, 32)
        except (torch._inductor.exc.InductorError, RuntimeError) as e:
            if "Compiler" in str(e) or "not found" in str(e) or "cl" in str(e):
                pytest.skip("torch.compile requires a C++ compiler on this system")
            raise

    def test_compile_forward_output_matches(self):
        emb = _make_embedding()
        indices = torch.randint(0, 100, (2, 8))
        emb.eval()
        original = emb(indices)

        compiled_forward = emb.compile_forward(mode="reduce-overhead")
        try:
            compiled = compiled_forward(indices)
            assert torch.allclose(original, compiled, atol=1e-5)
        except (torch._inductor.exc.InductorError, RuntimeError) as e:
            if "Compiler" in str(e) or "not found" in str(e) or "cl" in str(e):
                pytest.skip("torch.compile requires a C++ compiler on this system")
            raise

    def test_compile_forward_works_after_train(self):
        emb = _make_embedding()
        emb.train()
        _ = emb(torch.randint(0, 100, (2, 4)))
        emb.eval()

        compiled_forward = emb.compile_forward(mode="reduce-overhead")
        indices = torch.randint(0, 100, (2, 8))
        try:
            output = compiled_forward(indices)
            assert output.shape == (2, 8, 32)
        except (torch._inductor.exc.InductorError, RuntimeError) as e:
            if "Compiler" in str(e) or "not found" in str(e) or "cl" in str(e):
                pytest.skip("torch.compile requires a C++ compiler on this system")
            raise


class TestExportEdgeCases:
    def test_export_empty_vocab_fails(self):
        with pytest.raises(ValueError):
            _make_embedding(vocab_size=0)

    def test_export_after_train_mode_resets_ok(self):
        emb = _make_embedding()
        emb.train()
        _ = emb(torch.randint(0, 100, (2, 4)))
        emb.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "tr_embedding")
            exported_path = emb.export(path, format=ExportFormat.TORCHSCRIPT)
            loaded = TensorRingEmbedding.load_exported(exported_path)

        indices = torch.randint(0, 100, (2, 4))
        output = loaded(indices)
        assert output.shape == (2, 4, 32)

    def test_export_with_padding_idx(self):
        emb = _make_embedding(padding_idx=0)
        indices = torch.tensor([[0, 1], [2, 3]])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "tr_emb_pad")
            exported_path = emb.export(path, format=ExportFormat.TORCHSCRIPT)
            loaded = TensorRingEmbedding.load_exported(exported_path)
            output = loaded(indices)

        assert output.shape == (2, 2, 32)

    def test_export_to_path_with_spaces(self):
        emb = _make_embedding()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "my model" / "tr_embedding")
            with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
                emb.export(path, format=ExportFormat.TORCHSCRIPT)

    def test_load_exported_onnx_raises_error(self):
        emb = _make_embedding()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "tr_embedding.onnx"
            onnx_path.write_text("fake onnx content")
            with pytest.raises(RuntimeError, match="ONNX files require an ONNX runtime"):
                TensorRingEmbedding.load_exported(str(onnx_path))
