"""Tests for QuantizedTensorRingEmbedding quantization module."""

import pytest
import torch
import torch.nn as nn

from tensor_ring_decomposition import TensorRingEmbedding, QuantizedTensorRingEmbedding


class TestQuantizedTensorRingEmbedding:
    """Tests covering PTQ, QAT, LSQ, per-channel/per-tensor, eval caching,
    non-negative constraint, and optimizer compatibility."""

    # ── 1. Basic creation ───────────────────────────────────────────

    def test_create_per_channel(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, per_channel=True)
        assert qemb._quantized
        assert qemb._per_channel is True
        assert len(qemb._q_vocab_cores) > 0
        assert len(qemb._q_emb_cores) > 0
        for core in qemb._q_vocab_cores:
            assert core.dtype == torch.int8
        for core in qemb._q_emb_cores:
            assert core.dtype == torch.int8
        for s in qemb._vocab_scales:
            assert isinstance(s, torch.Tensor)
            assert s.ndim > 0
        for s in qemb._emb_scales:
            assert isinstance(s, torch.Tensor)
            assert s.ndim > 0

    # ── 2. Per-tensor quantization ──────────────────────────────────

    def test_create_per_tensor(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, per_channel=False)
        assert qemb._quantized
        assert qemb._per_channel is False
        for s in qemb._vocab_scales:
            assert isinstance(s, float)
        for s in qemb._emb_scales:
            assert isinstance(s, float)

    # ── 3. Forward pass shape ───────────────────────────────────────

    def test_forward_shape_single(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        indices = torch.tensor([0, 1, 2, 3])
        output = qemb(indices)
        assert output.shape == (4, 32)
        assert output.dtype == torch.float32

    def test_forward_shape_batched(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        indices = torch.tensor([[0, 1], [2, 3]])
        output = qemb(indices)
        assert output.shape == (2, 2, 32)

    def test_forward_shape_3d(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        indices = torch.tensor([[[0], [1]], [[2], [3]]])
        output = qemb(indices)
        assert output.shape == (2, 2, 1, 32)

    def test_forward_per_tensor_shape(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, per_channel=False)
        indices = torch.tensor([0, 1, 2, 3])
        output = qemb(indices)
        assert output.shape == (4, 32)

    def test_forward_qat_shape(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True)
        indices = torch.tensor([0, 1, 2, 3])
        output = qemb(indices)
        assert output.shape == (4, 32)

    def test_forward_qat_lsq_shape(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True, lsq=True)
        indices = torch.tensor([0, 1, 2, 3])
        output = qemb(indices)
        assert output.shape == (4, 32)

    def test_forward_large_vocab(self):
        emb = TensorRingEmbedding(1000, 64, rank=8)
        qemb = QuantizedTensorRingEmbedding(emb)
        indices = torch.randint(0, 1000, (16,))
        output = qemb(indices)
        assert output.shape == (16, 64)

    def test_forward_padding_idx(self):
        emb = TensorRingEmbedding(100, 32, rank=4, padding_idx=0)
        qemb = QuantizedTensorRingEmbedding(emb)
        indices = torch.tensor([0, 1, 2])
        output = qemb(indices)
        assert output.shape == (3, 32)

    # ── 4. bits_per_parameter property ─────────────────────────────

    def test_bits_per_parameter_qat(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True, per_channel=False)
        bpp = qemb.bits_per_parameter
        assert bpp == 8.0

    def test_bits_per_parameter_qat_lsq(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True, lsq=True)
        bpp = qemb.bits_per_parameter
        assert 8.0 < bpp < 32.0

    def test_bits_per_parameter_qat_per_channel(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True, per_channel=True)
        bpp = qemb.bits_per_parameter
        assert bpp == 8.0

    # ── 5. compression_ratio property ──────────────────────────────

    def test_compression_ratio_positive(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        ratio = qemb.compression_ratio
        assert ratio > 1.0

    def test_compression_ratio_ptq(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        base_ratio = emb.compression_ratio
        qratio = qemb.compression_ratio
        assert qratio > 1.0
        assert qratio >= base_ratio

    def test_compression_ratio_qat(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True)
        qratio = qemb.compression_ratio
        assert qratio > 1.0

    def test_compression_ratio_qat_lsq(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True, lsq=True)
        qratio = qemb.compression_ratio
        assert qratio > 1.0

    # ── 6. to_eval_mode() ──────────────────────────────────────────

    def test_to_eval_mode_caches(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        qemb.to_eval_mode()
        assert not qemb.training
        assert qemb._cache_valid
        assert qemb._emb_cache is not None
        assert isinstance(qemb._emb_cache, torch.Tensor)

    def test_to_eval_mode_no_gradients(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        qemb.to_eval_mode()
        assert not qemb._emb_cache.requires_grad

    def test_to_eval_mode_forward_uses_cache(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        qemb.to_eval_mode()
        indices = torch.tensor([0, 1, 2, 3])
        output = qemb(indices)
        assert output.shape == (4, 32)
        assert not output.requires_grad

    def test_to_eval_mode_then_train_recomputes(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        qemb.to_eval_mode()
        assert qemb._cache_valid
        qemb.train()
        assert qemb.training
        indices = torch.tensor([0, 1, 2, 3])
        output = qemb(indices)
        assert output.shape == (4, 32)

    def test_to_eval_mode_forward_no_grad_after_eval(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        qemb.to_eval_mode()
        with torch.no_grad():
            indices = torch.tensor([0, 1, 2, 3])
            output = qemb(indices)
        assert not output.requires_grad

    def test_to_eval_mode_qat_eval_cache_not_set(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True)
        qemb.to_eval_mode()
        assert not qemb.training
        assert not qemb._cache_valid
        assert qemb._emb_cache is None

    def test_to_eval_mode_returns_self(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        returned = qemb.to_eval_mode()
        assert returned is qemb

    # ── 7. train() mode enables gradient flow ──────────────────────

    def test_train_mode_ptq_output_no_grad(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        qemb.train()
        assert qemb.training
        indices = torch.tensor([0, 1, 2])
        output = qemb(indices)
        assert not output.requires_grad

    def test_train_mode_gradient_flow_qat(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True, per_channel=False)
        qemb.train()
        indices = torch.tensor([0, 1, 2])
        output = qemb(indices)
        loss = output.sum()
        loss.backward()
        grads = [p.grad for p in emb.parameters() if p.grad is not None]
        assert len(grads) > 0
        for g in grads:
            assert g is not None

    def test_train_mode_gradient_flow_qat_lsq(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True, lsq=True, per_channel=False)
        qemb.train()
        indices = torch.tensor([0, 1, 2])
        output = qemb(indices)
        loss = output.sum()
        loss.backward()
        lsq_grads = [p.grad for i in range(len(qemb._vocab_lsq_scales))
                     if (p := qemb._vocab_lsq_scales[i]).grad is not None]
        assert len(lsq_grads) > 0
        for g in lsq_grads:
            assert g is not None

    def test_train_mode_gradient_flow_qat_lsq_per_channel(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True, lsq=True, per_channel=True)
        qemb.train()
        indices = torch.tensor([0, 1, 2])
        output = qemb(indices)
        loss = output.sum()
        loss.backward()
        lsq_grads_vocab = [p.grad for i in range(len(qemb._vocab_lsq_scales))
                           if (p := qemb._vocab_lsq_scales[i]).grad is not None]
        lsq_grads_emb = [p.grad for i in range(len(qemb._emb_lsq_scales))
                         if (p := qemb._emb_lsq_scales[i]).grad is not None]
        assert len(lsq_grads_vocab) > 0
        assert len(lsq_grads_emb) > 0

    def test_train_mode_ptq_no_grad_after_eval(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        qemb.to_eval_mode()
        indices = torch.tensor([0, 1, 2])
        output = qemb(indices)
        assert not output.requires_grad

    # ── 8. QAT: forward + backward in train mode ───────────────────

    def test_qat_forward_backward_loss_drops(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True, lsq=True, per_channel=False)
        qemb.train()
        optimizer = torch.optim.SGD(qemb.parameters(), lr=1e-2)
        indices = torch.tensor([0, 1, 2, 3, 4, 5])
        target = torch.randn(6, 32)
        loss_before = None
        loss_after = None
        for step in range(5):
            optimizer.zero_grad()
            output = qemb(indices)
            loss = (output - target).pow(2).sum()
            if step == 0:
                loss_before = loss.item()
            loss.backward()
            optimizer.step()
        output = qemb(indices)
        loss_after = (output - target).pow(2).sum().item()
        assert loss_after <= loss_before * 1.05 + 0.1

    def test_qat_lsq_per_channel_forward_backward(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True, lsq=True, per_channel=True)
        qemb.train()
        optimizer = torch.optim.SGD(qemb.parameters(), lr=1e-2)
        indices = torch.tensor([0, 1, 2])
        target = torch.randn(3, 32)
        for _ in range(3):
            optimizer.zero_grad()
            output = qemb(indices)
            loss = (output - target).pow(2).sum()
            loss.backward()
            optimizer.step()

    def test_qat_non_lsq_fake_quantize(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True, lsq=False, per_channel=False)
        qemb.train()
        indices = torch.tensor([0, 1, 2])
        output = qemb(indices)
        loss = output.sum()
        loss.backward()
        grads = [p.grad for p in emb.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_qat_gradient_chain_unbroken(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True, lsq=True)
        qemb.train()
        indices = torch.tensor([0, 1, 2])
        output = qemb(indices)
        loss = output.mean()
        loss.backward()
        has_any_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in qemb.parameters()
        )
        assert has_any_grad

    # ── 9. Multiple quantize() calls ───────────────────────────────

    def test_multiple_quantize_calls_idempotent(self):
        emb = TensorRingEmbedding(1000, 64, rank=8)
        qemb = QuantizedTensorRingEmbedding(emb)
        first_vocab = [c.clone() for c in qemb._q_vocab_cores]
        qemb.quantize(emb)
        for c1, c2 in zip(first_vocab, qemb._q_vocab_cores):
            assert torch.equal(c1, c2)

    def test_quantize_after_forward(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        indices = torch.tensor([0, 1, 2, 3])
        _ = qemb(indices)
        qemb.quantize(emb)
        assert qemb._quantized
        output = qemb(indices)
        assert output.shape == (4, 32)

    def test_quantize_with_different_embedding(self):
        emb1 = TensorRingEmbedding(100, 32, rank=4)
        emb2 = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb1)
        qemb.quantize(emb2)
        indices = torch.tensor([0, 1, 2, 3])
        output = qemb(indices)
        assert output.shape == (4, 32)

    def test_quantize_after_eval_cache(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        qemb.to_eval_mode()
        qemb.quantize(emb)
        output = qemb(torch.tensor([0, 1]))
        assert output.shape == (2, 32)

    # ── 10. non_negative=True ──────────────────────────────────────

    def test_non_negative_output(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, non_negative=True)
        indices = torch.tensor([0, 1, 2, 3])
        output = qemb(indices)
        assert (output >= 0).all()

    def test_non_negative_batched(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, non_negative=True)
        indices = torch.tensor([[0, 1], [2, 3]])
        output = qemb(indices)
        assert output.shape == (2, 2, 32)
        assert (output >= 0).all()

    def test_non_negative_qat(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True, non_negative=True)
        indices = torch.tensor([0, 1, 2, 3])
        output = qemb(indices)
        assert (output >= 0).all()

    def test_non_negative_no_grad_eval(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, non_negative=True)
        qemb.to_eval_mode()
        indices = torch.tensor([0, 1, 2, 3])
        output = qemb(indices)
        assert (output >= 0).all()
        assert not output.requires_grad

    def test_non_negative_comparison(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb_pos = QuantizedTensorRingEmbedding(emb, non_negative=True)
        indices = torch.tensor([0, 1, 2, 3])
        output_pos = qemb_pos(indices)
        assert (output_pos >= 0).all()

    # ── 11. SGD / Adam compatibility ───────────────────────────────

    def test_sgd_compatibility_qat(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True)
        optimizer = torch.optim.SGD(qemb.parameters(), lr=0.01)
        indices = torch.tensor([0, 1, 2])
        output = qemb(indices)
        loss = output.sum()
        loss.backward()
        optimizer.step()

    def test_adam_compatibility_qat(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True)
        optimizer = torch.optim.Adam(qemb.parameters(), lr=0.001)
        indices = torch.tensor([0, 1, 2])
        output = qemb(indices)
        loss = output.sum()
        loss.backward()
        optimizer.step()

    def test_sgd_compatibility_qat_lsq(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True, lsq=True)
        optimizer = torch.optim.SGD(qemb.parameters(), lr=0.01)
        indices = torch.tensor([0, 1, 2])
        output = qemb(indices)
        loss = output.sum()
        loss.backward()
        optimizer.step()

    def test_adam_compatibility_qat_lsq(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True, lsq=True)
        optimizer = torch.optim.Adam(qemb.parameters(), lr=0.001)
        indices = torch.tensor([0, 1, 2])
        output = qemb(indices)
        loss = output.sum()
        loss.backward()
        optimizer.step()

    def test_optimizer_parameters_updated(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True, lsq=True)
        optimizer = torch.optim.SGD(qemb.parameters(), lr=1.0)
        params_before = [p.data.clone() for p in qemb.parameters()]
        indices = torch.tensor([0, 1, 2])
        output = qemb(indices)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        params_after = list(qemb.parameters())
        any_changed = any(
            not torch.equal(b, a.data) for b, a in zip(params_before, params_after)
        )
        assert any_changed

    def test_sgd_multi_step_convergence(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb, qat=True, lsq=True)
        optimizer = torch.optim.Adam(qemb.parameters(), lr=1e-2)
        indices = torch.tensor([0, 1, 2, 3, 4, 5])
        target = torch.randn(6, 32)
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            output = qemb(indices)
            loss = (output - target).pow(2).sum()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        assert losses[-1] <= losses[0] * 1.1

    # ── Edge cases ─────────────────────────────────────────────────

    def test_quantize_not_called_raises(self):
        qemb = QuantizedTensorRingEmbedding.__new__(QuantizedTensorRingEmbedding)
        nn.Module.__init__(qemb)
        qemb._quantized = False
        with pytest.raises(RuntimeError, match="quantize"):
            qemb(torch.tensor([0]))

    def test_gradient_not_computed_for_ptq(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        indices = torch.tensor([0, 1, 2])
        output = qemb(indices)
        assert not output.requires_grad
        assert list(qemb.parameters()) == []

    def test_metadata_preserved(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        assert qemb.vocab_size == 100
        assert qemb.embedding_dim == 32
        assert qemb.ring_components == emb.ring_components
        assert qemb.padding_idx == emb.padding_idx

    def test_multiple_forward_calls_consistent(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        indices = torch.tensor([0, 1, 2, 3])
        o1 = qemb(indices)
        o2 = qemb(indices)
        assert torch.equal(o1, o2)

    def test_different_indices_produce_different_outputs(self):
        emb = TensorRingEmbedding(100, 32, rank=4)
        qemb = QuantizedTensorRingEmbedding(emb)
        o_a = qemb(torch.tensor([0]))
        o_b = qemb(torch.tensor([1]))
        assert not torch.allclose(o_a, o_b, atol=1e-4)

    def test_per_channel_dim_nonzero(self):
        from tensor_ring_decomposition.quantization.quantize import _quantize_tensor_per_channel
        t = torch.randn(2, 5, 8)
        # dim=1 (second dimension) — verify shape and no error
        q, scales, zeros = _quantize_tensor_per_channel(t, dim=1)
        assert q.shape == t.shape
        assert scales.shape == (5,)
        # Verify that values are within int8 range (outer bounds)
        assert q.min() >= -128
        assert q.max() <= 127
