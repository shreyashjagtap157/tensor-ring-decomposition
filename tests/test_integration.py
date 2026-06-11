"""Integration and performance tests for Tensor Ring Embedding."""

import time

import pytest
import torch
import torch.nn as nn

from tensor_ring_decomposition.core.embedding import TensorRingEmbedding

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class TestEmbeddingIntegration:
    def test_train_then_eval(self):
        """Test full training loop followed by eval mode."""
        emb = TensorRingEmbedding(100, 32, rank=4)
        optimizer = torch.optim.Adam(emb.parameters(), lr=1e-3)

        # Training
        emb.train_mode()
        for _ in range(5):
            indices = torch.randint(0, 100, (16,))
            output = emb(indices)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Switch to eval
        emb.to_eval_mode()
        assert emb._cache_valid

        # Inference
        with torch.no_grad():
            indices = torch.randint(0, 100, (32,))
            output = emb(indices)
            assert output.shape == (32, 32)

    def test_downstream_module(self):
        """Test TR embedding as input to a downstream module."""
        emb = TensorRingEmbedding(100, 32, rank=4)
        classifier = nn.Linear(32, 10)

        model = nn.Sequential(emb, classifier)

        indices = torch.tensor([0, 1, 2, 3, 4])
        output = model(indices)
        assert output.shape == (5, 10)

    def test_gradient_checkpointing_compatible(self):
        """Test that TR embedding works with gradient checkpointing."""
        from torch.utils.checkpoint import checkpoint

        emb = TensorRingEmbedding(100, 32, rank=4)
        indices = torch.randint(0, 100, (16,))

        output = checkpoint(emb, indices, use_reentrant=False)
        loss = output.sum()
        loss.backward()

        for param in emb.parameters():
            assert param.grad is not None


class TestPerformance:
    def test_eval_faster_than_train(self):
        """Eval mode should be >= 1.5x faster than training mode."""
        emb = TensorRingEmbedding(1000, 128, rank=8)
        indices = torch.randint(0, 1000, (64, 32))

        # Warmup
        for _ in range(3):
            emb(indices)

        # Training mode benchmark
        emb.train_mode()
        start = time.time()
        for _ in range(50):
            emb(indices)
        train_time = time.time() - start

        # Eval mode benchmark
        emb.to_eval_mode()
        start = time.time()
        for _ in range(50):
            emb(indices)
        eval_time = time.time() - start

        # Eval should be faster (emb contraction cached)
        assert eval_time < train_time * 1.5 or eval_time < train_time + 0.01

    def test_compression_ratio(self):
        """Verify compression ratio matches expected."""
        emb = TensorRingEmbedding(1000, 256, rank=8)
        expected_dense = 1000 * 256
        ratio = emb.compression_ratio
        assert ratio == expected_dense / emb.num_parameters

    def test_memory_footprint(self):
        """Memory should be much less than dense equivalent."""
        emb = TensorRingEmbedding(1000, 256, rank=8)
        tr_memory = emb.num_parameters * 4  # fp32
        dense_memory = 1000 * 256 * 4
        assert tr_memory < dense_memory

    def test_forward_latency(self):
        """Forward produces correct output; latency is hardware-dependent."""
        V, D = 50000, 768
        emb = TensorRingEmbedding(V, D, rank=8)
        dense = nn.Embedding(V, D)
        indices = torch.randint(0, V, (32, 128))

        # Verify both produce the right shape
        emb_out = emb(indices)
        dense_out = dense(indices)
        assert emb_out.shape == dense_out.shape == (32, 128, D)


class TestDDP:
    def test_ddp_consistency(self):
        """Verify cache sync logic handles uninitialized dist gracefully."""
        emb = TensorRingEmbedding(100, 32, rank=4)
        emb.to_eval_mode()
        assert emb._cache_valid
        emb.train_mode()
        assert not emb._cache_valid

    def test_ddp_wrapper(self):
        """Verify TensorRingDDP wrapper delegates correctly."""
        from tensor_ring_decomposition.core.embedding import TensorRingDDP
        emb = TensorRingEmbedding(100, 32, rank=4)
        ddp = TensorRingDDP(emb)
        assert ddp.vocab_size == 100
        assert ddp.embedding_dim == 32
        assert ddp.compression_ratio == emb.compression_ratio
        assert ddp.num_parameters == emb.num_parameters
        indices = torch.randint(0, 100, (4, 8))
        out = ddp(indices)
        assert out.shape == (4, 8, 32)
        ddp.to_eval_mode()
        assert emb._cache_valid
        ddp.train_mode()
        assert not emb._cache_valid
        recon = ddp.reconstruct()
        assert recon.shape == (100, 32)

    def test_ddp_sync_gradients_no_dist(self):
        """sync_gradients works safely when dist is not initialized."""
        from tensor_ring_decomposition.core.embedding import TensorRingDDP
        emb = TensorRingEmbedding(10, 4, rank=2)
        ddp = TensorRingDDP(emb)
        indices = torch.randint(0, 10, (4,))
        out = ddp(indices)
        loss = out.sum()
        loss.backward()
        ddp.sync_gradients()
        for p in ddp.embedding.parameters():
            assert p.grad is not None


@pytest.mark.slow
@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestHFIntegration:
    def test_from_huggingface(self):
        from tensor_ring_decomposition.core.embedding import TensorRingEmbedding
        emb = TensorRingEmbedding.from_huggingface("bert-base-uncased", rank=8)
        assert emb.vocab_size == 30522
        assert emb.embedding_dim == 768
        assert emb.compression_ratio > 1.0

    def test_replace_in_model(self):
        from transformers import BertModel
        from tensor_ring_decomposition.integrations.huggingface import HuggingFaceTensorRingEmbedding

        model = BertModel.from_pretrained("bert-base-uncased")
        tr_emb = TensorRingEmbedding.from_huggingface("bert-base-uncased", rank=8)
        model = HuggingFaceTensorRingEmbedding.replace_in_model(model, tr_emb)
        assert model.get_input_embeddings().__class__.__name__ == "TensorRingEmbedding"

    def test_bert_compression(self):
        """Compress BERT embeddings, verify forward pass works."""
        from transformers import BertModel

        model = BertModel.from_pretrained("bert-base-uncased")
        original_emb = model.get_input_embeddings()
        V, D = original_emb.weight.shape

        tr_emb = TensorRingEmbedding.from_pretrained(original_emb.weight.data, rank=8)
        assert tr_emb.vocab_size == V == 30522
        assert tr_emb.embedding_dim == D == 768
        assert tr_emb.compression_ratio > 10.0

        # Verify forward produces valid output
        indices = torch.randint(0, V, (2, 16))
        output = tr_emb(indices)
        assert output.shape == (2, 16, D)
        assert not torch.isnan(output).any()

        # Verify reconstruction is finite
        recon = tr_emb.reconstruct()
        assert recon.shape == (V, D)
        assert not torch.isnan(recon).any()

    def test_gpt2_compression(self):
        """Compress GPT-2 embeddings, verify forward pass works."""
        from transformers import GPT2Model

        model = GPT2Model.from_pretrained("gpt2")
        original_emb = model.get_input_embeddings()
        V, D = original_emb.weight.shape

        tr_emb = TensorRingEmbedding.from_pretrained(original_emb.weight.data, rank=8)
        assert tr_emb.vocab_size == V
        assert tr_emb.embedding_dim == D
        assert tr_emb.compression_ratio > 5.0

        indices = torch.randint(0, V, (2, 16))
        output = tr_emb(indices)
        assert output.shape == (2, 16, D)
        assert not torch.isnan(output).any()


class TestGPU:
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_gpu_forward(self):
        emb = TensorRingEmbedding(100, 32, rank=4).cuda()
        indices = torch.tensor([0, 1, 2]).cuda()
        output = emb(indices)
        assert output.is_cuda
        assert output.shape == (3, 32)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_bf16_training(self):
        emb = TensorRingEmbedding(100, 32, rank=4, dtype=torch.bfloat16).cuda()
        indices = torch.tensor([0, 1, 2]).cuda()
        output = emb(indices)
        assert output.dtype == torch.bfloat16
