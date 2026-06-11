Quantization
============

Post-training int8 quantization for TR embeddings with AMX matmul support.

.. code-block:: python

   from tensor_ring_decomposition import TensorRingEmbedding, QuantizedTensorRingEmbedding

   emb = TensorRingEmbedding(50000, 768, rank=8)
   q_emb = QuantizedTensorRingEmbedding(emb, per_channel=True)

   # Forward pass with int8 gather and float dequant matmul
   indices = torch.randint(0, 50000, (4, 16))
   output = q_emb(indices)

   # Eval caching
   q_emb.to_eval_mode()

   # Check compression
   print(f"Quantized compression: {q_emb.compression_ratio:.1f}x")
   print(f"Bits per parameter: {q_emb.bits_per_parameter:.1f}")

Features
--------

* Per-channel or per-tensor int8 quantization
* Symmetric quantization (zero_point=0) for best performance
* int8 gather with on-the-fly float dequantization
* Eval mode caching for fast inference
* Compression ratio and bits-per-parameter metrics
