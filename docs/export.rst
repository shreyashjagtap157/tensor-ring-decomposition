Export
======

Export TR embeddings to ONNX or TorchScript for production deployment.

.. code-block:: python

   from tensor_ring_decomposition import TensorRingEmbedding, ExportFormat

   emb = TensorRingEmbedding(50000, 768, rank=8)

   # Export to ONNX
   path = emb.export("model_embedding", format=ExportFormat.ONNX)

   # Export to TorchScript
   path = emb.export("model_embedding", format=ExportFormat.TORCHSCRIPT)

   # Load a TorchScript-exported model
   loaded = TensorRingEmbedding.load_exported("model_embedding.torchscript")

Parameters
----------

* ``path``: Output file path (extension added automatically)
* ``format``: ``ExportFormat.ONNX`` or ``ExportFormat.TORCHSCRIPT``
* ``batch_size``: Static batch size for tracing (default 1)
* ``seq_len``: Sequence length for tracing (default 128)
* ``dynamic_axes``: Whether to use dynamic batch/sequence axes (ONNX only)
* ``input_dtype``: Input tensor dtype (default ``torch.long``)
