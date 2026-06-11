Quick Start
===========

Basic usage
-----------

.. code-block:: python

   import torch
   from tensor_ring_decomposition import TensorRingEmbedding

   # Explicit rank
   emb = TensorRingEmbedding(50000, 768, rank=8)

   # Target compression ratio
   emb = TensorRingEmbedding.from_compression_ratio(50000, 768, ratio=50)

   # From pretrained dense embedding
   emb = TensorRingEmbedding.from_pretrained(dense_weight, rank=8)

   # Use like nn.Embedding
   indices = torch.randint(0, 50000, (4, 16))
   output = emb(indices)  # (4, 16, 768)

Compress from HuggingFace
-------------------------

.. code-block:: python

   emb = TensorRingEmbedding.from_huggingface("bert-base-uncased", rank=8)
   print(f"Compression: {emb.compression_ratio:.1f}x")

Replace in a transformer model
------------------------------

.. code-block:: python

   from transformers import BertModel
   from tensor_ring_decomposition import HuggingFaceTensorRingEmbedding

   model = BertModel.from_pretrained("bert-base-uncased")
   tr_emb = TensorRingEmbedding.from_huggingface("bert-base-uncased", rank=8)
   model = HuggingFaceTensorRingEmbedding.replace_in_model(model, tr_emb)

High-level compress API
-----------------------

.. code-block:: python

   from tensor_ring_decomposition import compress

   # From a file
   emb = compress("path/to/model.safetensors", target_compression=50)

   # From HuggingFace
   emb = compress("bert-base-uncased", target_compression=50)

   # With autotuning
   emb = compress("bert-base-uncased", target_compression=50, autotune=True)
