Profiling and Metrics
=====================

Compression Tracker
-------------------

Logs compression metrics during training with configurable interval.

.. code-block:: python

   from tensor_ring_decomposition import CompressionTracker

   tracker = CompressionTracker(log_interval=10)
   tracker.log_metrics(epoch=1, step=50, recon_error=0.05, compression_ratio=50.0)

Quality Gate
------------

Validates that compression quality meets thresholds.

.. code-block:: python

   from tensor_ring_decomposition import QualityGate

   gate = QualityGate(max_recon_error=0.1, min_compression=10.0)
   assert gate.check(recon_error=0.05, compression_ratio=50.0)

TensorRing Callback
-------------------

Callback interface for monitoring training progress.

.. code-block:: python

   from tensor_ring_decomposition import TensorRingCallback

   class MyCallback(TensorRingCallback):
       def on_epoch_end(self, epoch, logs=None):
           print(f"Epoch {epoch} done")

Intrinsic Evaluation
--------------------

.. code-block:: python

   emb = TensorRingEmbedding.from_pretrained(original_matrix, rank=8)

   # Standard reconstruction error
   error = emb.reconstruction_error(original_matrix)

   # Distribution-aware reconstruction error (NeurIPS 2025)
   da_error = emb.distribution_aware_reconstruction_error(
       original_matrix, input_probs=token_frequencies
   )

   # Eigenspace overlap score
   eos = emb.eigenspace_overlap_score(original_matrix, k=10)
