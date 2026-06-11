Tensor Ring Decomposition for Embeddings
=========================================

Drop-in replacement for ``nn.Embedding`` using Tensor Ring Decomposition (TRD),
achieving 10--100x parameter reduction with minimal accuracy loss.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   api
   models
   quantization
   ddp
   export
   profiling
   testing
   changelog


Features
--------

* **10-100x compression** of embedding tables via Tensor Ring Decomposition
* **Gradient-safe eval cache**: :code:`to_eval_mode()` precomputes embedding contraction
* **Multiple init methods**: uniform, normal, kaiming, SVD, TR-SVD, distribution-aware
* **Gauge fixing**: QR-based orthogonalization to prevent scale drift
* **Safe serialization**: safetensors + HMAC-verified manifest
* **HuggingFace integration**: replace embeddings in transformer models
* **Int8 quantization**: post-training per-channel quantization
* **Autotuning**: automatic rank selection from compression / parameter / MSE budget
* **Spetral regularization**: spectral norm monitoring and regularization
* **Distribution-aware training**: NeurIPS 2025-style :math:`||(W-\hat{W})\Sigma^{1/2}||_F` loss
* **Intrinsic evaluation**: eigenspace overlap, distribution-aware reconstruction error
* **DDP support**: distributed data parallel wrapper with gradient sync
* **Export**: ONNX and TorchScript export for production deployment
* **63+ model profiles**: built-in profiles for popular architectures
