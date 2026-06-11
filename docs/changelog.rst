Changelog
=========

0.2.0 (unreleased)
------------------

Enterprise features:

* Distribution-aware training loss (NeurIPS 2025)
* Distribution-aware init method with token-frequency weighting
* Eigenspace overlap score (EOSk)
* Distribution-aware reconstruction error metric
* Automatic gauge fixing during training with configurable interval
* Full PyTorch nn.Embedding API (train/eval overrides, reset_parameters, weight)
* Input validation in forward pass
* Int8 per-channel quantization with forward pass
* 63+ model profiles (25+ new architectures)
* Full config serialization roundtrip with compression_metrics
* TensorRingDDP wrapper for distributed training
* SVD-spectrum-based MSE estimation for autotune
* Fixed optimal_rank MSE formula
* Benchmarks script

0.1.0
-----

Initial release:

* Tensor Ring Decomposition for embedding compression
* Basic init methods (uniform, normal, kaiming, SVD, TR-SVD)
* Gauge fixing (QR/RQ orthogonalization)
* Eval cache for fast inference
* HuggingFace integration
* Safe serialization with safetensors + HMAC
* Model registry with 38 built-in profiles
* ONNX/TorchScript export
