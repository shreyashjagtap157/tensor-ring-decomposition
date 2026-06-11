# Changelog

## 0.2.0 (unreleased)

### Enterprise Features
- Distribution-aware training loss (NeurIPS 2025) — `||(W-Ŵ)Σ^{1/2}||_F`
- Distribution-aware init method with token-frequency weighting
- Eigenspace overlap score (EOSk) for intrinsic evaluation
- Automatic gauge fixing during training with configurable interval
- Full PyTorch `nn.Embedding` API (`train()`/`eval()` overrides, `reset_parameters()`, `weight` property, `num_embeddings`)
- Input validation in forward pass (configurable)
- Int8 per-channel quantization with proper forward pass
- 63+ model profiles (25+ new architectures: GPT-J, GPT-NeoX, Pythia, StarCoder, DBRX, Command R, Jamba, MPT, Yi, DeepSeek, StableLM, Zephyr, Solar, XVERSE, Qwen 2.5, LLaMA 3.1/3.2, Granite)
- Full config serialization roundtrip with compression metrics in manifest
- `TensorRingDDP` wrapper for distributed training with gradient sync
- Fixed SVD-spectrum-based MSE estimation in `autotune()`
- Fixed `optimal_rank()` MSE formula — parameter fraction proportional to squared-error budget
- 242-test enterprise test suite (303 total passing, 34 skipped)

### Benchmarking
- `benchmarks/benchmark_tr_vs_dense.py` — comprehensive latency/memory/compression comparison

## 0.1.0

### Initial Release
- Tensor Ring Decomposition for embedding compression
- Basic init methods (uniform, normal, kaiming, SVD, TR-SVD)
- Gauge fixing (QR/RQ orthogonalization)
- Eval cache for fast inference
- HuggingFace integration with `HuggingFaceTensorRingEmbedding`
- Safe serialization with safetensors + HMAC-verified manifest
- Model registry with 38 built-in profiles
- ONNX/TorchScript export
- Monitoring (CompressionTracker, QualityGate, TensorRingCallback)
