# Changelog

## 0.4.0 (unreleased)

### Fixes
- `replace_in_model()`: added skip patterns for position/token-type/segment embeddings
- `trustworthiness()` / `continuity()`: fixed cross-space distance comparison and 1-indexed rank normalization
- `truncate_ranks()`: use `torch.svd_lowrank` for large matrices
- `load_from_transformers()`: attempt embedding-only download before loading full model
- `distribution_aware_reconstruction_error_v2()`: refactored into 4 helper methods (reduced nesting depth)
- `_train_to_matrix()` / `_sample_reconstruct()`: deduplicated forward logic via `_compute_forward()`
- `NonNegativeClamp.backward`: removed wasteful `.clone()` (already applied in prior audit)

### New Features
- `tie_weights(linear_layer)`: weight tying for tied input/output embeddings
- `distill(teacher_matrix, ...)`: knowledge distillation from dense teacher to TR student
- `adjust_rank(new_rank)`: progressive rank adjustment with warm-start (pad or truncate)
- `to_onnx_runtime(path)`: export to ONNX and return ONNX Runtime InferenceSession
- Added `.hf_cache/` to `.gitignore`

### Housekeeping
- Removed unused `_emb_strides` dead storage from both embeddings
- Removed dead `spectral_gap` local-search loop
- Version bump to 0.4.0

## 0.3.0 (released)

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
